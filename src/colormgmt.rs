//! Client-side `wp_color_management_v1` wiring (hand-rolled — SCTK has no
//! helper for it). Collects the compositor's advertised capabilities, turns
//! a [`ColorEncoding`] into a parametric image description, and tags
//! surfaces with it.

use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use wayland_client::{globals::GlobalList, Connection, Dispatch, Proxy, QueueHandle, WEnum};
use wayland_protocols::wp::color_management::v1::client::{
    wp_color_management_surface_feedback_v1::{self, WpColorManagementSurfaceFeedbackV1},
    wp_color_management_surface_v1::WpColorManagementSurfaceV1,
    wp_color_manager_v1::{
        self, Feature, Primaries, RenderIntent, TransferFunction, WpColorManagerV1,
    },
    wp_image_description_creator_params_v1::WpImageDescriptionCreatorParamsV1,
    wp_image_description_info_v1::{self, WpImageDescriptionInfoV1},
    wp_image_description_v1::{self, WpImageDescriptionV1},
};

use crate::app::App;
use crate::cli::Intent;
use crate::color::{ColorEncoding, PrimaryVolume, Tf};

/// The bound color manager plus everything it advertised before `done`.
pub struct ColorState {
    pub manager: WpColorManagerV1,
    tfs: Vec<TransferFunction>,
    primaries: Vec<Primaries>,
    features: Vec<Feature>,
    intents: Vec<RenderIntent>,
    pub done: bool,
}

impl ColorState {
    pub fn bind(globals: &GlobalList, qh: &QueueHandle<App>) -> Option<ColorState> {
        let manager = globals
            .bind::<WpColorManagerV1, App, ()>(qh, 1..=1, ())
            .ok()?;
        Some(ColorState {
            manager,
            tfs: Vec::new(),
            primaries: Vec::new(),
            features: Vec::new(),
            intents: Vec::new(),
            done: false,
        })
    }

    fn feature(&self, f: Feature) -> bool {
        self.features.contains(&f)
    }

    /// Create a parametric image description for `enc`, degrading where
    /// harmless and failing where rendering would be wrong:
    /// - luminances are dropped (with a warning) if `set_luminances` is
    ///   missing — defaults are close enough for SDR;
    /// - an unsupported TF or primary set is an error — mis-tagged pixels
    ///   defeat the point of this tool.
    pub fn create_description(
        &self,
        qh: &QueueHandle<App>,
        enc: &ColorEncoding,
    ) -> Result<DescriptionHandle> {
        if !self.feature(Feature::Parametric) {
            bail!("compositor lacks parametric image descriptions");
        }
        let tf = protocol_tf(enc.tf);
        if !self.tfs.contains(&tf) {
            // main's capability-adaptation pass re-encodes unsupported TFs
            // before descriptions are created; reaching this is a bug there.
            bail!("compositor does not support transfer function {tf:?}");
        }

        let status: DescStatus = Arc::new(Mutex::new(Status::Pending));
        let params = self.manager.create_parametric_creator(qh, ());
        params.set_tf_named(tf);

        match enc.primaries {
            PrimaryVolume::Srgb => self.set_named_primaries(&params, Primaries::Srgb)?,
            PrimaryVolume::DisplayP3 => self.set_named_primaries(&params, Primaries::DisplayP3)?,
            PrimaryVolume::Bt2020 => self.set_named_primaries(&params, Primaries::Bt2020)?,
            PrimaryVolume::Custom(c) => {
                if !self.feature(Feature::SetPrimaries) {
                    bail!("image needs custom primaries but compositor lacks set_primaries");
                }
                let f = |v: f64| (v * 1_000_000.0).round() as i32;
                params.set_primaries(
                    f(c.r.0),
                    f(c.r.1),
                    f(c.g.0),
                    f(c.g.1),
                    f(c.b.0),
                    f(c.b.1),
                    f(c.w.0),
                    f(c.w.1),
                );
            }
        }

        if let Some(lum) = enc.luminances {
            if self.feature(Feature::SetLuminances) {
                params.set_luminances(
                    (lum.min * 10_000.0).round() as u32,
                    lum.max.round() as u32,
                    lum.reference.round() as u32,
                );
            } else {
                tracing::warn!("compositor lacks set_luminances; using TF defaults");
            }
        }

        let object = params.create(qh, status.clone());
        Ok(DescriptionHandle { object, status })
    }

    fn set_named_primaries(
        &self,
        params: &WpImageDescriptionCreatorParamsV1,
        p: Primaries,
    ) -> Result<()> {
        if self.primaries.contains(&p) {
            params.set_primaries_named(p);
            return Ok(());
        }
        // Fall back to explicit chromaticities for a named set the
        // compositor didn't list (prism lists all three we use).
        if !self.feature(Feature::SetPrimaries) {
            bail!("compositor supports neither {p:?} nor set_primaries");
        }
        let c = match p {
            Primaries::Srgb => crate::color::SRGB_CHROMA,
            Primaries::DisplayP3 => crate::color::DISPLAY_P3_CHROMA,
            Primaries::Bt2020 => crate::color::BT2020_CHROMA,
            _ => bail!("unmapped named primaries {p:?}"),
        };
        let f = |v: f64| (v * 1_000_000.0).round() as i32;
        params.set_primaries(
            f(c.r.0),
            f(c.r.1),
            f(c.g.0),
            f(c.g.1),
            f(c.b.0),
            f(c.b.1),
            f(c.w.0),
            f(c.w.1),
        );
        Ok(())
    }

    /// Subscribe to the preferred image description for `surface` (which
    /// follows the output it sits on) and fire the first query. The
    /// resolved target luminance lands in [`App::tone_targets`] keyed by
    /// `output`; `preferred_changed` re-queries.
    pub fn watch_preferred(
        &self,
        qh: &QueueHandle<App>,
        surface: &wayland_client::protocol::wl_surface::WlSurface,
        output: String,
    ) -> WpColorManagementSurfaceFeedbackV1 {
        let fb = self
            .manager
            .get_surface_feedback(surface, qh, FeedbackData(output));
        query_preferred(&fb, qh);
        fb
    }

    /// Wrap `surface` and attach `desc` with the chosen intent.
    pub fn tag_surface(
        &self,
        qh: &QueueHandle<App>,
        surface: &wayland_client::protocol::wl_surface::WlSurface,
        desc: &DescriptionHandle,
        intent: Intent,
    ) -> WpColorManagementSurfaceV1 {
        let intent = match intent {
            Intent::Perceptual => RenderIntent::Perceptual,
            Intent::Relative => RenderIntent::Relative,
            Intent::Absolute => RenderIntent::Absolute,
        };
        let cm = self.manager.get_surface(surface, qh, ());
        cm.set_image_description(&desc.object, intent);
        cm
    }
}

fn protocol_tf(tf: Tf) -> TransferFunction {
    match tf {
        Tf::Srgb => TransferFunction::Srgb,
        Tf::Gamma22 => TransferFunction::Gamma22,
        Tf::Bt1886 => TransferFunction::Bt1886,
        Tf::Pq => TransferFunction::St2084Pq,
        Tf::Linear => TransferFunction::ExtLinear,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    Pending,
    Ready,
    Failed(String),
}

type DescStatus = Arc<Mutex<Status>>;

/// A created image description plus its (event-driven) readiness. The
/// protocol forbids using a description before `ready`, so callers
/// roundtrip until [`Self::status`] leaves `Pending`.
pub struct DescriptionHandle {
    pub object: WpImageDescriptionV1,
    status: DescStatus,
}

impl DescriptionHandle {
    pub fn status(&self) -> Status {
        self.status.lock().unwrap().clone()
    }
}

impl Dispatch<WpColorManagerV1, ()> for App {
    fn event(
        state: &mut App,
        _: &WpColorManagerV1,
        event: wp_color_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<App>,
    ) {
        let Some(color) = state.color.as_mut() else {
            return;
        };
        use wp_color_manager_v1::Event;
        match event {
            Event::SupportedTfNamed {
                tf: WEnum::Value(tf),
            } => color.tfs.push(tf),
            Event::SupportedPrimariesNamed {
                primaries: WEnum::Value(p),
            } => color.primaries.push(p),
            Event::SupportedFeature {
                feature: WEnum::Value(f),
            } => color.features.push(f),
            Event::SupportedIntent {
                render_intent: WEnum::Value(i),
            } => color.intents.push(i),
            Event::Done => color.done = true,
            _ => {}
        }
    }
}

impl Dispatch<WpImageDescriptionV1, DescStatus> for App {
    fn event(
        _: &mut App,
        desc: &WpImageDescriptionV1,
        event: wp_image_description_v1::Event,
        status: &DescStatus,
        _: &Connection,
        _: &QueueHandle<App>,
    ) {
        use wp_image_description_v1::Event;
        match event {
            Event::Ready { .. } => *status.lock().unwrap() = Status::Ready,
            Event::Failed { cause, msg } => {
                tracing::error!(
                    ?cause,
                    msg,
                    id = desc.id().protocol_id(),
                    "image description failed"
                );
                *status.lock().unwrap() = Status::Failed(msg);
            }
            _ => {}
        }
    }
}

// ---- preferred-description feedback (for --tone-map auto) ----

/// User data carrying the output name a feedback/info object reports for.
#[derive(Debug, Clone)]
pub struct FeedbackData(pub String);

/// Issue (or re-issue) the preferred-description query on a feedback
/// object: get_preferred_parametric → get_information; the description
/// object itself is discarded once the info request is in flight.
fn query_preferred(fb: &WpColorManagementSurfaceFeedbackV1, qh: &QueueHandle<App>) {
    let output = fb.data::<FeedbackData>().unwrap().clone();
    let desc = fb.get_preferred_parametric(qh, output.clone());
    desc.get_information(qh, output);
    desc.destroy();
}

impl Dispatch<WpColorManagementSurfaceFeedbackV1, FeedbackData> for App {
    fn event(
        _: &mut App,
        fb: &WpColorManagementSurfaceFeedbackV1,
        event: wp_color_management_surface_feedback_v1::Event,
        _: &FeedbackData,
        _: &Connection,
        qh: &QueueHandle<App>,
    ) {
        use wp_color_management_surface_feedback_v1::Event;
        if matches!(event, Event::PreferredChanged { .. }) {
            // Output HDR mode changed (or the surface moved outputs);
            // re-resolve. Already-treated images are not re-targeted
            // mid-flight — restart picks up the new target.
            query_preferred(fb, qh);
        }
    }
}

// The preferred description object: ready/failed don't matter, only the
// info derived from it. Distinguished from owned descriptions by the
// user-data type.
impl Dispatch<WpImageDescriptionV1, FeedbackData> for App {
    fn event(
        _: &mut App,
        _: &WpImageDescriptionV1,
        _: wp_image_description_v1::Event,
        _: &FeedbackData,
        _: &Connection,
        _: &QueueHandle<App>,
    ) {
    }
}

impl Dispatch<WpImageDescriptionInfoV1, FeedbackData> for App {
    fn event(
        state: &mut App,
        _: &WpImageDescriptionInfoV1,
        event: wp_image_description_info_v1::Event,
        data: &FeedbackData,
        _: &Connection,
        qh: &QueueHandle<App>,
    ) {
        use wp_image_description_info_v1::Event;
        let output = &data.0;
        match event {
            Event::TargetMaxCll { max_cll } => {
                state.pending_targets.entry(output.clone()).or_default().0 = Some(max_cll as f64);
            }
            Event::TargetLuminance { max_lum, .. } => {
                state.pending_targets.entry(output.clone()).or_default().1 = Some(max_lum as f64);
            }
            Event::Luminances { reference_lum, .. } => {
                // The preferred encoding's reference white: the luminance shader
                // value 1.0 maps to under the anchored intent.
                state.pending_targets.entry(output.clone()).or_default().2 =
                    Some(reference_lum as f64);
            }
            Event::Done => {
                let (cll, lum_max, reference) =
                    state.pending_targets.remove(output).unwrap_or_default();
                // Master against target_luminance.max — the mastering-display
                // peak the compositor advertises as the value to tone-map to
                // (prism's `advertised-peak-nits`, deliberately decoupled from
                // the HDR_OUTPUT_METADATA signaling). target_max_cll is that
                // signaling/panel value (often the marketing peak) and is a
                // fallback only, used when target_luminance is somehow absent.
                let Some(max) = lum_max.or(cll) else {
                    tracing::warn!(output, "preferred description had no target luminance");
                    return;
                };
                let reference = reference.unwrap_or(crate::app::DEFAULT_OUTPUT_LUM.reference);
                let lum = crate::app::OutputLum { reference, max };
                let changed = state.output_lums.get(output).map(|l| (l.reference, l.max))
                    != Some((reference, max));
                tracing::info!(
                    output,
                    ref_nits = reference,
                    target_nits = max,
                    "output advertised luminance"
                );
                state.output_lums.insert(output.clone(), lum);
                // Static shaders baked stale luminance into their one frame
                // (caps may arrive after the first render); redraw them.
                // Animated shaders pick the new values up on their next frame.
                if changed {
                    state.redraw_static_shaders(qh);
                }
            }
            _ => {}
        }
    }
}

// No events on these.
wayland_client::delegate_noop!(App: ignore WpImageDescriptionCreatorParamsV1);
wayland_client::delegate_noop!(App: ignore WpColorManagementSurfaceV1);
