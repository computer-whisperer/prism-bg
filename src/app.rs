//! The Wayland application: one background layer surface per matched
//! output, each a solid-color parent (1×1 buffer, viewport-stretched) with
//! an optional image subsurface on top. The compositor does all scaling
//! (viewport) and all color conversion (each image surface is tagged with
//! its source description via `wp_color_management_v1`).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use smithay_client_toolkit::reexports::calloop::{
    timer::{TimeoutAction, Timer},
    LoopHandle,
};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, Region},
    delegate_compositor, delegate_layer, delegate_output, delegate_registry, delegate_shm,
    delegate_simple, delegate_subcompositor,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
        WaylandSurface,
    },
    shm::{
        slot::{Buffer, SlotPool},
        Shm, ShmHandler,
    },
    subcompositor::SubcompositorState,
};
use wayland_client::{
    globals::GlobalList,
    protocol::{wl_output, wl_shm, wl_subsurface::WlSubsurface, wl_surface::WlSurface},
    Connection, QueueHandle,
};
use wayland_protocols::wp::alpha_modifier::v1::client::{
    wp_alpha_modifier_surface_v1::WpAlphaModifierSurfaceV1, wp_alpha_modifier_v1::WpAlphaModifierV1,
};
use wayland_protocols::wp::color_management::v1::client::wp_color_management_surface_v1::WpColorManagementSurfaceV1;
use wayland_protocols::wp::viewporter::client::{
    wp_viewport::WpViewport, wp_viewporter::WpViewporter,
};

use crate::cli::{Args, Color, Intent, Mode, OutputSpec};
use crate::colormgmt::{ColorState, DescriptionHandle, Status};
use crate::decode::DecodedImage;
use crate::playlist::Playlist;
use crate::surfaces::{place, upload, upload_tiled, WireRgb8};

/// Image identity for deduplication: path + effective luminance treatment
/// (the same file treated differently for different outputs is different
/// pixels).
pub type ImageKey = (PathBuf, Option<(u64, u64, u64)>);

/// The effective treatment for a spec once `--tone-map auto` has been
/// resolved to concrete nits (or `None` when not requested / unavailable).
pub fn resolve_treatment(
    spec: &OutputSpec,
    tone_nits: Option<f64>,
) -> Option<crate::color::LuminanceControl> {
    let t = crate::color::LuminanceControl {
        tone_map: tone_nits,
        ..spec.luminance.unwrap_or_default()
    };
    (!t.is_empty()).then_some(t)
}

/// A decoded, treated, capability-adapted image plus its (shared)
/// compositor-side description.
pub struct LoadedImage {
    pub image: Arc<DecodedImage>,
    pub description: Option<DescriptionHandle>,
}

struct ImagePart {
    subsurface: WlSubsurface,
    surface: WlSurface,
    viewport: WpViewport,
    /// `wp_alpha_modifier_surface_v1`, created only for parts attached
    /// with a crossfade (the multiplier ramps 0 → max).
    alpha: Option<WpAlphaModifierSurfaceV1>,
    /// Keeps the color-management surface wrapper (and with it the
    /// description binding) alive.
    cm: Option<WpColorManagementSurfaceV1>,
    image: Arc<DecodedImage>,
    mode: Mode,
    /// What this part shows, for change detection in [`App::service`].
    key: ImageKey,
}

impl Drop for ImagePart {
    /// wayland-client proxies don't send destructors on Drop; rotation
    /// (and output removal) need the subsurface tree actually gone.
    /// wl_subsurface.destroy unmaps the child immediately.
    fn drop(&mut self) {
        if let Some(cm) = &self.cm {
            cm.destroy();
        }
        // The protocol requires destroying this before the wl_surface.
        if let Some(alpha) = &self.alpha {
            alpha.destroy();
        }
        self.viewport.destroy();
        self.subsurface.destroy();
        self.surface.destroy();
    }
}

/// An in-flight crossfade: the outgoing part stays mapped (stacked below
/// the incoming one) while the incoming surface's alpha multiplier ramps
/// up, then both are dropped here at completion. The compositor does the
/// blend — prism in linear fp16, so the dissolve is gamma-correct.
struct Fade {
    start: Instant,
    duration: Duration,
    /// Outgoing subsurface tree, destroyed when the fade finishes.
    _old_part: ImagePart,
    /// The outgoing part's pixels must outlive its surface.
    _old_buffer: Option<Buffer>,
}

struct Wallpaper {
    output: wl_output::WlOutput,
    name: String,
    spec: OutputSpec,
    layer: LayerSurface,
    viewport: WpViewport,
    color: Color,
    /// Built lazily by [`App::service`] once the tone target (if `auto`)
    /// and the image description are resolved.
    image_part: Option<ImagePart>,
    /// Crossfade in progress (`--fade`), stepped by frame callbacks on
    /// the layer surface.
    fade: Option<Fade>,
    /// Image preparation failed; don't retry every service pass.
    broken: bool,
    /// Keeps the preferred-description subscription alive (auto mode).
    _feedback: Option<
        wayland_protocols::wp::color_management::v1::client
            ::wp_color_management_surface_feedback_v1::WpColorManagementSurfaceFeedbackV1,
    >,
    /// Logical size from the last configure; 0 until configured.
    size: (u32, u32),
    scale: i32,
    color_buffer: Option<Buffer>,
    image_buffer: Option<Buffer>,
}

pub struct App {
    pub registry_state: RegistryState,
    pub output_state: OutputState,
    pub shm: Shm,
    pub pool: SlotPool,
    pub compositor: CompositorState,
    pub subcompositor: SubcompositorState,
    pub layer_shell: LayerShell,
    pub viewporter: SimpleViewporter,
    /// `wp_alpha_modifier_v1` if the compositor has it; `--fade` degrades
    /// to a hard cut without it.
    pub alpha_modifier: Option<WpAlphaModifierV1>,
    pub color: Option<ColorState>,
    pub intent: Intent,
    pub specs: Vec<OutputSpec>,
    /// Raw decoded images by path, kept for deriving treated variants
    /// (per-output tone targets, hotplug).
    pub raw_images: HashMap<PathBuf, Arc<DecodedImage>>,
    /// Treated + capability-adapted images by (path, treatment).
    pub images: HashMap<ImageKey, LoadedImage>,
    /// Resolved per-output tone-map targets (nits), from the preferred
    /// image description's target_max_cll / target_luminance.
    pub tone_targets: HashMap<String, f64>,
    /// In-flight info collection per output: (target_max_cll,
    /// target_luminance.max).
    pub pending_targets: HashMap<String, (Option<f64>, Option<f64>)>,
    /// Rotation state per `--image-list` spec group, indexed by
    /// `OutputSpec::playlist`. Advanced by per-playlist timers in `main`.
    pub playlists: Vec<Playlist>,
    /// For inserting the fade ticker when a crossfade starts.
    loop_handle: LoopHandle<'static, App>,
    /// A fade ticker is currently inserted (one drives all fades).
    fade_timer_armed: bool,
    wallpapers: Vec<Wallpaper>,
}

/// Crossfade tick period, ~60 Hz.
const FADE_TICK: Duration = Duration::from_millis(16);

pub type SimpleViewporter = smithay_client_toolkit::registry::SimpleGlobal<WpViewporter, 1>;

impl App {
    pub fn new(
        globals: &GlobalList,
        qh: &QueueHandle<App>,
        args: &Args,
        raw_images: HashMap<PathBuf, Arc<DecodedImage>>,
        playlists: Vec<Playlist>,
        loop_handle: LoopHandle<'static, App>,
    ) -> Result<App> {
        let compositor =
            CompositorState::bind(globals, qh).context("wl_compositor not available")?;
        let subcompositor =
            SubcompositorState::bind(compositor.wl_compositor().clone(), globals, qh)
                .context("wl_subcompositor not available")?;
        let layer_shell = LayerShell::bind(globals, qh).context(
            "zwlr_layer_shell_v1 not available (compositor without layer-shell support?)",
        )?;
        let shm = Shm::bind(globals, qh).context("wl_shm not available")?;
        let pool = SlotPool::new(4096, &shm).context("creating shm pool")?;
        let viewporter =
            SimpleViewporter::bind(globals, qh).context("wp_viewporter not available")?;
        let alpha_modifier: Option<WpAlphaModifierV1> = globals.bind(qh, 1..=1, ()).ok();
        if alpha_modifier.is_none() && args.specs.iter().any(|s| s.fade.is_some()) {
            tracing::warn!(
                "compositor lacks wp_alpha_modifier_v1; --fade disabled (rotation cuts hard)"
            );
        }
        let color = if args.no_color_management {
            None
        } else {
            let c = ColorState::bind(globals, qh);
            if c.is_none() {
                tracing::warn!(
                    "compositor lacks wp_color_management_v1; images will be presented untagged \
                     (assumed sRGB)"
                );
            }
            c
        };

        Ok(App {
            registry_state: RegistryState::new(globals),
            output_state: OutputState::new(globals, qh),
            shm,
            pool,
            compositor,
            subcompositor,
            layer_shell,
            viewporter,
            alpha_modifier,
            color,
            intent: args.intent,
            specs: args.specs.clone(),
            raw_images,
            images: HashMap::new(),
            tone_targets: HashMap::new(),
            pending_targets: HashMap::new(),
            playlists,
            loop_handle,
            fade_timer_armed: false,
            wallpapers: Vec::new(),
        })
    }

    /// Create a wallpaper for `output` if a spec matches and none exists.
    pub fn add_output(&mut self, qh: &QueueHandle<App>, output: wl_output::WlOutput) {
        let info = self.output_state.info(&output);
        let name = info
            .as_ref()
            .and_then(|i| i.name.clone())
            .unwrap_or_default();
        if self.wallpapers.iter().any(|w| w.output == output) {
            return;
        }
        let Some(spec) = crate::cli::spec_for_output(&self.specs, &name) else {
            tracing::debug!(output = name, "no spec matches; skipping");
            return;
        };
        let spec = spec.clone();
        let scale = info.map_or(1, |i| i.scale_factor);

        let surface = self.compositor.create_surface(qh);
        let layer = self.layer_shell.create_layer_surface(
            qh,
            surface,
            Layer::Background,
            Some("wallpaper"),
            Some(&output),
        );
        layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
        layer.set_exclusive_zone(-1);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer.set_size(0, 0);
        let viewport = self.get_viewport(qh, layer.wl_surface());

        // The parent is always opaque (solid color under an image, or the
        // solid color itself).
        if let Ok(region) = Region::new(&self.compositor) {
            region.add(0, 0, i32::MAX, i32::MAX);
            layer
                .wl_surface()
                .set_opaque_region(Some(region.wl_region()));
        }

        // `--tone-map auto` needs the output's preferred description;
        // subscribe before the first service pass.
        let wants_image = (spec.image.is_some() || spec.playlist.is_some())
            && spec.effective_mode() != Mode::SolidColor;
        let feedback = match (&self.color, spec.tone_map, wants_image) {
            (Some(color), Some(crate::cli::ToneMap::Auto), true) => {
                Some(color.watch_preferred(qh, layer.wl_surface(), name.clone()))
            }
            (None, Some(crate::cli::ToneMap::Auto), true) => {
                tracing::warn!(
                    output = name,
                    "--tone-map auto needs wp_color_management_v1; tone mapping disabled"
                );
                None
            }
            _ => None,
        };

        // Bare commit maps the layer surface; the configure callback draws
        // the color, service() attaches the image once it's prepared.
        layer.commit();

        tracing::info!(output = name, mode = ?spec.effective_mode(), "wallpaper surface created");
        let color = spec.color.unwrap_or(Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        });
        self.wallpapers.push(Wallpaper {
            output,
            name,
            spec,
            layer,
            viewport,
            color,
            image_part: None,
            fade: None,
            broken: false,
            _feedback: feedback,
            size: (0, 0),
            scale,
            color_buffer: None,
            image_buffer: None,
        });
    }

    /// Attach images to wallpapers whose prerequisites have resolved:
    /// the tone target (when `--tone-map auto`) and the image description
    /// readiness. A wallpaper needs attention when its desired image (the
    /// playlist's current entry) differs from the attached one — the old
    /// part keeps showing until the new one is ready. Called from the
    /// main loop after every dispatch — cheap when nothing is pending.
    pub fn service(&mut self, qh: &QueueHandle<App>) {
        for i in 0..self.wallpapers.len() {
            let wp = &self.wallpapers[i];
            if wp.broken {
                continue;
            }
            let spec = wp.spec.clone();
            let name = wp.name.clone();
            // Playlist specs show the playlist's current entry.
            let path = match spec.playlist {
                Some(p) => self.playlists[p].current().to_path_buf(),
                None => match spec.image.clone() {
                    Some(path) => path,
                    None => continue,
                },
            };
            if spec.effective_mode() == Mode::SolidColor {
                continue;
            }

            // Resolve the tone target.
            let tone = match spec.tone_map {
                Some(crate::cli::ToneMap::Nits(n)) => Some(n),
                Some(crate::cli::ToneMap::Auto) => {
                    if self.color.is_none() {
                        None // warned at add_output
                    } else {
                        match self.tone_targets.get(&name) {
                            Some(&t) => Some(t),
                            None => continue, // feedback still in flight
                        }
                    }
                }
                None => None,
            };

            let treatment = resolve_treatment(&spec, tone);
            let key: ImageKey = (path, treatment.map(|t| t.key()));
            if self.wallpapers[i]
                .image_part
                .as_ref()
                .is_some_and(|p| p.key == key)
            {
                continue; // already showing it
            }
            if let Err(e) = self.ensure_image(qh, &key, treatment) {
                tracing::error!(output = name, "preparing image failed: {e:#}");
                self.wallpapers[i].broken = true;
                continue;
            }
            let ready = match &self.images[&key].description {
                None => true, // no color management; attach untagged
                Some(d) => match d.status() {
                    Status::Ready => true,
                    Status::Pending => false, // ready event will re-trigger
                    Status::Failed(msg) => {
                        tracing::error!(
                            output = name,
                            "compositor rejected image description: {msg}"
                        );
                        self.wallpapers[i].broken = true;
                        continue;
                    }
                },
            };
            if ready {
                self.attach_image(qh, i, &key);
            }
        }
    }

    /// Treat + capability-adapt + describe the image for `key` if it isn't
    /// cached yet.
    fn ensure_image(
        &mut self,
        qh: &QueueHandle<App>,
        key: &ImageKey,
        treatment: Option<crate::color::LuminanceControl>,
    ) -> Result<()> {
        if self.images.contains_key(key) {
            return Ok(());
        }
        let raw = self
            .raw_images
            .get(&key.0)
            .context("raw image missing (bug)")?
            .clone();
        let treated = match treatment {
            Some(ctrl) => {
                let t = raw.luminance_controlled(ctrl);
                tracing::info!(
                    path = %key.0.display(),
                    ?ctrl,
                    luminances = ?t.encoding.luminances,
                    "luminance treatment applied"
                );
                Arc::new(t)
            }
            None => raw,
        };
        let adapted = self.adapt_image(treated)?;
        let description = match &self.color {
            Some(color) => Some(color.create_description(qh, &adapted.encoding)?),
            None => None,
        };
        self.images.insert(
            key.clone(),
            LoadedImage {
                image: adapted,
                description,
            },
        );
        Ok(())
    }

    /// Adapt one image to compositor capabilities, two ordered axes (see
    /// the module docs in `decode`): TF vocabulary first (full-precision
    /// pixels), buffer container second (fp16 → unorm16+PQ → 8-bit).
    fn adapt_image(&self, image: Arc<DecodedImage>) -> Result<Arc<DecodedImage>> {
        use crate::color::Tf;
        use wayland_client::protocol::wl_shm::Format;

        let formats = self.shm.formats();
        let fp16_ok = formats.contains(&Format::Abgr16161616f)
            && std::env::var_os("PRISM_BG_FORCE_NO_FP16").is_none();
        let unorm16_ok = formats.contains(&Format::Abgr16161616);
        let sdr_tf = match &self.color {
            Some(c) => [Tf::Srgb, Tf::Gamma22, Tf::Bt1886]
                .into_iter()
                .find(|&t| c.supports_tf(t)),
            None => Some(Tf::Srgb),
        };

        let mut image = image;

        // Axis 1: TF vocabulary.
        if let Some(color) = &self.color {
            let tf = image.encoding.tf;
            if !color.supports_tf(tf) {
                match tf {
                    Tf::Srgb | Tf::Gamma22 | Tf::Bt1886 | Tf::Linear => {
                        let target =
                            sdr_tf.context("compositor supports no display-referred TF")?;
                        if tf == Tf::Linear {
                            tracing::warn!(
                                ?target,
                                "compositor lacks ext_linear; re-encoding (HDR clips at \
                                 reference white)"
                            );
                        } else {
                            tracing::info!(from = ?tf, to = ?target, "re-encoding TF");
                        }
                        image = Arc::new(image.reencoded_tf(target));
                    }
                    Tf::Pq => anyhow::bail!("compositor does not support the PQ transfer function"),
                }
            }
        }

        // Axis 2: buffer container.
        if !fp16_ok && matches!(image.pixels, crate::decode::Pixels::RgbaF16(_)) {
            let target = sdr_tf.context("compositor supports no display-referred TF")?;
            if unorm16_ok {
                let pq_ok = self.color.as_ref().is_some_and(|c| c.supports_tf(Tf::Pq));
                tracing::info!(
                    pq = pq_ok && image.encoding.tf == Tf::Linear,
                    "compositor lacks fp16 shm; repacking as 16-bit unorm"
                );
                image = Arc::new(image.repacked_unorm16(pq_ok, target));
            } else {
                tracing::warn!(
                    ?target,
                    "compositor lacks fp16 and 16-bit shm; quantizing to 8-bit"
                );
                image = Arc::new(image.quantized_to_8bit(target));
            }
        }
        Ok(image)
    }

    /// Build the image subsurface for wallpaper `i` from the prepared
    /// image at `key`, tag it, and draw. When this replaces an existing
    /// part and `--fade` is in effect, the old part stays mapped (a new
    /// subsurface stacks topmost, so the incoming one covers it) and the
    /// incoming surface fades in over it.
    fn attach_image(&mut self, qh: &QueueHandle<App>, i: usize, key: &ImageKey) {
        // A fade still running here means rotation outpaced it; cut it
        // short so there is exactly one outgoing part.
        self.finish_fade(i);

        let wp = &self.wallpapers[i];
        let fade_duration = wp
            .spec
            .fade
            .filter(|_| self.alpha_modifier.is_some() && wp.image_part.is_some());

        let loaded = &self.images[key];
        let (subsurface, child) = self
            .subcompositor
            .create_subsurface(wp.layer.wl_surface().clone(), qh);
        let viewport = self.get_viewport(qh, &child);
        let cm = match (&self.color, &loaded.description) {
            (Some(color), Some(desc)) => Some(color.tag_surface(qh, &child, desc, self.intent)),
            _ => None,
        };
        let alpha = fade_duration.map(|_| {
            let alpha = self
                .alpha_modifier
                .as_ref()
                .expect("checked above")
                .get_surface(&child, qh, ());
            alpha.set_multiplier(0);
            alpha
        });
        // The opaque-region hint must reflect what the compositor may
        // skip drawing under us: a fading part is translucent whatever
        // its pixels say, so it gets the hint at fade completion instead.
        if !loaded.image.has_alpha && fade_duration.is_none() {
            if let Ok(region) = Region::new(&self.compositor) {
                region.add(0, 0, i32::MAX, i32::MAX);
                child.set_opaque_region(Some(region.wl_region()));
            }
        }
        let part = ImagePart {
            subsurface,
            surface: child,
            viewport,
            alpha,
            cm,
            image: loaded.image.clone(),
            mode: wp.spec.effective_mode(),
            key: key.clone(),
        };
        tracing::info!(output = wp.name, fade = ?fade_duration, "image attached");
        let wp = &mut self.wallpapers[i];
        let old_part = wp.image_part.replace(part);
        let old_buffer = wp.image_buffer.take();
        if let Some(duration) = fade_duration {
            wp.fade = Some(Fade {
                start: Instant::now(),
                duration,
                _old_part: old_part.expect("fade only replaces an existing part"),
                _old_buffer: old_buffer,
            });
            self.arm_fade_timer();
        }
        self.draw(i);
    }

    /// Insert the fade ticker if it isn't running. A calloop timer rather
    /// than frame callbacks: an occluded surface gets no frame callbacks
    /// (the parent is always hidden under its own opaque image part, and
    /// the whole wallpaper may sit under a fullscreen window), which
    /// would stall a callback-paced fade indefinitely.
    fn arm_fade_timer(&mut self) {
        if self.fade_timer_armed {
            return;
        }
        let inserted = self.loop_handle.insert_source(
            Timer::from_duration(FADE_TICK),
            |_, _, app: &mut App| {
                if app.step_fades() {
                    TimeoutAction::ToDuration(FADE_TICK)
                } else {
                    app.fade_timer_armed = false;
                    TimeoutAction::Drop
                }
            },
        );
        match inserted {
            Ok(_) => self.fade_timer_armed = true,
            Err(e) => {
                // Can't animate; cut every pending fade to its end state.
                tracing::error!("inserting fade timer: {e}; cutting hard");
                for i in 0..self.wallpapers.len() {
                    self.finish_fade(i);
                    self.wallpapers[i].layer.commit();
                }
            }
        }
    }

    /// Advance every running crossfade by one tick; returns whether any
    /// is still running (keeps the ticker alive).
    fn step_fades(&mut self) -> bool {
        for i in 0..self.wallpapers.len() {
            self.step_fade(i);
        }
        self.wallpapers.iter().any(|w| w.fade.is_some())
    }

    /// Advance wallpaper `i`'s crossfade: bump the incoming surface's
    /// alpha multiplier and re-commit (the subsurface is sync, so the
    /// parent commit latches it).
    fn step_fade(&mut self, i: usize) {
        let Some(fade) = &self.wallpapers[i].fade else {
            return;
        };
        let t = fade.start.elapsed().as_secs_f64() / fade.duration.as_secs_f64();
        tracing::trace!(output = self.wallpapers[i].name, t, "fade step");
        if t >= 1.0 {
            self.finish_fade(i);
            self.wallpapers[i].layer.commit();
            return;
        }
        let wp = &self.wallpapers[i];
        let part = wp.image_part.as_ref().expect("fading wallpaper has a part");
        let alpha = part.alpha.as_ref().expect("fading part has a multiplier");
        alpha.set_multiplier((t * u32::MAX as f64) as u32);
        part.surface.commit();
        wp.layer.commit();
    }

    /// Complete wallpaper `i`'s crossfade, if one is running: full
    /// opacity, the deferred opaque-region hint, and the outgoing part
    /// dropped (destroying its subsurface tree). The caller commits the
    /// parent to latch the child state.
    fn finish_fade(&mut self, i: usize) {
        let wp = &mut self.wallpapers[i];
        let Some(_fade) = wp.fade.take() else {
            return;
        };
        let part = wp.image_part.as_ref().expect("fading wallpaper has a part");
        let alpha = part.alpha.as_ref().expect("fading part has a multiplier");
        alpha.set_multiplier(u32::MAX);
        if !part.image.has_alpha {
            if let Ok(region) = Region::new(&self.compositor) {
                region.add(0, 0, i32::MAX, i32::MAX);
                part.surface.set_opaque_region(Some(region.wl_region()));
            }
        }
        part.surface.commit();
        // _fade drops here: outgoing subsurface destroyed, buffer freed.
    }

    fn get_viewport(&self, qh: &QueueHandle<App>, surface: &WlSurface) -> WpViewport {
        self.viewporter
            .get()
            .expect("viewporter bound at startup")
            .get_viewport(surface, qh, ())
    }

    fn draw(&mut self, index: usize) {
        if let Err(e) = self.try_draw(index) {
            tracing::error!(
                output = self.wallpapers[index].name,
                "drawing wallpaper failed: {e:#}"
            );
        }
    }

    fn try_draw(&mut self, index: usize) -> Result<()> {
        let wp = &mut self.wallpapers[index];
        let (w, h) = wp.size;
        if w == 0 || h == 0 {
            return Ok(());
        }

        // 8-bit wire format: Abgr8888 matches RGBA memory directly but is
        // optional (KWin lacks it); fall back to swizzling into the
        // spec-mandatory Argb8888. PRISM_BG_FORCE_ARGB=1 forces the
        // swizzle path for testing.
        let wire = if self.shm.formats().contains(&wl_shm::Format::Abgr8888)
            && std::env::var_os("PRISM_BG_FORCE_ARGB").is_none()
        {
            WireRgb8::Abgr
        } else {
            WireRgb8::ArgbSwizzled
        };

        // Image subsurface first; as a synchronized subsurface its state is
        // latched by the parent commit below.
        if let Some(part) = &wp.image_part {
            let placement = place(
                part.mode,
                (w, h),
                wp.scale,
                (part.image.width, part.image.height),
            );
            let buffer = match placement.tile {
                Some((tw, th)) => upload_tiled(&mut self.pool, &part.image, tw, th, wire)?,
                None => upload(&mut self.pool, &part.image, wire)?,
            };
            buffer
                .attach_to(&part.surface)
                .context("attaching image buffer")?;
            if let Some((x, y, sw, sh)) = placement.src {
                part.viewport.set_source(x, y, sw, sh);
            } else {
                part.viewport.set_source(-1.0, -1.0, -1.0, -1.0);
            }
            part.viewport
                .set_destination(placement.dest.0, placement.dest.1);
            part.subsurface
                .set_position(placement.pos.0, placement.pos.1);
            part.surface.damage_buffer(0, 0, i32::MAX, i32::MAX);
            part.surface.commit();
            wp.image_buffer = Some(buffer);
        }

        // Parent: 1×1 solid color, viewport-stretched to the whole output.
        // sRGB-encoded, premultiplied (alpha 1): plain 8-bit is exact.
        let (buffer, canvas) = self
            .pool
            .create_buffer(1, 1, 4, wl_shm::Format::Argb8888)
            .context("creating color buffer")?;
        let px = |v: f64| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        // The pool may hand back a canvas larger than requested (minimum
        // slot size); only the buffer's own bytes matter.
        canvas[..4].copy_from_slice(&[px(wp.color.b), px(wp.color.g), px(wp.color.r), 0xff]);
        buffer
            .attach_to(wp.layer.wl_surface())
            .context("attaching color buffer")?;
        wp.viewport.set_destination(w as i32, h as i32);
        wp.layer
            .wl_surface()
            .damage_buffer(0, 0, i32::MAX, i32::MAX);
        wp.layer.commit();
        wp.color_buffer = Some(buffer);
        Ok(())
    }

    /// Advance playlist `idx` and re-attach every wallpaper showing it.
    /// Called from the per-playlist rotation timer. Decoding is
    /// synchronous — a slow decode stalls the event loop briefly, which
    /// is fine at rotation cadence.
    pub fn rotate(&mut self, qh: &QueueHandle<App>, idx: usize) {
        let previous = self.playlists[idx].current().to_path_buf();

        // Find the next decodable entry, skipping (with a warning) files
        // that fail — a deleted or corrupt entry must not kill the daemon.
        let mut next = None;
        for _ in 0..self.playlists[idx].len() {
            self.playlists[idx].advance();
            let path = self.playlists[idx].current().to_path_buf();
            if self.raw_images.contains_key(&path) {
                next = Some(path);
                break;
            }
            match crate::decode::load(&path) {
                Ok(img) => {
                    tracing::info!(path = %path.display(), "image loaded");
                    self.raw_images.insert(path.clone(), Arc::new(img));
                    next = Some(path);
                    break;
                }
                Err(e) => {
                    tracing::warn!(path = %path.display(), "skipping playlist entry: {e:#}")
                }
            }
        }
        let Some(next) = next else {
            tracing::error!("playlist has no loadable entries; keeping current wallpaper");
            return;
        };
        if next == previous {
            return; // single-entry list (or every other entry failed)
        }
        tracing::info!(from = %previous.display(), to = %next.display(), "rotating wallpaper");

        // The old part stays attached (and displayed) until service()
        // swaps in — or fades in — the new one; with the playlist
        // advanced, the key comparison flags these wallpapers as stale.
        for wp in &mut self.wallpapers {
            if wp.spec.playlist == Some(idx) {
                wp.broken = false; // retry with the new entry
            }
        }
        self.evict_unused_images();
        self.service(qh);
    }

    /// Drop cached raw/treated images not reachable from any static spec
    /// or current playlist position — keeps long-running rotation
    /// memory-flat.
    fn evict_unused_images(&mut self) {
        let mut live: HashSet<PathBuf> =
            self.specs.iter().filter_map(|s| s.image.clone()).collect();
        live.extend(self.playlists.iter().map(|p| p.current().to_path_buf()));
        self.raw_images.retain(|path, _| live.contains(path));
        self.images.retain(|(path, _), _| live.contains(path));
    }

    fn remove_output(&mut self, output: &wl_output::WlOutput) {
        if let Some(i) = self.wallpapers.iter().position(|w| &w.output == output) {
            let wp = self.wallpapers.remove(i);
            tracing::info!(output = wp.name, "output removed");
        }
    }
}

impl CompositorHandler for App {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &WlSurface,
        _: i32,
    ) {
        // Output scale is tracked via update_output.
    }

    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &WlSurface,
        _: wl_output::Transform,
    ) {
    }

    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlSurface, _: u32) {
        // Static content; we never request frame callbacks (and an
        // occluded wallpaper wouldn't receive them — fades tick on a
        // calloop timer instead).
    }

    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for App {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(&mut self, _: &Connection, qh: &QueueHandle<Self>, output: wl_output::WlOutput) {
        // Safe at any time: image attachment is deferred to service(),
        // which only runs once main's setup roundtrips are done.
        self.add_output(qh, output);
    }

    fn update_output(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        let Some(info) = self.output_state.info(&output) else {
            return;
        };
        if let Some(i) = self.wallpapers.iter().position(|w| w.output == output) {
            if self.wallpapers[i].scale != info.scale_factor {
                self.wallpapers[i].scale = info.scale_factor;
                // Scale affects center (1:1 pixels) and tile (assembled
                // buffer); other modes are scale-independent.
                let affected = self.wallpapers[i]
                    .image_part
                    .as_ref()
                    .is_some_and(|p| matches!(p.mode, Mode::Center | Mode::Tile));
                if affected {
                    self.finish_fade(i); // don't leave the outgoing part stale
                    self.draw(i);
                }
            }
        }
    }

    fn output_destroyed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        self.remove_output(&output);
    }
}

impl LayerShellHandler for App {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, layer: &LayerSurface) {
        if let Some(i) = self.wallpapers.iter().position(|w| &w.layer == layer) {
            let wp = self.wallpapers.remove(i);
            tracing::info!(output = wp.name, "layer surface closed by compositor");
        }
    }

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let Some(i) = self.wallpapers.iter().position(|w| &w.layer == layer) else {
            return;
        };
        let new_size = configure.new_size;
        if new_size == self.wallpapers[i].size && self.wallpapers[i].color_buffer.is_some() {
            return;
        }
        // A resize mid-fade would leave the outgoing part at stale
        // geometry (only the current part is redrawn); cut to the end.
        self.finish_fade(i);
        self.wallpapers[i].size = new_size;
        self.draw(i);
    }
}

impl ShmHandler for App {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl ProvidesRegistryState for App {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState];
}

impl AsMut<SimpleViewporter> for App {
    fn as_mut(&mut self) -> &mut SimpleViewporter {
        &mut self.viewporter
    }
}

delegate_compositor!(App);
delegate_subcompositor!(App);
delegate_output!(App);
delegate_shm!(App);
delegate_layer!(App);
delegate_registry!(App);
delegate_simple!(App, WpViewporter, 1);
wayland_client::delegate_noop!(App: ignore WpViewport);
wayland_client::delegate_noop!(App: WpAlphaModifierV1);
wayland_client::delegate_noop!(App: WpAlphaModifierSurfaceV1);
