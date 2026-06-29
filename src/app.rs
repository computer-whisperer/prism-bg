//! The Wayland application: one background layer surface per matched output.
//! Every non-solid wallpaper renders on the GPU — images as a degenerate
//! shader graph, `--shader` wallpapers directly — into an fp16 dmabuf
//! attached to the layer surface and tagged via `wp_color_management_v1`.
//! Solid-color wallpapers use a 1×1 viewport-stretched shm buffer.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use half::f16;
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, Region, SurfaceData},
    delegate_compositor, delegate_layer, delegate_output, delegate_pointer, delegate_registry,
    delegate_seat, delegate_shm, delegate_simple,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        pointer::{
            CursorIcon, PointerData, PointerEvent, PointerEventKind, PointerHandler, ThemeSpec,
            ThemedPointer,
        },
        Capability, SeatHandler, SeatState,
    },
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
};
use wayland_client::{
    globals::GlobalList,
    protocol::{wl_output, wl_pointer::WlPointer, wl_seat::WlSeat, wl_shm, wl_surface::WlSurface},
    Connection, Proxy, QueueHandle,
};
use wayland_protocols::wp::viewporter::client::{
    wp_viewport::WpViewport, wp_viewporter::WpViewporter,
};

use wayland_protocols::wp::linux_dmabuf::zv1::client::zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1;

use smithay_client_toolkit::reexports::calloop;

use crate::audio::AudioCapture;
use crate::cli::{Args, Color, DarkHours, Intent, Luminance, Mode, OutputSpec, ProfileMode};
use std::time::{Duration, Instant};
use crate::color::{ColorEncoding, LuminanceControl};
use crate::colormgmt::ColorState;
use crate::decode::DecodedImage;
use crate::gpu::GpuPool;
use crate::playlist::Playlist;
use crate::shader::{DmabufState, ShaderSurface, Tiling};

/// Image identity for deduplication: path + effective luminance treatment
/// (the same file treated differently for different outputs is different
/// pixels).
pub type ImageKey = (PathBuf, Option<(u64, u64, u64)>);

/// A decoded + treated image in the unified working space (extended-linear,
/// sRGB primaries), ready for GPU upload. This is the product of the expensive,
/// unbounded-latency CPU work — file decode + luminance shaping + colour-space
/// conversion — which the prep worker does off the event-loop thread.
pub struct PreparedImage {
    pixels: Vec<f16>,
    encoding: ColorEncoding,
    w: u32,
    h: u32,
}

/// Convert an already-decoded image to a [`PreparedImage`]. Pure CPU, no I/O —
/// shared by the worker (after it decodes) and the synchronous startup path
/// (which already holds the decoded raw).
fn prepare_from_raw(raw: &DecodedImage, treatment: Option<LuminanceControl>) -> PreparedImage {
    let (pixels, encoding, w, h) = match treatment {
        Some(ctrl) => {
            let t = raw.luminance_controlled(ctrl);
            let (p, e) = t.to_working_space();
            (p, e, t.width, t.height)
        }
        None => {
            let (p, e) = raw.to_working_space();
            (p, e, raw.width, raw.height)
        }
    };
    PreparedImage {
        pixels,
        encoding,
        w,
        h,
    }
}

/// A request to the prep worker: decode + treat `path` into a [`PreparedImage`]
/// identified by `key`. Opaque to `main`, which only shuttles the channels.
pub struct PrepJob {
    key: ImageKey,
    path: PathBuf,
    treatment: Option<LuminanceControl>,
}

/// The worker's reply for one [`PrepJob`]: the prepared image, or the decode
/// error (a broken/missing file must not stall the daemon).
pub struct PrepResult {
    key: ImageKey,
    /// The image's average-luminance class, computed on the decode thread for
    /// `--dark-hours` filtering; `None` on decode failure.
    class: Option<Luminance>,
    result: Result<PreparedImage>,
}

/// Mean-luminance cutoff splitting dark from bright wallpapers. Heuristic — a
/// mostly-dark scene with sparse highlights lands well below it, a daylit photo
/// well above.
const DARK_LUMINANCE_CUTOFF: f32 = 0.4;

/// Bucket a mean luminance (`0..1`) into a [`Luminance`] class.
fn classify_luminance(mean: f32) -> Luminance {
    if mean < DARK_LUMINANCE_CUTOFF {
        Luminance::Dark
    } else {
        Luminance::Bright
    }
}

/// Spawn the background image-prep worker. Returns the job sender (held by
/// [`App`]) and the result channel (inserted into the event loop by `main`, so a
/// finished prep wakes the loop and drives [`App::on_image_prepared`]). The
/// worker exits when the job sender drops (App teardown).
pub fn spawn_prep_worker() -> (mpsc::Sender<PrepJob>, calloop::channel::Channel<PrepResult>) {
    let (job_tx, job_rx) = mpsc::channel::<PrepJob>();
    let (res_tx, res_rx) = calloop::channel::channel::<PrepResult>();
    std::thread::Builder::new()
        .name("prism-bg-imgprep".into())
        .spawn(move || {
            // Blocks on each job; ends when the job sender (App) drops.
            for job in job_rx {
                let decoded = crate::decode::load(&job.path);
                // Classify on this thread, where the decoded pixels live.
                let class = decoded
                    .as_ref()
                    .ok()
                    .map(|raw| classify_luminance(raw.mean_luminance()));
                let result = decoded.map(|raw| {
                    tracing::info!(path = %job.path.display(), "image loaded (background)");
                    prepare_from_raw(&raw, job.treatment)
                });
                if res_tx
                    .send(PrepResult {
                        key: job.key,
                        class,
                        result,
                    })
                    .is_err()
                {
                    break; // event loop gone
                }
            }
        })
        .expect("spawning image-prep thread");
    (job_tx, res_rx)
}

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

/// The luminance an output advertises (cd/m²), resolved from its preferred
/// image description. `reference` is the diffuse-white level shader value `1.0`
/// maps to under the anchored intent; `max` is the mastering-display peak to
/// master highlights against (the compositor's advertised peak, deliberately
/// distinct from the panel's marketing/HDR-metadata `max_cll`).
#[derive(Clone, Copy, Debug)]
pub struct OutputLum {
    pub reference: f64,
    pub max: f64,
}

/// SDR-safe luminance used before an output's preferred description resolves
/// (and when there's no color management): `1.0` = 203 nits, no highlight
/// headroom, so a shader renders plain SDR and can't overblow until real caps
/// arrive.
pub(crate) const DEFAULT_OUTPUT_LUM: OutputLum = OutputLum {
    reference: 203.0,
    max: 203.0,
};

/// In-flight luminances for one output while its preferred description streams
/// in: `(target_max_cll, target_luminance.max, luminances.reference)`, each
/// `Some` once the matching info event arrives, read positionally on `Done`.
pub type PendingLum = (Option<f64>, Option<f64>, Option<f64>);

struct Wallpaper {
    output: wl_output::WlOutput,
    name: String,
    spec: OutputSpec,
    layer: LayerSurface,
    viewport: WpViewport,
    color: Color,
    /// Image preparation failed; don't retry every service pass.
    broken: bool,
    /// The image source currently loaded into the GPU surface (path +
    /// luminance treatment); drives staleness detection in [`App::service`].
    /// `None` for shader/solid wallpapers and before the first image loads.
    loaded: Option<ImageKey>,
    /// Keeps the preferred-description subscription alive (auto mode).
    _feedback: Option<
        wayland_protocols::wp::color_management::v1::client
            ::wp_color_management_surface_feedback_v1::WpColorManagementSurfaceFeedbackV1,
    >,
    /// Logical size from the last configure; 0 until configured.
    size: (u32, u32),
    scale: i32,
    /// 1×1 solid background buffer (solid-color wallpapers, and the
    /// pre-render background of image wallpapers).
    color_buffer: Option<Buffer>,
    /// The GPU surface that renders this wallpaper — an image (as a degenerate
    /// graph) or a `--shader`. `None` for solid-color wallpapers and until an
    /// image wallpaper's first source is built. Renders dmabufs attached to
    /// the layer surface.
    shader: Option<ShaderSurface>,
    /// Per-output GPU render-time profiling window (unused unless `--profile-gpu`).
    profile_state: ProfileState,
}

/// Drives one output's GPU-profiling report window. The samples themselves live
/// in the renderer ([`ShaderSurface::drain_profile`]); this tracks *when* to
/// report and resets on each shader load.
#[derive(Default)]
struct ProfileState {
    /// Last [`ShaderSurface::load_generation`] seen; a change resets the window.
    generation: u64,
    /// Start of the current measurement window; `None` between an `OnLoad`
    /// report and the next load.
    window_start: Option<Instant>,
    /// `OnLoad`: already reported for this load (so we stay quiet).
    reported: bool,
}

pub struct App {
    pub registry_state: RegistryState,
    pub output_state: OutputState,
    pub shm: Shm,
    pub pool: SlotPool,
    pub compositor: CompositorState,
    pub layer_shell: LayerShell,
    pub viewporter: SimpleViewporter,
    /// Seats are always tracked, but a pointer is only created (and surfaces
    /// made input-receiving) when a shader uses `iMouse` — see `wants_mouse`.
    pub seat_state: SeatState,
    /// One themed pointer per seat that has the capability; empty unless
    /// `wants_mouse`. Kept alive here; events arrive via [`PointerHandler`].
    pointers: Vec<ThemedPointer>,
    /// A shader somewhere references `iMouse`, so we bind pointers and make the
    /// interactive surfaces input-receiving (non-interactive ones get an empty
    /// input region so they stay click-through, as a wallpaper should).
    wants_mouse: bool,
    pub color: Option<ColorState>,
    pub intent: Intent,
    /// GPU render-time profiling mode (`--profile-gpu[-every]`); `Off` by default.
    profile: ProfileMode,
    /// `--dark-hours` window; when set, playlists prefer dark/bright by clock.
    dark_hours: Option<DarkHours>,
    pub specs: Vec<OutputSpec>,
    /// Raw decoded images by path, kept for deriving treated variants
    /// (per-output tone targets, hotplug). Populated synchronously at startup
    /// (fail-fast); rotation images decode off-thread and never land here.
    pub raw_images: HashMap<PathBuf, Arc<DecodedImage>>,
    /// Working-space images ready for upload, keyed by [`ImageKey`]. Filled by
    /// the prep worker (rotation) or synchronously from `raw_images` (startup);
    /// one entry feeds every output that shares the key. Pruned by liveness in
    /// [`Self::evict_unused_images`].
    prepared: HashMap<ImageKey, Arc<PreparedImage>>,
    /// Keys with a prep job in flight, so concurrent outputs (and re-`service`
    /// passes) don't enqueue the same decode twice.
    prep_in_flight: HashSet<ImageKey>,
    /// Job channel to the background prep worker. Dropping it stops the worker.
    prep_jobs: mpsc::Sender<PrepJob>,
    /// Resolved per-output advertised luminance, from the preferred image
    /// description: the reference white (shader `1.0` maps here) and the peak
    /// to master against. Drives both `--tone-map auto` (the `max`) and the
    /// shader `iRefWhite`/`iMaxLum` uniforms.
    pub output_lums: HashMap<String, OutputLum>,
    /// In-flight info collection per output, each field filled as its preferred-
    /// description event arrives and consumed on `Done`. See [`PendingLum`].
    pub pending_targets: HashMap<String, PendingLum>,
    /// Rotation state per `--image-list` spec group, indexed by
    /// `OutputSpec::playlist`. Advanced by per-playlist timers in `main`.
    pub playlists: Vec<Playlist>,
    wallpapers: Vec<Wallpaper>,
    // Declared after `wallpapers` so they drop *after* it: a ShaderSurface
    // tears down its Vulkan objects in Drop using devices the pool owns.
    /// Vulkan backend (one logical device per physical GPU), created lazily
    /// only when a `--shader` spec exists.
    pub gpus: Option<GpuPool>,
    /// `zwp_linux_dmabuf_v1` (bound v4 for per-surface feedback).
    pub dmabuf: Option<DmabufState>,
    /// PipeWire spectrum capture, started lazily the first time a shader that
    /// references the audio uniforms is built. `None` until then (and forever
    /// if no shader uses audio).
    audio: Option<AudioCapture>,
}

pub type SimpleViewporter = smithay_client_toolkit::registry::SimpleGlobal<WpViewporter, 1>;

impl App {
    pub fn new(
        globals: &GlobalList,
        qh: &QueueHandle<App>,
        args: &Args,
        raw_images: HashMap<PathBuf, Arc<DecodedImage>>,
        playlists: Vec<Playlist>,
        prep_jobs: mpsc::Sender<PrepJob>,
    ) -> Result<App> {
        let compositor =
            CompositorState::bind(globals, qh).context("wl_compositor not available")?;
        let layer_shell = LayerShell::bind(globals, qh).context(
            "zwlr_layer_shell_v1 not available (compositor without layer-shell support?)",
        )?;
        let shm = Shm::bind(globals, qh).context("wl_shm not available")?;
        let pool = SlotPool::new(4096, &shm).context("creating shm pool")?;
        let viewporter =
            SimpleViewporter::bind(globals, qh).context("wp_viewporter not available")?;
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

        // Every non-solid wallpaper now renders on the GPU (images as a
        // degenerate shader graph, shaders directly), so the GPU backend and
        // dmabuf import are needed whenever any spec draws something.
        let wants_gpu = args.specs.iter().any(|s| {
            s.shader.is_some()
                || ((s.image.is_some() || s.image_list.is_some())
                    && s.effective_mode() != Mode::SolidColor)
        });
        // Does any shader want pointer input? Peek the source files now (a cheap
        // startup-only read) so the seat's pointer is created — and surfaces get
        // the right input region — before the first output is added. A read
        // failure here just defers to the real error when the shader is built.
        // Same usage scan as the per-surface detection, so a shader that merely
        // *declares* iMouse to reach a later field (iDate/iFrame) doesn't count.
        let wants_mouse = args
            .specs
            .iter()
            .filter_map(|s| s.shader.as_ref())
            .any(|p| {
                std::fs::read_to_string(p)
                    .is_ok_and(|src| crate::shader::usage_scan_source(&src).contains("iMouse"))
            });
        if wants_mouse {
            tracing::info!("a shader uses iMouse; binding seat pointer for interactivity");
        }
        let gpus = if wants_gpu {
            Some(GpuPool::new().context("initializing GPU backend for wallpaper rendering")?)
        } else {
            None
        };
        // Bind at version 4 for per-surface feedback (the output's GPU +
        // importable modifiers). Required for GPU-rendered wallpapers.
        let dmabuf = globals
            .bind::<ZwpLinuxDmabufV1, _, _>(qh, 4..=4, ())
            .ok()
            .map(|proxy| DmabufState { proxy });
        if wants_gpu && dmabuf.is_none() {
            bail!("compositor lacks zwp_linux_dmabuf_v1 v4; needed for GPU wallpaper presentation");
        }

        // Classify the images decoded synchronously at startup (each playlist's
        // initial entry), so the first rotation can already filter on them.
        // Rotation images are classified as they decode (see on_image_prepared).
        let mut playlists = playlists;
        for (path, raw) in &raw_images {
            let class = classify_luminance(raw.mean_luminance());
            for pl in &mut playlists {
                pl.set_class(path, class);
            }
        }

        Ok(App {
            registry_state: RegistryState::new(globals),
            output_state: OutputState::new(globals, qh),
            shm,
            pool,
            compositor,
            layer_shell,
            viewporter,
            seat_state: SeatState::new(globals, qh),
            pointers: Vec::new(),
            wants_mouse,
            color,
            intent: args.intent,
            profile: args.profile,
            dark_hours: args.dark_hours,
            specs: args.specs.clone(),
            raw_images,
            prepared: HashMap::new(),
            prep_in_flight: HashSet::new(),
            prep_jobs,
            output_lums: HashMap::new(),
            pending_targets: HashMap::new(),
            playlists,
            wallpapers: Vec::new(),
            gpus,
            dmabuf,
            audio: None,
        })
    }

    /// Start the PipeWire spectrum capture if it isn't already running. Called
    /// the first time a shader referencing the audio uniforms is built, so the
    /// capture (and its visible audio stream) only exists when something wants
    /// it. Idempotent.
    fn ensure_audio_capture(&mut self) {
        if self.audio.is_none() {
            tracing::info!("audio-reactive shader detected; starting PipeWire spectrum capture");
            self.audio = Some(crate::audio::AudioCapture::start());
        }
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

        // Subscribe to the output's preferred description (which carries its
        // advertised luminance) before the first service pass. Needed by
        // `--tone-map auto` images and by every shader (which masters its
        // content to the output's reference white + peak via iRefWhite/iMaxLum).
        let wants_image = (spec.image.is_some() || spec.playlist.is_some())
            && spec.effective_mode() != Mode::SolidColor;
        let wants_auto = matches!(spec.tone_map, Some(crate::cli::ToneMap::Auto)) && wants_image;
        let is_shader = spec.shader.is_some();
        let feedback = match &self.color {
            Some(color) if wants_auto || is_shader => {
                Some(color.watch_preferred(qh, layer.wl_surface(), name.clone()))
            }
            None if wants_auto => {
                tracing::warn!(
                    output = name,
                    "--tone-map auto needs wp_color_management_v1; tone mapping disabled"
                );
                None
            }
            _ => None,
        };

        // A shader spec renders on the GPU instead of attaching an image.
        // Compile it and subscribe to per-surface dmabuf feedback (which
        // resolves the output's GPU); targets are built once feedback and
        // the configure size are both in.
        let shader = match &spec.shader {
            Some(path) => match self.build_shader(qh, path, spec.fps) {
                Ok(mut s) => {
                    if let Some(dmabuf) = &self.dmabuf {
                        s.request_feedback(dmabuf, qh, layer.wl_surface(), name.clone());
                    }
                    // First audio-reactive shader spins up the capture.
                    if s.uses_audio() {
                        self.ensure_audio_capture();
                    }
                    Some(s)
                }
                Err(e) => {
                    tracing::error!(output = name, "shader setup failed: {e:#}");
                    None
                }
            },
            None => None,
        };
        let shader_broken = spec.shader.is_some() && shader.is_none();

        // Pointer focus: a wallpaper should be click-through unless it's an
        // interactive (iMouse) shader. Surfaces default to an infinite input
        // region, so once we've bound a pointer (`wants_mouse`) we must give
        // every non-interactive surface an empty input region or it would
        // swallow desktop clicks. Interactive surfaces keep the default region.
        if self.wants_mouse {
            let interactive = shader.as_ref().is_some_and(|s| s.uses_mouse());
            if !interactive {
                if let Ok(empty) = Region::new(&self.compositor) {
                    layer.wl_surface().set_input_region(Some(empty.wl_region()));
                }
            }
        }

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
            broken: shader_broken,
            loaded: None,
            _feedback: feedback,
            size: (0, 0),
            scale,
            color_buffer: None,
            shader,
            profile_state: ProfileState::default(),
        });
    }

    /// Read and compile a shader file into a [`ShaderSurface`]. The GPU and
    /// targets are chosen later, once dmabuf feedback resolves the output's
    /// device and the configure size is known.
    fn build_shader(
        &self,
        qh: &QueueHandle<App>,
        path: &std::path::Path,
        fps: Option<u32>,
    ) -> Result<ShaderSurface> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("reading shader {}", path.display()))?;
        // Texture channel paths resolve relative to the .frag file's directory.
        let base_dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        ShaderSurface::new(qh, &source, base_dir, self.color.as_ref(), fps)
    }

    /// Feed a dmabuf-feedback event to the named output's GPU surface (image
    /// or shader) and (re)draw if it resolved a new GPU/modifier set. Called
    /// from the feedback dispatch handler.
    pub fn on_surface_feedback(
        &mut self,
        qh: &QueueHandle<App>,
        output: &str,
        event: wayland_protocols::wp::linux_dmabuf::zv1::client
            ::zwp_linux_dmabuf_feedback_v1::Event,
    ) {
        let Some(i) = self.wallpapers.iter().position(|w| w.name == output) else {
            return;
        };
        let resolved = match self.wallpapers[i].shader.as_mut() {
            Some(s) => s.feedback_event(event),
            None => return,
        };
        if resolved {
            self.draw(qh, i);
        }
    }

    /// Update wallpapers whose desired image differs from what's loaded into
    /// their GPU surface: resolve the tone target (deferring `--tone-map auto`
    /// until the output's luminance lands), then (re)build the image surface.
    /// The old frame keeps showing until the new one renders. Called from the
    /// main loop after every dispatch — cheap when nothing changed.
    pub fn service(&mut self, qh: &QueueHandle<App>) {
        for i in 0..self.wallpapers.len() {
            if self.wallpapers[i].broken || self.wallpapers[i].spec.shader.is_some() {
                continue; // --shader wallpapers render via feedback/frame, not here
            }
            let name = self.wallpapers[i].name.clone();
            // A playlist whose current entry is a shader: prepare + display it as
            // a GPU shader source (dissolving like any other rotation). Keyed by
            // path alone (no tone treatment), so it dedups against `loaded`.
            if let Some(p) = self.wallpapers[i].spec.playlist {
                let src = self.playlists[p].current();
                if src.is_shader() {
                    let path = src.path().to_path_buf();
                    let key: ImageKey = (path.clone(), None);
                    if self.wallpapers[i].loaded.as_ref() == Some(&key) {
                        continue; // already showing it
                    }
                    if let Err(e) = self.show_shader(qh, i, &path) {
                        tracing::error!(output = name, "preparing shader failed: {e:#}");
                        self.wallpapers[i].broken = true;
                    }
                    continue;
                }
            }
            // Desired image path: the playlist's current entry, or static -i.
            let path = match self.wallpapers[i].spec.playlist {
                Some(p) => self.playlists[p].current().path().to_path_buf(),
                None => match self.wallpapers[i].spec.image.clone() {
                    Some(path) => path,
                    None => continue, // solid color: drawn at configure
                },
            };
            if self.wallpapers[i].spec.effective_mode() == Mode::SolidColor {
                continue;
            }

            // Resolve the tone target (auto waits for the output's luminance).
            let tone = match self.wallpapers[i].spec.tone_map {
                Some(crate::cli::ToneMap::Nits(n)) => Some(n),
                Some(crate::cli::ToneMap::Auto) => {
                    if self.color.is_none() {
                        None // warned at add_output
                    } else {
                        match self.output_lums.get(&name) {
                            Some(lum) => Some(lum.max),
                            None => continue, // feedback still in flight
                        }
                    }
                }
                None => None,
            };

            let treatment = resolve_treatment(&self.wallpapers[i].spec, tone);
            let key: ImageKey = (path.clone(), treatment.map(|t| t.key()));
            if self.wallpapers[i].loaded.as_ref() == Some(&key) {
                continue; // already showing it
            }
            // Resolve the prepared image. Already cached → show it now. Else, if
            // the raw decode is in hand (startup / static `-i`, decoded fail-fast
            // in main) prepare it synchronously; otherwise (rotation) hand it to
            // the background worker and keep the current wallpaper up until the
            // result arrives — never decode an arbitrary 4K file on this thread.
            let prep = match self.prepared.get(&key) {
                Some(p) => p.clone(),
                None => match self.raw_images.get(&path) {
                    Some(raw) => {
                        let p = Arc::new(prepare_from_raw(raw, treatment));
                        self.prepared.insert(key.clone(), p.clone());
                        p
                    }
                    None => {
                        self.enqueue_prep(&key, treatment);
                        continue;
                    }
                },
            };
            if let Err(e) = self.show_image(qh, i, &key, &prep) {
                tracing::error!(output = name, "preparing image failed: {e:#}");
                self.wallpapers[i].broken = true;
            }
        }
    }

    /// Queue a background decode+prep for `key` unless it's already prepared or
    /// in flight. The current wallpaper keeps displaying until the result lands
    /// in [`Self::on_image_prepared`].
    fn enqueue_prep(&mut self, key: &ImageKey, treatment: Option<LuminanceControl>) {
        if self.prepared.contains_key(key) || !self.prep_in_flight.insert(key.clone()) {
            return;
        }
        let job = PrepJob {
            key: key.clone(),
            path: key.0.clone(),
            treatment,
        };
        if self.prep_jobs.send(job).is_err() {
            tracing::error!(path = %key.0.display(), "image-prep worker gone; cannot prepare");
            self.prep_in_flight.remove(key);
        }
    }

    /// A background prep finished: cache it and re-`service` so every output
    /// waiting on this key swaps to it (dissolving if `--fade`). A decode
    /// failure marks the outputs targeting that file broken — the current
    /// wallpaper stays up, and the next rotation tick advances past it.
    pub fn on_image_prepared(&mut self, res: PrepResult, qh: &QueueHandle<App>) {
        self.prep_in_flight.remove(&res.key);
        match res.result {
            Ok(prep) => {
                // Record the learned luminance so the next rotation can filter
                // on it (lazy: an image is classified the first time it decodes).
                if let Some(c) = res.class {
                    for pl in &mut self.playlists {
                        pl.set_class(&res.key.0, c);
                    }
                }
                self.prepared.insert(res.key, Arc::new(prep));
                self.service(qh);
            }
            Err(e) => {
                tracing::warn!(path = %res.key.0.display(), "image prep failed: {e:#}");
                self.mark_path_broken(&res.key.0);
            }
        }
    }

    /// Mark every wallpaper whose desired image is `path` as broken, so
    /// `service` stops re-enqueuing a file that won't decode (reset on the next
    /// rotation, which advances to the following entry).
    fn mark_path_broken(&mut self, path: &Path) {
        let stale: Vec<usize> = self
            .wallpapers
            .iter()
            .enumerate()
            .filter(|(_, wp)| {
                let cur = match wp.spec.playlist {
                    Some(p) => Some(self.playlists[p].current().path()),
                    None => wp.spec.image.as_deref(),
                };
                cur == Some(path)
            })
            .map(|(i, _)| i)
            .collect();
        for i in stale {
            self.wallpapers[i].broken = true;
        }
    }

    /// Build (or swap in place) the GPU surface that renders wallpaper `i` from
    /// an already-[`PreparedImage`] (decode + treatment + working-space
    /// conversion done up front, off-thread for rotations): wrap it in an
    /// [`crate::gpu::image_graph`] and create the surface (first load) or
    /// [`ShaderSurface::set_source`] it (rotation).
    fn show_image(
        &mut self,
        qh: &QueueHandle<App>,
        i: usize,
        key: &ImageKey,
        prep: &PreparedImage,
    ) -> Result<()> {
        use crate::color::Tf;

        let (w, h) = (prep.w, prep.h);
        let texture = crate::gpu::TextureData {
            width: w,
            height: h,
            // The upload copies the pixels; clone keeps the cache entry shared.
            pixels: crate::gpu::TexturePixels::LinearF16(prep.pixels.clone()),
        };
        let encoding = prep.encoding;
        let mode = self.wallpapers[i].spec.effective_mode();
        let c = self.wallpapers[i].color;
        // The letterbox/background color, converted to working-space linear.
        let bg = [
            Tf::Srgb.eotf(c.r as f32),
            Tf::Srgb.eotf(c.g as f32),
            Tf::Srgb.eotf(c.b as f32),
        ];
        let graph = crate::gpu::image_graph(w, h, mode, bg);
        let src = crate::shader::PreparedSource::image(graph, texture, encoding);

        let color = self.color.as_ref();
        let dmabuf = self.dmabuf.as_ref();
        // Blur-dissolve on rotation when --fade is set; a hard cut otherwise.
        let fade = self.wallpapers[i].spec.fade;
        let wp = &mut self.wallpapers[i];
        match wp.shader.as_mut() {
            Some(s) => s.set_source(qh, src, color, fade),
            None => {
                let mut s = ShaderSurface::from_source(qh, src, color);
                if let Some(dmabuf) = dmabuf {
                    s.request_feedback(dmabuf, qh, wp.layer.wl_surface(), wp.name.clone());
                }
                wp.shader = Some(s);
            }
        }
        wp.loaded = Some(key.clone());
        self.draw(qh, i);
        Ok(())
    }

    /// Build (or swap in place) the GPU surface that renders wallpaper `i`'s
    /// current playlist *shader* entry: parse + prepare the shader, spin up
    /// audio capture if it's reactive, then create the surface (first load) or
    /// [`ShaderSurface::set_source`] it (rotation, blur-dissolving with `--fade`).
    fn show_shader(
        &mut self,
        qh: &QueueHandle<App>,
        i: usize,
        path: &std::path::Path,
    ) -> Result<()> {
        let glsl = std::fs::read_to_string(path)
            .with_context(|| format!("reading shader {}", path.display()))?;
        let base_dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let fps = self.wallpapers[i].spec.fps;
        let src = crate::shader::PreparedSource::shader(&glsl, base_dir, fps)?;
        // A reactive playlist shader spins up the capture on arrival.
        if src.uses_audio {
            self.ensure_audio_capture();
        }
        // Pointer input is decided once at startup from --shader specs only, so
        // a playlist shader that reads iMouse gets zero unless a static --shader
        // also wanted the pointer.
        if src.uses_mouse && !self.wants_mouse {
            tracing::warn!(
                path = %path.display(),
                "playlist shader uses iMouse, but pointer input is wired only for --shader \
                 wallpapers; it will read zero"
            );
        }
        let fade = self.wallpapers[i].spec.fade;
        let color = self.color.as_ref();
        let dmabuf = self.dmabuf.as_ref();
        let wp = &mut self.wallpapers[i];
        match wp.shader.as_mut() {
            Some(s) => s.set_source(qh, src, color, fade),
            None => {
                let mut s = ShaderSurface::from_source(qh, src, color);
                if let Some(dmabuf) = dmabuf {
                    s.request_feedback(dmabuf, qh, wp.layer.wl_surface(), wp.name.clone());
                }
                wp.shader = Some(s);
            }
        }
        wp.loaded = Some((path.to_path_buf(), None));
        self.draw(qh, i);
        Ok(())
    }

    fn get_viewport(&self, qh: &QueueHandle<App>, surface: &WlSurface) -> WpViewport {
        self.viewporter
            .get()
            .expect("viewporter bound at startup")
            .get_viewport(surface, qh, ())
    }

    fn draw(&mut self, qh: &QueueHandle<App>, index: usize) {
        let result = if self.wallpapers[index].shader.is_some() {
            self.try_draw_shader(qh, index)
        } else {
            // No GPU surface yet (a solid-color wallpaper, or an image whose
            // surface service() hasn't built): show the solid background.
            self.draw_color(index)
        };
        match result {
            Err(e) => {
                tracing::error!(
                    output = self.wallpapers[index].name,
                    "drawing wallpaper failed: {e:#}"
                );
                // A shader failure (e.g. modifier negotiation) won't fix itself;
                // stop retrying every frame.
                if self.wallpapers[index].shader.is_some() {
                    self.wallpapers[index].broken = true;
                }
            }
            Ok(()) if self.profile.enabled() && self.wallpapers[index].shader.is_some() => {
                self.service_profile(index);
            }
            Ok(()) => {}
        }
    }

    /// Window and emit this output's GPU render-time report, per `--profile-gpu`.
    /// Called after each successful shader draw; cheap (mostly an early return).
    fn service_profile(&mut self, index: usize) {
        let mode = self.profile;
        let now = Instant::now();
        let wp = &mut self.wallpapers[index];
        let Some(generation) = wp.shader.as_ref().map(|s| s.load_generation()) else {
            return;
        };

        // A new shader load (initial, GPU change, or rotation) restarts the
        // window and discards any partial samples from before it.
        if generation != wp.profile_state.generation {
            wp.profile_state.generation = generation;
            wp.profile_state.window_start = Some(now);
            wp.profile_state.reported = false;
            if let Some(s) = &wp.shader {
                let _ = s.drain_profile();
            }
            return;
        }

        let window = match mode {
            ProfileMode::Off => return,
            ProfileMode::OnLoad => Duration::from_secs(5),
            ProfileMode::Every(d) => d,
        };
        let start = match wp.profile_state.window_start {
            Some(s) => s,
            // No active window: in OnLoad we've already reported for this load
            // and stay quiet; otherwise open a window now.
            None => {
                if !wp.profile_state.reported {
                    wp.profile_state.window_start = Some(now);
                }
                return;
            }
        };
        let elapsed = now.duration_since(start);
        if elapsed < window {
            return;
        }

        if let Some(acc) = wp.shader.as_ref().and_then(|s| s.drain_profile()) {
            if acc.count > 0 {
                let secs = elapsed.as_secs_f64();
                let avg_ms = (acc.sum_ns as f64 / acc.count as f64) / 1e6;
                let max_ms = acc.max_ns as f64 / 1e6;
                // Fraction of wall-clock the GPU spent on this output's render.
                let busy = (acc.sum_ns as f64 / (secs * 1e9)) * 100.0;
                let fps = acc.count as f64 / secs;
                tracing::info!(
                    output = wp.name,
                    "GPU {avg_ms:.2} ms/frame avg, {max_ms:.2} ms max — {busy:.1}% of wall \
                     ({count} frames / {secs:.1}s, {fps:.0} fps)",
                    count = acc.count,
                );
            }
        }

        match mode {
            // One report per load: close the window until the next load.
            ProfileMode::OnLoad => {
                wp.profile_state.reported = true;
                wp.profile_state.window_start = None;
            }
            // Roll straight into the next interval.
            ProfileMode::Every(_) => wp.profile_state.window_start = Some(now),
            ProfileMode::Off => {}
        }
    }

    /// Render one shader frame for wallpaper `index` on the GPU that drives
    /// its output (per dmabuf feedback). No-op until both the configure size
    /// and feedback have arrived. Disjoint field borrows (gpus / dmabuf /
    /// color / wallpaper) keep the borrow checker happy.
    fn try_draw_shader(&mut self, qh: &QueueHandle<App>, index: usize) -> Result<()> {
        let (w, h) = self.wallpapers[index].size;
        if w == 0 || h == 0 {
            return Ok(()); // not configured yet
        }
        // Which GPU drives this output? Resolved from feedback; until then
        // there's nothing to render.
        let device_dev = match self.wallpapers[index]
            .shader
            .as_ref()
            .and_then(|s| s.resolved_device())
        {
            Some(d) => d,
            None => return Ok(()), // feedback pending
        };

        // Where this output sits in the global cluster, so the shader can
        // tile continuously across the workspace. Computed before the `gpus`
        // mut borrow below, since it reads `output_state` immutably.
        let output = self.wallpapers[index].output.clone();
        let tiling = self.cluster_placement(&output, (w, h));
        // The output's advertised luminance to master against (SDR-safe default
        // until its preferred description resolves). Read before the mut borrow.
        let lum = self
            .output_lums
            .get(&self.wallpapers[index].name)
            .copied()
            .unwrap_or(DEFAULT_OUTPUT_LUM);
        // Latest spectrum (zeroed silence if no capture is running).
        let audio = self
            .audio
            .as_ref()
            .map(|a| a.snapshot())
            .unwrap_or_else(<crate::gpu::AudioUniforms as bytemuck::Zeroable>::zeroed);

        let pool = self.gpus.as_mut().context("GPU backend not initialized")?;
        let gpu = pool.get_for_device(device_dev)?;
        let dmabuf = self.dmabuf.as_ref().context("dmabuf not available")?;
        let color = self.color.as_ref();
        let intent = self.intent;
        let profile = self.profile.enabled();
        let wp = &mut self.wallpapers[index];
        let scale = wp.scale.max(1) as u32;
        let device_size = (w * scale, h * scale);
        let shader = wp.shader.as_mut().context("not a shader wallpaper")?;
        shader.render_frame(
            gpu,
            dmabuf,
            qh,
            wp.layer.wl_surface(),
            &wp.viewport,
            device_size,
            (w, h),
            tiling,
            &audio,
            (lum.reference as f32, lum.max as f32),
            color,
            intent,
            profile,
        )?;
        Ok(())
    }

    /// This output's placement in the global cluster bounding box, for shader
    /// tiling — the union of every output's logical rect (from `xdg-output`,
    /// surfaced by SCTK as `logical_position`/`logical_size`). Returned in
    /// y-up logical pixels with the origin at the cluster's bottom-left, to
    /// match the shader's y-up `fragCoord`; the layout's y-down rects are
    /// flipped here. Falls back to this output alone (offset 0, size ==
    /// global) when its logical geometry is unavailable — degrading to the
    /// per-output behavior with no continuity, never breaking.
    fn cluster_placement(&self, output: &wl_output::WlOutput, logical: (u32, u32)) -> Tiling {
        let rect = |o: &wl_output::WlOutput| {
            self.output_state
                .info(o)
                .and_then(|i| Some((i.logical_position?, i.logical_size?)))
        };
        let all: Vec<_> = self
            .output_state
            .outputs()
            .filter_map(|o| rect(&o))
            .collect();
        tiling_from_rects(rect(output), &all, logical)
    }

    /// Redraw every *static* shader. Animated shaders self-heal on their next
    /// frame (uniforms are recomputed per render), but a static shader rendered
    /// its single frame already, so a change to any per-frame uniform input —
    /// the cluster layout (output added/removed/moved/resized) or an output's
    /// advertised luminance — needs an explicit redraw to take effect.
    pub(crate) fn redraw_static_shaders(&mut self, qh: &QueueHandle<App>) {
        for i in 0..self.wallpapers.len() {
            let needs = self.wallpapers[i]
                .shader
                .as_ref()
                .is_some_and(|s| !s.animated());
            if needs {
                self.draw(qh, i);
            }
        }
    }

    /// Attach the solid background: a 1×1 color buffer, viewport-stretched to
    /// the whole output. Used for solid-color wallpapers and as the immediate
    /// background of an image wallpaper before its GPU surface first renders.
    fn draw_color(&mut self, index: usize) -> Result<()> {
        let (w, h) = self.wallpapers[index].size;
        if w == 0 || h == 0 {
            return Ok(());
        }
        let color = self.wallpapers[index].color;
        // sRGB-encoded, premultiplied (alpha 1): plain 8-bit is exact.
        let (buffer, canvas) = self
            .pool
            .create_buffer(1, 1, 4, wl_shm::Format::Argb8888)
            .context("creating color buffer")?;
        let px = |v: f64| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        // The pool may hand back a canvas larger than requested (minimum
        // slot size); only the buffer's own bytes matter.
        canvas[..4].copy_from_slice(&[px(color.b), px(color.g), px(color.r), 0xff]);
        let wp = &mut self.wallpapers[index];
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
    /// Called from the per-playlist rotation timer. Images are *not* decoded
    /// here — that happens on the background prep worker (an arbitrary 4K file
    /// from anywhere has unbounded decode latency and must never block the event
    /// loop); the current wallpaper stays up until the prep lands. Shaders are
    /// still validated synchronously (a cheap parse) so a broken shader entry is
    /// skipped immediately.
    /// The luminance class to prefer right now, or `None` when `--dark-hours`
    /// is unset (no time-of-day filtering).
    fn desired_luminance(&self) -> Option<Luminance> {
        self.dark_hours
            .map(|dh| dh.preference(crate::shader::local_minute_of_day()))
    }

    pub fn rotate(&mut self, qh: &QueueHandle<App>, idx: usize) {
        let previous = self.playlists[idx].current().path().to_path_buf();
        let desired = self.desired_luminance();

        // Advance to the next entry. Shaders are parse-validated so a broken one
        // is skipped now; images are accepted optimistically — the worker
        // validates by decoding, and a failed result skips on the next tick.
        //
        // Two passes: the first honors the `--dark-hours` luminance preference;
        // if nothing eligible is also loadable, the second ignores it so a pool
        // with no entry of the preferred class still rotates rather than freezes.
        let mut next = None;
        for honor_filter in [true, false] {
            for _ in 0..self.playlists[idx].len() {
                self.playlists[idx].advance();
                if honor_filter && !self.playlists[idx].current_eligible(desired) {
                    continue;
                }
                let src = self.playlists[idx].current();
                let path = src.path().to_path_buf();
                if !src.is_shader() {
                    next = Some(path);
                    break;
                }
                match crate::shader::validate_shader_file(&path) {
                    Ok(()) => {
                        next = Some(path);
                        break;
                    }
                    Err(e) => {
                        tracing::warn!(path = %path.display(), "skipping playlist entry: {e:#}")
                    }
                }
            }
            // First pass succeeded, or there was no filter to relax → done.
            if next.is_some() || desired.is_none() {
                break;
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

    /// Drop cached raw and prepared images not reachable from any static spec
    /// or current playlist position — keeps long-running rotation memory-flat.
    /// A just-rotated entry is the playlist's *current* path, so its prep
    /// survives; the outgoing one is freed (its pixels already live in the GPU
    /// surface, so a dissolve still finishes).
    fn evict_unused_images(&mut self) {
        let mut live: HashSet<PathBuf> =
            self.specs.iter().filter_map(|s| s.image.clone()).collect();
        live.extend(
            self.playlists
                .iter()
                .map(|p| p.current().path().to_path_buf()),
        );
        self.raw_images.retain(|path, _| live.contains(path));
        self.prepared.retain(|key, _| live.contains(&key.0));
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

    fn frame(&mut self, _: &Connection, qh: &QueueHandle<Self>, surface: &WlSurface, _: u32) {
        // Image wallpapers are static and never request frame callbacks.
        // Animated shader wallpapers re-render here and arm the next
        // callback; an occluded surface gets none, so animation pauses
        // (and with it the GPU cost) until it's visible again.
        if let Some(i) = self
            .wallpapers
            .iter()
            .position(|w| w.layer.wl_surface() == surface)
        {
            let animate = !self.wallpapers[i].broken
                && self.wallpapers[i]
                    .shader
                    .as_ref()
                    .is_some_and(|s| s.needs_redraw());
            if animate {
                self.draw(qh, i);
            }
        }
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
        // A new output grows the cluster box; existing static shaders retile.
        self.redraw_static_shaders(qh);
    }

    fn update_output(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        let Some(info) = self.output_state.info(&output) else {
            return;
        };
        if let Some(i) = self.wallpapers.iter().position(|w| w.output == output) {
            if self.wallpapers[i].scale != info.scale_factor {
                self.wallpapers[i].scale = info.scale_factor;
                // A GPU surface's device-pixel target scales with the output,
                // so it must re-render; a solid-color buffer is scale-free.
                if self.wallpapers[i].shader.is_some() {
                    self.draw(qh, i);
                }
            }
        }
        // A move/resize of any output shifts the cluster box; retile static
        // shaders globally (animated ones pick it up on their next frame).
        self.redraw_static_shaders(qh);
    }

    fn output_destroyed(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        self.remove_output(&output);
        // Losing an output shrinks the cluster box; existing static shaders
        // retile around the smaller workspace.
        self.redraw_static_shaders(qh);
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
        qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let Some(i) = self.wallpapers.iter().position(|w| &w.layer == layer) else {
            return;
        };
        let new_size = configure.new_size;
        // Already drawn at this size? (Shader wallpapers have no color
        // buffer; their ring being built is the equivalent signal.)
        let drawn =
            self.wallpapers[i].color_buffer.is_some() || self.wallpapers[i].shader.is_some();
        if new_size == self.wallpapers[i].size && drawn {
            return;
        }
        self.wallpapers[i].size = new_size;
        self.draw(qh, i);
    }
}

impl ShmHandler for App {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl SeatHandler for App {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: WlSeat) {}

    /// Create a themed pointer for this seat the moment it gains the pointer
    /// capability — but only if a shader wants `iMouse`. The theme makes the
    /// surface show a normal cursor (set on each Enter); without `wants_mouse`
    /// we bind nothing, so plain wallpapers never touch input.
    fn new_capability(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        seat: WlSeat,
        capability: Capability,
    ) {
        if capability != Capability::Pointer || !self.wants_mouse {
            return;
        }
        let surface = self.compositor.create_surface(qh);
        match self.seat_state.get_pointer_with_theme::<App, SurfaceData>(
            qh,
            &seat,
            self.shm.wl_shm(),
            surface,
            ThemeSpec::default(),
        ) {
            Ok(pointer) => self.pointers.push(pointer),
            Err(e) => tracing::warn!("pointer init failed; iMouse shaders inert: {e}"),
        }
    }

    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        seat: WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Pointer {
            self.pointers
                .retain(|p| p.pointer().data::<PointerData>().map(|d| d.seat()) != Some(&seat));
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, seat: WlSeat) {
        self.pointers
            .retain(|p| p.pointer().data::<PointerData>().map(|d| d.seat()) != Some(&seat));
    }
}

impl PointerHandler for App {
    /// Route pointer events to the interactive shader under the cursor. Only
    /// `iMouse` shaders have a non-empty input region, so every event we get
    /// belongs to one. A change to the resolved `iMouse` repaints a *static*
    /// shader (repaint-on-motion); animated ones absorb it on their next frame.
    fn pointer_frame(
        &mut self,
        conn: &Connection,
        qh: &QueueHandle<Self>,
        pointer: &WlPointer,
        events: &[PointerEvent],
    ) {
        // Apply every event in the frame first, then redraw each affected
        // static shader at most once: a frame can batch several motions (and a
        // press + motion), and a single render of the final state is both
        // cheaper and more correct (it can't overwrite the press frame's
        // `sign(iMouse.w)` with the motion that followed it in the same batch).
        let mut redraw: HashSet<usize> = HashSet::new();
        for event in events {
            let Some(i) = self
                .wallpapers
                .iter()
                .position(|w| w.layer.wl_surface() == &event.surface)
            else {
                continue;
            };
            // The wallpaper owns pointer focus while the cursor is over it; give
            // it the normal arrow (the compositor leaves the cursor to us).
            if matches!(event.kind, PointerEventKind::Enter { .. }) {
                if let Some(p) = self.pointers.iter().find(|p| p.pointer() == pointer) {
                    let _ = p.set_cursor(conn, CursorIcon::Default);
                }
            }
            let pos = (event.position.0 as f32, event.position.1 as f32);
            let changed = match self.wallpapers[i].shader.as_mut() {
                Some(s) => s.pointer_event(&event.kind, pos),
                None => continue,
            };
            // A static shader must be redrawn to pick the change up; an animated
            // one will on its next frame callback, and a broken one not at all.
            let static_live = !self.wallpapers[i].broken
                && self.wallpapers[i]
                    .shader
                    .as_ref()
                    .is_some_and(|s| !s.animated());
            if changed && static_live {
                redraw.insert(i);
            }
        }
        for i in redraw {
            self.draw(qh, i);
        }
    }
}

impl ProvidesRegistryState for App {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState];
}

impl AsMut<SimpleViewporter> for App {
    fn as_mut(&mut self) -> &mut SimpleViewporter {
        &mut self.viewporter
    }
}

delegate_compositor!(App);
delegate_output!(App);
delegate_shm!(App);
delegate_layer!(App);
delegate_seat!(App);
delegate_pointer!(App);
delegate_registry!(App);
delegate_simple!(App, WpViewporter, 1);
wayland_client::delegate_noop!(App: ignore WpViewport);

/// An output's logical rectangle in layout (y-down) pixels: `((x, y), (w, h))`.
type LogicalRect = ((i32, i32), (i32, i32));

/// Pure cluster-tiling geometry, split from [`App::cluster_placement`] so it
/// can be tested without a live `OutputState`. `this` is the target output's
/// logical rect and `all` is every output's rect — both in y-down layout
/// pixels. Returns the y-up cluster placement (origin at the cluster's
/// bottom-left). When `this` is `None` (no `xdg-output` geometry), falls back
/// to `fallback` (the output's own logical size) as a lone output.
fn tiling_from_rects(
    this: Option<LogicalRect>,
    all: &[LogicalRect],
    fallback: (u32, u32),
) -> Tiling {
    let Some(((px, py), (lw, lh))) = this else {
        let s = [fallback.0 as f32, fallback.1 as f32];
        return Tiling {
            offset: [0.0, 0.0],
            output_size: s,
            global: s,
        };
    };
    // Union of all reported rects, seeded with this output's own (so a `this`
    // missing from `all` still bounds the box correctly).
    let (mut minx, mut miny, mut maxx, mut maxy) = (px, py, px + lw, py + lh);
    for &((ox, oy), (ow, oh)) in all {
        minx = minx.min(ox);
        miny = miny.min(oy);
        maxx = maxx.max(ox + ow);
        maxy = maxy.max(oy + oh);
    }
    Tiling {
        // Bottom-left in y-up space: flip the y-down top edge `py`.
        offset: [(px - minx) as f32, (maxy - (py + lh)) as f32],
        output_size: [lw as f32, lh as f32],
        global: [(maxx - minx) as f32, (maxy - miny) as f32],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiling_lone_output_is_origin() {
        // A single output sits at the origin and spans the whole cluster.
        let r = ((0, 0), (1920, 1080));
        let t = tiling_from_rects(Some(r), &[r], (1920, 1080));
        assert_eq!(t.offset, [0.0, 0.0]);
        assert_eq!(t.output_size, [1920.0, 1080.0]);
        assert_eq!(t.global, [1920.0, 1080.0]);
    }

    #[test]
    fn tiling_missing_geometry_falls_back_to_lone() {
        // No xdg-output info: treat as a lone output at its configured size.
        let t = tiling_from_rects(None, &[], (2560, 1440));
        assert_eq!(t.offset, [0.0, 0.0]);
        assert_eq!(t.output_size, [2560.0, 1440.0]);
        assert_eq!(t.global, [2560.0, 1440.0]);
    }

    #[test]
    fn tiling_side_by_side_continuous_x() {
        // Two 1920x1080 outputs side by side. The right one is offset by the
        // left's width so x is continuous across the seam; both are full
        // height, so y-offset is 0 for both.
        let left = ((0, 0), (1920, 1080));
        let right = ((1920, 0), (1920, 1080));
        let all = [left, right];
        let tl = tiling_from_rects(Some(left), &all, (1920, 1080));
        let tr = tiling_from_rects(Some(right), &all, (1920, 1080));
        assert_eq!(tl.offset, [0.0, 0.0]);
        assert_eq!(tr.offset, [1920.0, 0.0]);
        assert_eq!(tl.global, [3840.0, 1080.0]);
        assert_eq!(tr.global, [3840.0, 1080.0]);
        // The left output's right edge meets the right output's left edge.
        assert_eq!(tl.offset[0] + tl.output_size[0], tr.offset[0]);
    }

    #[test]
    fn tiling_flips_y_to_up_origin() {
        // A tall output (top, y-down 0) above a short one (below it). In y-up
        // cluster space the lower output is at y=0 and the upper one above it.
        let top = ((0, 0), (1000, 600)); // y-down: top edge at 0
        let bottom = ((0, 600), (1000, 400)); // directly below
        let all = [top, bottom];
        let tt = tiling_from_rects(Some(top), &all, (1000, 600));
        let tb = tiling_from_rects(Some(bottom), &all, (1000, 400));
        assert_eq!(tt.global, [1000.0, 1000.0]);
        // Bottom output's bottom-left is the cluster origin.
        assert_eq!(tb.offset, [0.0, 0.0]);
        // Top output sits above it: its bottom edge is at the bottom's height.
        assert_eq!(tt.offset, [0.0, 400.0]);
        // Their edges meet: bottom's top (offset_y + height) == top's bottom.
        assert_eq!(tb.offset[1] + tb.output_size[1], tt.offset[1]);
    }

    #[test]
    fn tiling_handles_negative_origin() {
        // Compositor layouts can place outputs at negative coords (left of
        // the primary). The cluster origin normalizes those away.
        let primary = ((0, 0), (1920, 1080));
        let leftof = ((-1280, 0), (1280, 1024));
        let all = [primary, leftof];
        let tp = tiling_from_rects(Some(primary), &all, (1920, 1080));
        let tl = tiling_from_rects(Some(leftof), &all, (1280, 1024));
        // Leftmost output starts at cluster x=0.
        assert_eq!(tl.offset[0], 0.0);
        // Primary is shifted right by the left output's width.
        assert_eq!(tp.offset[0], 1280.0);
        assert_eq!(tp.global, [3200.0, 1080.0]);
    }
}
