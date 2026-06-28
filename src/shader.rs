//! Shader wallpapers: per-output GPU rendering presented via dmabuf.
//!
//! Each [`ShaderSurface`] renders a fragment shader into fp16 dmabuf
//! [`RenderTarget`]s, imports them as `wl_buffer`s (`zwp_linux_dmabuf_v1`),
//! and attaches them to the layer surface — tagged extended-linear so the
//! output goes through the compositor's calibrated, HDR-aware pipeline like
//! a decoded image.
//!
//! **Multi-GPU**: the surface renders on the GPU that actually drives its
//! output, discovered from the compositor's v4 dmabuf *feedback* (the
//! per-surface `main_device` is that output's render node, and the tranches
//! list the modifiers it can import). Rendering on the wrong GPU would make
//! the compositor detile-and-copy our buffer across the PCIe bus every
//! frame. The device-specific renderer + target ring live in [`DeviceState`]
//! and are rebuilt if the output moves to a different GPU.
//!
//! Animation is driven by frame callbacks (vsync-paced, paused when the
//! surface is occluded). A shader renders once unless it *uses* a
//! self-advancing input — `iTime`/`iTimeDelta`, `iFrame`, `iDate`, the audio
//! uniforms, or evolving buffers — in which case it keeps redrawing. Detection
//! scans uses, not the mandatory positional declarations (see
//! [`usage_scan_source`]). `iMouse` is event-driven, not self-advancing: a
//! static mouse shader repaints only on pointer input.

use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use ash::vk;
use smithay_client_toolkit::seat::pointer::PointerEventKind;
use wayland_client::protocol::{wl_buffer::WlBuffer, wl_surface::WlSurface};
use wayland_client::{Dispatch, QueueHandle};
use wayland_protocols::wp::color_management::v1::client::wp_color_management_surface_v1::WpColorManagementSurfaceV1;
use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_buffer_params_v1::{self, ZwpLinuxBufferParamsV1},
    zwp_linux_dmabuf_feedback_v1::{self, ZwpLinuxDmabufFeedbackV1},
    zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1,
};
use wayland_protocols::wp::viewporter::client::wp_viewport::WpViewport;

use crate::app::App;
use crate::cli::Intent;
use crate::color::{ColorEncoding, PrimaryVolume, Tf};
use crate::colormgmt::{ColorState, DescriptionHandle, Status};
use crate::gpu::{
    AudioUniforms, Gpu, RenderTarget, ShaderRenderer, ShaderUniforms, TextureData, Transition,
    RENDER_DRM_FOURCC,
};

/// Number of dmabuf targets cycled per surface. Three lets the GPU render
/// the next frame while the compositor still references the current one
/// (and the just-previous one drains its release), without ever stalling.
const RING: usize = 3;

/// Shader output color space: extended-linear with sRGB primaries, 1.0 =
/// reference white. The fp16 buffer carries values verbatim and the
/// compositor does the conversion — the same contract as a linear-light
/// image. (Authoring in BT.2020/PQ could be exposed later via flags.)
const SHADER_ENCODING: ColorEncoding = ColorEncoding {
    tf: Tf::Linear,
    primaries: PrimaryVolume::Srgb,
    luminances: None,
};

/// This output's placement in the global multi-monitor cluster, handed to
/// the shader so patterns tile continuously across the workspace. All fields
/// are y-up logical pixels with the origin at the cluster's bottom-left
/// (matching the y-up `fragCoord` the vertex stage emits). A lone output is
/// `offset (0,0)`, `output_size == global`.
#[derive(Clone, Copy)]
pub struct Tiling {
    /// This output's bottom-left corner in cluster space.
    pub offset: [f32; 2],
    /// This output's logical size.
    pub output_size: [f32; 2],
    /// The whole cluster's logical size.
    pub global: [f32; 2],
}

/// The dmabuf global (bound at v4 for feedback). Per-surface feedback
/// objects are created on demand from [`ShaderSurface`].
pub struct DmabufState {
    pub proxy: ZwpLinuxDmabufV1,
}

/// Per-surface dmabuf-feedback userdata: the output name, so the dispatch
/// handler can find the wallpaper this feedback is for.
#[derive(Debug, Clone)]
pub struct FeedbackId(pub String);

/// What the compositor's per-surface feedback resolved to: the GPU that
/// drives this output (DRM render node `dev_t`) and the modifiers it can
/// import for [`RENDER_DRM_FOURCC`].
#[derive(Debug, Clone)]
pub struct ResolvedFeedback {
    pub device: u64,
    pub modifiers: Vec<u64>,
}

/// Accumulates one feedback "batch" (the events between `done`s). The
/// format table persists across batches (only re-sent when it changes);
/// the per-tranche scratch and collected modifiers reset each batch.
#[derive(Default)]
struct FeedbackAccum {
    /// Set after a `done`; the next event starts a fresh batch (reset).
    batch_done: bool,
    main_device: Option<u64>,
    /// (fourcc, modifier) by table index, mmap'd from `format_table`.
    format_table: Vec<(u32, u64)>,
    tranche_target: Option<u64>,
    tranche_indices: Vec<u16>,
    /// Modifiers for our fourcc gathered from this batch's tranches.
    collected: Vec<u64>,
}

impl FeedbackAccum {
    /// Handle one feedback event. Returns the freshly resolved feedback on
    /// `done`, else `None`.
    fn event(&mut self, event: zwp_linux_dmabuf_feedback_v1::Event) -> Option<ResolvedFeedback> {
        use zwp_linux_dmabuf_feedback_v1::Event;
        // A new batch begins on the first event after a `done`.
        if self.batch_done && !matches!(event, Event::Done) {
            self.batch_done = false;
            self.main_device = None;
            self.tranche_target = None;
            self.tranche_indices.clear();
            self.collected.clear();
        }
        match event {
            Event::MainDevice { device } => self.main_device = dev_from_bytes(&device),
            Event::FormatTable { fd, size } => {
                self.format_table = parse_format_table(fd, size);
            }
            Event::TrancheTargetDevice { device } => self.tranche_target = dev_from_bytes(&device),
            Event::TrancheFormats { indices } => {
                self.tranche_indices = indices
                    .chunks_exact(2)
                    .map(|c| u16::from_ne_bytes([c[0], c[1]]))
                    .collect();
            }
            Event::TrancheDone => {
                // Only this tranche's modifiers if it targets the device we
                // render on — a multi-GPU feedback can carry tranches for
                // other devices, whose modifiers we can't import.
                if self.tranche_target == self.main_device {
                    for &idx in &self.tranche_indices {
                        if let Some(&(fourcc, modifier)) = self.format_table.get(idx as usize) {
                            if fourcc == RENDER_DRM_FOURCC && !self.collected.contains(&modifier) {
                                self.collected.push(modifier);
                            }
                        }
                    }
                }
                self.tranche_target = None;
                self.tranche_indices.clear();
            }
            Event::Done => {
                self.batch_done = true;
                if let Some(device) = self.main_device {
                    return Some(ResolvedFeedback {
                        device,
                        modifiers: self.collected.clone(),
                    });
                }
            }
            _ => {}
        }
        None
    }
}

/// One target in the ring: GPU image+dmabuf, its framebuffer, the imported
/// `wl_buffer`, and a flag the compositor flips when it releases the buffer.
struct RingTarget {
    target: RenderTarget,
    framebuffer: vk::Framebuffer,
    buffer: WlBuffer,
    /// True when the buffer is free to render into (not held by the
    /// compositor). Starts free; cleared on attach, set on `wl_buffer.release`.
    available: Arc<AtomicBool>,
}

/// An in-progress cross-blur-dissolve between the previous source (`outgoing`)
/// and the current one (`DeviceState::renderer`, the incoming). Each frame both
/// render offscreen via `blend`, which composites them into the dmabuf by
/// `progress`. Dropped once `progress` reaches 1 (the incoming becomes the sole
/// source). The outgoing keeps its own animation clock so a departing animated
/// shader continues from where it was rather than restarting.
struct ActiveTransition {
    outgoing: ShaderRenderer,
    blend: Transition,
    /// Wall-clock start of the dissolve, and its total duration.
    started: Instant,
    duration: Duration,
    /// The outgoing source's independent `iTime`/`iTimeDelta`/`iFrame` state,
    /// carried over from before the swap.
    out_started: Option<Instant>,
    out_last_time: Option<f32>,
    out_frame_count: i32,
}

/// The device-specific half of a shader surface: the pipeline and target
/// ring on one GPU. Rebuilt if the output moves to a different GPU.
struct DeviceState {
    /// DRM render-node `dev_t` this state's GPU corresponds to.
    device_dev: u64,
    /// Device handle for framebuffer/buffer teardown in `Drop` (ash idiom);
    /// the owning `GpuPool` keeps every device alive past these.
    device: ash::Device,
    renderer: ShaderRenderer,
    ring: Vec<RingTarget>,
    /// Device-pixel size of the targets.
    size: (u32, u32),
    /// Active blur-dissolve transition, if one is running.
    transition: Option<ActiveTransition>,
}

impl Drop for DeviceState {
    fn drop(&mut self) {
        // SAFETY: wait for in-flight work, then free framebuffers and
        // wl_buffers before the ring's RenderTargets (their images) drop.
        unsafe {
            let _ = self.device.device_wait_idle();
            for t in &self.ring {
                self.device.destroy_framebuffer(t.framebuffer, None);
                t.buffer.destroy();
            }
        }
        // ring (RenderTargets) and renderer drop after this body, on the
        // still-valid device.
    }
}

/// Linux button code for the left mouse button (`BTN_LEFT`), the button
/// Shadertoy's `iMouse` tracks.
const BTN_LEFT: u32 = 0x110;

/// Accumulated pointer state for a shader surface, in surface-local logical
/// pixels (y-down, as `wl_pointer` delivers). Resolved into the device-pixel,
/// y-up `iMouse` vec4 at render time, where the surface size is known.
#[derive(Clone, Copy, Default)]
struct PointerState {
    /// Cursor position while the left button is (or was last) held.
    pos: (f32, f32),
    /// Position of the most recent left-button press.
    click: (f32, f32),
    /// Whether the left button is currently down.
    down: bool,
    /// True from a press until the next render consumes it (drives the
    /// `sign(iMouse.w)` "clicked this frame" bit).
    clicked_frame: bool,
    /// Whether a press has ever happened (until then `iMouse` is all-zero).
    ever: bool,
}

impl PointerState {
    /// Apply one pointer event. `pos` is the event's surface-local position.
    /// Returns whether the resolved `iMouse` value changed (so the caller can
    /// redraw a static shader). Hover motion with no button held is ignored, to
    /// match Shadertoy (`iMouse.xy` only tracks while pressed) and to avoid
    /// waking a static shader on every mouse move.
    fn apply(&mut self, kind: &PointerEventKind, pos: (f32, f32)) -> bool {
        match kind {
            PointerEventKind::Press {
                button: BTN_LEFT, ..
            } => {
                self.pos = pos;
                self.click = pos;
                self.down = true;
                self.clicked_frame = true;
                self.ever = true;
                true
            }
            PointerEventKind::Release {
                button: BTN_LEFT, ..
            } => {
                let was = self.down;
                self.down = false;
                was
            }
            // Leaving the surface ends any drag (no further motion arrives).
            PointerEventKind::Leave { .. } => {
                let was = self.down;
                self.down = false;
                was
            }
            PointerEventKind::Motion { .. } if self.down => {
                self.pos = pos;
                true
            }
            _ => false,
        }
    }

    /// Resolve the `iMouse` vec4 in device pixels, y-up. `size` is the device
    /// resolution, `logical` the surface's logical size (the space pointer
    /// events arrive in); the ratio handles output scale and fractional scaling.
    fn uniform(&self, size: (u32, u32), logical: (u32, u32)) -> [f32; 4] {
        if !self.ever {
            return [0.0; 4];
        }
        let to_device = |p: (f32, f32)| {
            let nx = p.0 / logical.0.max(1) as f32;
            let ny = p.1 / logical.1.max(1) as f32;
            // y-down logical → y-up device.
            (nx * size.0 as f32, (1.0 - ny) * size.1 as f32)
        };
        let (px, py) = to_device(self.pos);
        let (cx, cy) = to_device(self.click);
        [
            px,
            py,
            if self.down { cx } else { -cx },
            if self.clicked_frame { cy } else { -cy },
        ]
    }
}

/// Current local wall-clock as the Shadertoy `iDate` vec4: `(year, month
/// [0-11], day-of-month, seconds-since-midnight)`, the last component fractional
/// for a smooth sweep. Goes through `localtime_r`, so it honors the system
/// timezone (Shadertoy's `iDate` is local, not UTC).
fn local_date() -> [f32; 4] {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as libc::time_t;
    let frac = now.subsec_nanos() as f32 / 1.0e9;
    // SAFETY: `localtime_r` is the reentrant variant (touches no shared static);
    // it reads `secs` and writes a fully-initialized `tm` into our stack slot.
    // On failure it returns null and leaves `tm` zeroed (date reads as epoch),
    // which is a harmless degradation for a wallpaper clock.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe {
        libc::localtime_r(&secs, &mut tm);
    }
    [
        (tm.tm_year + 1900) as f32,
        tm.tm_mon as f32,
        tm.tm_mday as f32,
        (tm.tm_hour * 3600 + tm.tm_min * 60 + tm.tm_sec) as f32 + frac,
    ]
}

/// Decode a static image channel to 8-bit RGBA for upload. Path is resolved
/// relative to the `.frag` file's directory. Pixels are kept sRGB-encoded as
/// stored; the GPU format chosen at upload decides interpretation — raw `UNORM`
/// (Shadertoy default, `tex.srgb == false`) passes the stored value straight to
/// the shader, while `SRGB` linearizes on read for color images. HDR/wide-gamut
/// sources are flattened to 8-bit sRGB (texture color management is a later
/// refinement; authors handle HDR in-shader for now).
fn load_texture(
    base_dir: &std::path::Path,
    tex: &crate::shadergraph::TextureSpec,
) -> Result<TextureData> {
    let path = base_dir.join(&tex.path);
    let decoded = crate::decode::load(&path)
        .with_context(|| format!("loading texture {:?} ({})", tex.name, path.display()))?;
    let img8 = decoded.quantized_to_8bit(crate::color::Tf::Srgb);
    let crate::decode::Pixels::Rgba8(rgba) = img8.pixels else {
        bail!(
            "texture {:?}: expected 8-bit pixels after quantization",
            tex.name
        );
    };
    Ok(TextureData {
        width: img8.width,
        height: img8.height,
        pixels: crate::gpu::TexturePixels::Rgba8 {
            data: rgba,
            srgb: tex.srgb,
        },
    })
}

/// A copy of shader source with comments and the `push_constant` block removed,
/// so a substring scan detects USES of a uniform (`pc.iTime`) rather than its
/// mandatory positional declaration (`float iTime;`). The push block is
/// positional — to read a late field a shader must declare every field before
/// it — so scanning raw source would flag e.g. every `iMouse` shader as
/// animated just for declaring `iTime`. Used only for feature detection; the
/// stripped text is never compiled.
pub(crate) fn usage_scan_source(source: &str) -> String {
    strip_push_blocks(&strip_comments(source))
}

/// Remove `//` line and `/* */` block comments (delimiters are ASCII, so byte
/// scanning never splits a multibyte char). Also drops `//!pass`/`//!common`
/// directives, which is fine — only substring detection reads the result.
fn strip_comments(source: &str) -> String {
    let b = source.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Remove each `layout(push_constant …) uniform … { … }` block, leaving only
/// the surrounding `layout(` / instance name (neither carries field names).
/// Brace-matched so it stops at the block's close. Keys on the `push_constant`
/// token, not the exact `layout(push_constant)` spelling, so qualifier variants
/// like `layout(push_constant, std430)` or `layout( push_constant )` are caught.
fn strip_push_blocks(source: &str) -> String {
    let mut out = source.to_string();
    while let Some(start) = out.find("push_constant") {
        let Some(rel_open) = out[start..].find('{') else {
            break; // malformed; the real compile will report it
        };
        let open = start + rel_open;
        let bytes = out.as_bytes();
        let mut depth = 0i32;
        let mut close = None;
        for (i, &c) in bytes.iter().enumerate().skip(open) {
            match c {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        match close {
            Some(end) => out.replace_range(start..=end, " "),
            None => break, // unbalanced; leave it (won't compile anyway)
        }
    }
    out
}

pub struct ShaderSurface {
    /// The shader resolved into its render graph; compiled per GPU into a
    /// `DeviceState`.
    spec: crate::shadergraph::GraphSpec,
    /// Decoded static image channels (8-bit sRGB), in `spec.textures` order.
    /// Loaded once on the CPU; each [`DeviceState`] uploads its own copy.
    textures: Vec<TextureData>,
    /// Whether the shader needs continuous redraw (self-advancing inputs:
    /// `iTime`/`iTimeDelta`, `iFrame`, `iDate`, audio, or buffers). Drives
    /// frame-callback animation; a static shader renders once.
    animated: bool,
    /// Whether the shader references the audio uniforms (`iAudio*`); if any
    /// surface does, the app spins up the PipeWire capture.
    uses_audio: bool,
    /// Whether the shader references `iMouse`; if any surface does, the app
    /// binds a seat pointer and makes the interactive surfaces input-receiving.
    uses_mouse: bool,
    /// Pointer state, in surface-local logical pixels with a y-down origin (as
    /// delivered by `wl_pointer`); converted to the device-pixel, y-up `iMouse`
    /// convention at render time. See [`Self::mouse_uniform`].
    ptr: PointerState,
    /// iTime origin, set on the first rendered frame.
    started: Option<Instant>,
    /// `iTime` of the previous rendered frame, for `iTimeDelta`; `None` until
    /// the first frame.
    last_time: Option<f32>,
    /// Frames rendered so far (`iFrame`); `0` on the first frame.
    frame_count: i32,
    /// `--fps` cap as a minimum interval between renders; `None` is uncapped
    /// (vsync). Only throttles animated shaders.
    min_interval: Option<Duration>,
    /// Wall-clock of the last actual render, for the `--fps` throttle.
    last_render: Option<Instant>,
    /// Extended-linear description for the surface; `None` without CM.
    description: Option<DescriptionHandle>,
    cm: Option<WpColorManagementSurfaceV1>,
    tagged: bool,
    /// Per-surface dmabuf feedback (kept alive) + its accumulator + result.
    feedback: Option<ZwpLinuxDmabufFeedbackV1>,
    accum: FeedbackAccum,
    resolved: Option<ResolvedFeedback>,
    /// Renderer + target ring on the resolved GPU.
    state: Option<DeviceState>,
    /// The graph/textures were swapped (playlist rotation): rebuild the
    /// renderer in place on the next render, keeping the dmabuf ring.
    source_dirty: bool,
    /// When `Some`, the next source swap blur-dissolves over this duration
    /// instead of cutting; consumed when the transition is built. Set by
    /// [`Self::set_source`] with a fade.
    pending_fade: Option<Duration>,
    /// The outgoing source's animation clock `(started, last_time, frame_count)`,
    /// snapshotted at the swap so a departing animated shader keeps advancing
    /// through the dissolve. Paired with `pending_fade`.
    outgoing_timing: Option<(Option<Instant>, Option<f32>, i32)>,
}

/// Build the compositor-side color description for `encoding`, or `None`
/// without color management (the surface attaches untagged).
fn make_description(
    qh: &QueueHandle<App>,
    color: Option<&ColorState>,
    encoding: &ColorEncoding,
) -> Option<DescriptionHandle> {
    match color {
        Some(c) => match c.create_description(qh, encoding) {
            Ok(d) => Some(d),
            Err(e) => {
                tracing::warn!("surface color description unavailable: {e:#}; attaching untagged");
                None
            }
        },
        None => None,
    }
}

/// A fully-prepared wallpaper source — image or shader — ready to display in a
/// [`ShaderSurface`]. Carries the render graph, its static textures, the dmabuf
/// color tag, and the per-source behavior (animation, audio/mouse use, frame
/// cap). Built by the app from a static spec or a playlist entry, then handed to
/// [`ShaderSurface::from_source`] (first load) or [`ShaderSurface::set_source`]
/// (rotation). The two source kinds differ only in these fields, so the surface
/// and its transition machinery treat them uniformly.
pub struct PreparedSource {
    pub graph: crate::shadergraph::GraphSpec,
    pub textures: Vec<TextureData>,
    pub encoding: ColorEncoding,
    pub animated: bool,
    pub uses_audio: bool,
    pub uses_mouse: bool,
    pub min_interval: Option<Duration>,
}

impl PreparedSource {
    /// Prepare a shader source from `fragment_glsl`: parse + validate the graph
    /// (any `/*!prism …*/` metadata), load its static texture channels (paths
    /// relative to `base_dir`), and detect its behavior — so a bad shader fails
    /// here rather than silently on the GPU. Tagged extended-linear
    /// ([`SHADER_ENCODING`]).
    pub fn shader(
        fragment_glsl: &str,
        base_dir: &std::path::Path,
        fps: Option<u32>,
    ) -> Result<PreparedSource> {
        let spec = crate::shadergraph::parse(fragment_glsl)?;
        crate::gpu::validate_graph(&spec)?;
        let textures = spec
            .textures
            .iter()
            .map(|t| load_texture(base_dir, t))
            .collect::<Result<Vec<_>>>()?;
        // Feature detection scans for USES, not declarations: the push block is
        // positional, so a shader that reads any late field must declare every
        // field before it — scanning the raw source would flag e.g. every iMouse
        // shader as "uses iTime" merely for the mandatory `float iTime;`. The
        // scan copy has comments and the push block stripped, leaving only
        // accesses like `pc.iTime`. See [`usage_scan_source`].
        let scan = usage_scan_source(fragment_glsl);
        let uses_audio = scan.contains("iAudio");
        // A shader that reads iMouse wants pointer input (only wired for static
        // --shader specs; a playlist shader using iMouse reads zero). It does
        // *not* make the shader animated — a static mouse shader renders once
        // and then only on pointer events.
        let uses_mouse = scan.contains("iMouse");
        // Anything that advances on its own — iTime/iTimeDelta (the latter
        // contains "iTime"), the per-frame counter iFrame, the wall clock iDate,
        // audio, or evolving buffers — needs continuous redraw to progress.
        let animated = scan.contains("iTime")
            || scan.contains("iFrame")
            || scan.contains("iDate")
            || uses_audio
            || spec.has_buffers();
        if fps.is_some() && !animated {
            tracing::warn!("--fps ignored: shader is static (no iTime), renders a single frame");
        }
        // Cap as a minimum inter-render interval; only meaningful when animated.
        let min_interval = fps
            .filter(|_| animated)
            .map(|n| Duration::from_secs_f64(1.0 / n as f64));
        Ok(PreparedSource {
            graph: spec,
            textures,
            encoding: SHADER_ENCODING,
            animated,
            uses_audio,
            uses_mouse,
            min_interval,
        })
    }

    /// Prepare an image source: a degenerate single-pass graph (from
    /// [`crate::gpu::image_graph`]) over the image already converted to
    /// working-space fp16, tagged with that working space. Static — no
    /// animation, audio, or mouse.
    pub fn image(
        graph: crate::shadergraph::GraphSpec,
        texture: TextureData,
        encoding: ColorEncoding,
    ) -> PreparedSource {
        PreparedSource {
            graph,
            textures: vec![texture],
            encoding,
            animated: false,
            uses_audio: false,
            uses_mouse: false,
            min_interval: None,
        }
    }
}

/// Parse + validate a shader file without loading its textures — the cheap
/// "is this entry usable?" check used when seeking/rotating a playlist past
/// broken entries. Texture-load failures surface later, at display time.
pub fn validate_shader_file(path: &std::path::Path) -> Result<()> {
    let glsl = std::fs::read_to_string(path)
        .with_context(|| format!("reading shader {}", path.display()))?;
    let spec = crate::shadergraph::parse(&glsl)?;
    crate::gpu::validate_graph(&spec)?;
    Ok(())
}

impl ShaderSurface {
    /// Prepare a shader surface from `fragment_glsl`. Compilation and target
    /// allocation are deferred until feedback resolves the output's GPU.
    pub fn new(
        qh: &QueueHandle<App>,
        fragment_glsl: &str,
        base_dir: &std::path::Path,
        color: Option<&ColorState>,
        fps: Option<u32>,
    ) -> Result<ShaderSurface> {
        Ok(Self::from_source(
            qh,
            PreparedSource::shader(fragment_glsl, base_dir, fps)?,
            color,
        ))
    }

    /// Build a surface displaying `src` (image or shader). The GPU build is
    /// deferred until feedback resolves the output's device.
    pub fn from_source(
        qh: &QueueHandle<App>,
        src: PreparedSource,
        color: Option<&ColorState>,
    ) -> ShaderSurface {
        let description = make_description(qh, color, &src.encoding);
        ShaderSurface {
            spec: src.graph,
            textures: src.textures,
            animated: src.animated,
            uses_audio: src.uses_audio,
            uses_mouse: src.uses_mouse,
            ptr: PointerState::default(),
            last_time: None,
            frame_count: 0,
            started: None,
            min_interval: src.min_interval,
            last_render: None,
            description,
            cm: None,
            tagged: false,
            feedback: None,
            accum: FeedbackAccum::default(),
            resolved: None,
            state: None,
            source_dirty: false,
            pending_fade: None,
            outgoing_timing: None,
        }
    }

    /// Swap the displayed source in place (playlist rotation): replace the graph,
    /// textures, behavior flags, and color tag, keeping the surface, dmabuf ring,
    /// and feedback resolution. With `fade: None` the new source renders on the
    /// next draw and the old frame stays presented until then (a flash-free hard
    /// cut). With `fade: Some(d)` the swap instead blur-dissolves from the old
    /// source to the new one over `d` (see [`ActiveTransition`]); the surface
    /// keeps requesting frame callbacks until the dissolve completes. Works for
    /// any source pair — image↔image, image↔shader, shader↔shader.
    pub fn set_source(
        &mut self,
        qh: &QueueHandle<App>,
        src: PreparedSource,
        color: Option<&ColorState>,
        fade: Option<Duration>,
    ) {
        // Snapshot the outgoing source's clock before it's reset for the
        // incoming, so a departing animated shader keeps advancing mid-dissolve.
        if fade.is_some() {
            self.outgoing_timing = Some((self.started, self.last_time, self.frame_count));
            self.pending_fade = fade;
        }
        self.spec = src.graph;
        self.textures = src.textures;
        self.animated = src.animated;
        self.uses_audio = src.uses_audio;
        self.uses_mouse = src.uses_mouse;
        self.min_interval = src.min_interval;
        self.source_dirty = true;
        // Re-tag with the new source's working space.
        if let Some(cm) = self.cm.take() {
            cm.destroy();
        }
        self.description = make_description(qh, color, &src.encoding);
        self.tagged = false;
        // The incoming source renders one frame from t = 0, unthrottled (so the
        // dissolve's first frame isn't gated by the outgoing source's --fps).
        self.started = None;
        self.last_time = None;
        self.frame_count = 0;
        self.last_render = None;
    }

    /// Whether the surface should keep requesting frame callbacks: an animated
    /// shader, or an in-progress / about-to-start blur-dissolve transition.
    pub fn needs_redraw(&self) -> bool {
        self.animated
            || self.pending_fade.is_some()
            || self.state.as_ref().is_some_and(|s| s.transition.is_some())
    }

    pub fn animated(&self) -> bool {
        self.animated
    }

    /// Whether this shader references the audio uniforms (`iAudio*`).
    pub fn uses_audio(&self) -> bool {
        self.uses_audio
    }

    /// Whether this shader references `iMouse` (and so wants pointer input).
    pub fn uses_mouse(&self) -> bool {
        self.uses_mouse
    }

    /// Apply a pointer event (surface-local position `pos`). Returns whether the
    /// resolved `iMouse` changed — the caller redraws a static shader when so
    /// (an animated one picks the new value up on its next frame anyway).
    pub fn pointer_event(&mut self, kind: &PointerEventKind, pos: (f32, f32)) -> bool {
        self.ptr.apply(kind, pos)
    }

    /// Subscribe to per-surface dmabuf feedback for `surface`. The result
    /// (output GPU + importable modifiers) arrives via [`Self::feedback_event`].
    pub fn request_feedback(
        &mut self,
        dmabuf: &DmabufState,
        qh: &QueueHandle<App>,
        surface: &WlSurface,
        output: String,
    ) {
        let fb = dmabuf
            .proxy
            .get_surface_feedback(surface, qh, FeedbackId(output));
        self.feedback = Some(fb);
    }

    /// Feed one feedback event. Returns `true` when a `done` produced a new
    /// resolution (the caller should (re)draw to pick up the GPU/modifiers).
    pub fn feedback_event(&mut self, event: zwp_linux_dmabuf_feedback_v1::Event) -> bool {
        if let Some(resolved) = self.accum.event(event) {
            let changed = self.resolved.as_ref().map(|r| r.device) != Some(resolved.device);
            if changed {
                tracing::info!(
                    device = format!("{:#x}", resolved.device),
                    modifiers = resolved.modifiers.len(),
                    "shader output GPU resolved from dmabuf feedback"
                );
            }
            self.resolved = Some(resolved);
            return true;
        }
        false
    }

    /// The DRM `dev_t` of the GPU this surface should render on, once
    /// feedback has resolved it.
    pub fn resolved_device(&self) -> Option<u64> {
        self.resolved.as_ref().map(|r| r.device)
    }

    /// Render the next frame on `gpu` (which must drive this output) and
    /// present it on `surface`, scaling the device-pixel buffer to `logical`
    /// via `viewport`. (Re)builds the pipeline if the GPU changed and the
    /// ring if the size changed. Returns whether the surface is animated.
    #[allow(clippy::too_many_arguments)]
    pub fn render_frame(
        &mut self,
        gpu: &Gpu,
        dmabuf: &DmabufState,
        qh: &QueueHandle<App>,
        surface: &WlSurface,
        viewport: &WpViewport,
        size: (u32, u32),
        logical: (u32, u32),
        tiling: Tiling,
        audio: &AudioUniforms,
        // This output's `(reference_white, peak)` luminance in cd/m² — the
        // shader's `iRefWhite`/`iMaxLum`. `1.0` maps to `reference_white`;
        // highlight headroom is `peak / reference_white`.
        lum: (f32, f32),
        color: Option<&ColorState>,
        intent: Intent,
    ) -> Result<bool> {
        if size.0 == 0 || size.1 == 0 {
            return Ok(self.animated);
        }

        // --fps cap: if we rendered too recently, skip this callback's render
        // but re-arm the next one with a cheap empty commit so animation keeps
        // ticking. Staying frame-callback-driven preserves the occluded-surface
        // pause (no callback while hidden → no wasted GPU). The 5% tolerance
        // keeps a cap that's near a vsync divisor (e.g. 30 on 60Hz) from
        // jittering down to the next-lower divisor; the cost is a hair of
        // overshoot, never undershoot.
        if let (Some(min), Some(last)) = (self.min_interval, self.last_render) {
            if last.elapsed() < min.mul_f64(0.95) {
                surface.frame(qh, surface.clone());
                surface.commit();
                return Ok(self.animated);
            }
        }

        let device_dev = self
            .resolved
            .as_ref()
            .context("render_frame before feedback resolved")?
            .device;

        // Modifiers both the compositor can import on this GPU (feedback)
        // and the GPU can render into — so no implicit detile on either side.
        let importable = &self.resolved.as_ref().unwrap().modifiers;
        let candidates: Vec<(u64, u32)> = gpu
            .renderable_modifiers()
            .into_iter()
            .filter(|(m, _)| importable.contains(m))
            .collect();
        if candidates.is_empty() {
            bail!(
                "no fp16 DRM modifier is both renderable on this GPU and importable by the \
                 compositor for this output ({} compositor modifiers)",
                importable.len()
            );
        }

        // (Re)build pipeline if the GPU changed.
        if self.state.as_ref().map(|s| s.device_dev) != Some(device_dev) {
            self.state = Some(DeviceState {
                device_dev,
                device: gpu.device.clone(),
                renderer: ShaderRenderer::new(gpu, &self.spec, &self.textures, RING)?,
                ring: Vec::new(),
                size: (0, 0),
                transition: None,
            });
        }
        // Source swapped (rotation): build the incoming renderer and recreate
        // the ring framebuffers against its render pass (the dmabuf ring itself
        // is kept). With a pending fade the *old* renderer is retained as the
        // outgoing half of a blur-dissolve (see [`ActiveTransition`]); otherwise
        // it's dropped for a flash-free hard cut. Skipped when there's no ring
        // yet (the GPU-change branch above already built the new spec).
        if self.source_dirty {
            if let Some(st) = self.state.as_mut() {
                if !st.ring.is_empty() {
                    let mut incoming = ShaderRenderer::new(gpu, &self.spec, &self.textures, RING)?;
                    // SAFETY: drain in-flight work before swapping framebuffers.
                    unsafe {
                        let _ = st.device.device_wait_idle();
                        for t in &st.ring {
                            st.device.destroy_framebuffer(t.framebuffer, None);
                        }
                    }
                    let mut fbs = Vec::with_capacity(st.ring.len());
                    for t in &st.ring {
                        fbs.push(incoming.create_framebuffer(&t.target)?);
                    }
                    for (t, fb) in st.ring.iter_mut().zip(fbs) {
                        t.framebuffer = fb;
                    }
                    incoming.resize(gpu, st.size.0, st.size.1)?;
                    // Supersede any still-running transition: its incoming (the
                    // current `st.renderer`) becomes this dissolve's outgoing.
                    st.transition = None;
                    match self.pending_fade.take() {
                        Some(duration) => {
                            let mut blend = Transition::new(gpu, RING)?;
                            blend.resize(gpu, st.size.0, st.size.1)?;
                            let (out_started, out_last_time, out_frame_count) =
                                self.outgoing_timing.take().unwrap_or((None, None, 0));
                            let outgoing = std::mem::replace(&mut st.renderer, incoming);
                            st.transition = Some(ActiveTransition {
                                outgoing,
                                blend,
                                started: Instant::now(),
                                duration,
                                out_started,
                                out_last_time,
                                out_frame_count,
                            });
                        }
                        None => {
                            self.outgoing_timing = None;
                            st.renderer = incoming; // old renderer dropped here
                        }
                    }
                }
            }
            self.source_dirty = false;
            // If a fade was requested but no transition could start (no ring
            // yet — the source changed before the first render), drop it so the
            // fresh source just renders normally rather than leaving the surface
            // perpetually "needs redraw".
            self.pending_fade = None;
            self.outgoing_timing = None;
        }

        // (Re)build the ring if the size changed.
        let need_ring = {
            let st = self.state.as_ref().unwrap();
            st.ring.is_empty() || st.size != size
        };
        if need_ring {
            // A resize mid-dissolve abandons it (the outgoing/blend targets are
            // the old size): snap to the incoming, which build_ring resizes.
            if let Some(st) = self.state.as_mut() {
                st.transition = None;
            }
            self.build_ring(gpu, dmabuf, qh, size, &candidates)?;
        }

        // Tag the surface once the description is ready (idempotent).
        self.tag_if_ready(surface, color, intent, qh);

        let st = self.state.as_mut().unwrap();
        let Some(idx) = st
            .ring
            .iter()
            .position(|t| t.available.load(Ordering::Acquire))
        else {
            tracing::trace!("no free shader target this frame; skipping");
            return Ok(self.animated);
        };
        let time = match self.started {
            Some(t) => t.elapsed().as_secs_f32(),
            None => {
                self.started = Some(Instant::now());
                0.0
            }
        };
        // Wall-clock seconds since the previous rendered frame (0 on the first).
        let time_delta = self.last_time.map_or(0.0, |prev| (time - prev).max(0.0));
        let mouse = self.ptr.uniform(size, logical);
        // The "clicked this frame" bit is one-shot: clear it now that this
        // render has captured it, so the next frame sees sign(iMouse.w) < 0.
        self.ptr.clicked_frame = false;
        let uniforms = ShaderUniforms {
            resolution: [size.0 as f32, size.1 as f32],
            time,
            _pad: 0.0,
            output_offset: tiling.offset,
            output_size: tiling.output_size,
            global_resolution: tiling.global,
            ref_white: lum.0,
            max_lum: lum.1,
            mouse,
            date: local_date(),
            time_delta,
            frame: self.frame_count,
        };
        // Disjoint borrows of the renderer, ring, and transition so a blend can
        // sample the incoming renderer (`renderer`) while mutating the
        // transition state.
        let DeviceState {
            renderer,
            ring,
            transition,
            ..
        } = st;
        let rt = &ring[idx];
        let mut transition_done = false;
        let keep;
        match transition.as_mut() {
            Some(tr) => {
                // Dissolve progress, clamped; the final frame lands exactly at 1.
                let dur = tr.duration.as_secs_f32().max(1e-3);
                let progress = (tr.started.elapsed().as_secs_f32() / dur).clamp(0.0, 1.0);
                // The outgoing source advances on its own carried-over clock.
                let out_time = tr.out_started.map_or(0.0, |t| t.elapsed().as_secs_f32());
                let out_delta = tr.out_last_time.map_or(0.0, |p| (out_time - p).max(0.0));
                let out_uniforms = ShaderUniforms {
                    time: out_time,
                    time_delta: out_delta,
                    frame: tr.out_frame_count,
                    ..uniforms
                };
                tr.blend
                    .render_source(0, &tr.outgoing, idx, &out_uniforms, audio)
                    .context("rendering outgoing source")?;
                tr.blend
                    .render_source(1, renderer, idx, &uniforms, audio)
                    .context("rendering incoming source")?;
                tr.blend
                    .blend(
                        idx,
                        &rt.target,
                        rt.framebuffer,
                        progress,
                        tr.outgoing.frame_semaphore(idx),
                        renderer.frame_semaphore(idx),
                    )
                    .context("blending transition frame")?;
                tr.out_last_time = Some(out_time);
                tr.out_frame_count = tr.out_frame_count.wrapping_add(1);
                transition_done = progress >= 1.0;
                // Keep ticking until the dissolve ends; after that, only if the
                // incoming is itself animated.
                keep = !transition_done || self.animated;
            }
            None => {
                renderer
                    .render(idx, &rt.target, rt.framebuffer, &uniforms, audio)
                    .context("rendering shader frame")?;
                keep = self.animated;
            }
        }
        // Advance the incoming per-frame counters now the render is committed to.
        self.last_time = Some(time);
        self.frame_count = self.frame_count.wrapping_add(1);
        rt.available.store(false, Ordering::Release);
        surface.attach(Some(&rt.buffer), 0, 0);
        self.last_render = Some(Instant::now());
        viewport.set_source(-1.0, -1.0, -1.0, -1.0);
        viewport.set_destination(logical.0 as i32, logical.1 as i32);
        surface.damage_buffer(0, 0, i32::MAX, i32::MAX);
        if keep {
            // Request the next callback before committing; delivered when
            // ready (and not while occluded).
            surface.frame(qh, surface.clone());
        }
        surface.commit();
        // A finished dissolve drops here, leaving the incoming as the sole source.
        if transition_done {
            *transition = None;
        }
        Ok(self.animated)
    }

    fn build_ring(
        &mut self,
        gpu: &Gpu,
        dmabuf: &DmabufState,
        qh: &QueueHandle<App>,
        size: (u32, u32),
        candidates: &[(u64, u32)],
    ) -> Result<()> {
        let st = self.state.as_mut().unwrap();
        // Drop the old ring first (frees framebuffers/buffers/images).
        if !st.ring.is_empty() {
            unsafe {
                let _ = st.device.device_wait_idle();
                for t in &st.ring {
                    st.device.destroy_framebuffer(t.framebuffer, None);
                    t.buffer.destroy();
                }
            }
            st.ring.clear();
        }
        for _ in 0..RING {
            let target = RenderTarget::new(gpu, size.0, size.1, candidates)
                .context("creating shader render target")?;
            let framebuffer = st.renderer.create_framebuffer(&target)?;
            let available = Arc::new(AtomicBool::new(true));
            let buffer = import_dmabuf(dmabuf, qh, &target, available.clone());
            st.ring.push(RingTarget {
                target,
                framebuffer,
                buffer,
                available,
            });
        }
        // Multi-pass / feedback shaders: (re)create the ping-pong buffer
        // textures at the new size. No-op for a plain single-pass renderer.
        st.renderer.resize(gpu, size.0, size.1)?;
        st.size = size;
        tracing::info!(
            width = size.0,
            height = size.1,
            modifier = format!("{:#x}", st.ring[0].target.modifier),
            "shader targets (re)created"
        );
        Ok(())
    }

    fn tag_if_ready(
        &mut self,
        surface: &WlSurface,
        color: Option<&ColorState>,
        intent: Intent,
        qh: &QueueHandle<App>,
    ) {
        if self.tagged {
            return;
        }
        let (Some(color), Some(desc)) = (color, &self.description) else {
            self.tagged = true; // nothing to tag (no CM)
            return;
        };
        match desc.status() {
            Status::Ready => {
                self.cm = Some(color.tag_surface(qh, surface, desc, intent));
                self.tagged = true;
                tracing::info!("shader surface tagged (extended-linear)");
            }
            Status::Pending => {} // try again next frame
            Status::Failed(ref msg) => {
                tracing::error!("shader color description failed: {msg}; leaving untagged");
                self.tagged = true;
            }
        }
    }
}

impl Drop for ShaderSurface {
    fn drop(&mut self) {
        // DeviceState's Drop frees its GPU objects; destroy the wl objects.
        self.state = None;
        if let Some(cm) = &self.cm {
            cm.destroy();
        }
        if let Some(fb) = &self.feedback {
            fb.destroy();
        }
    }
}

/// Read a `dev_t` from a feedback `device` array (native byte order).
fn dev_from_bytes(bytes: &[u8]) -> Option<u64> {
    if bytes.len() < 8 {
        return None;
    }
    Some(u64::from_ne_bytes(bytes[..8].try_into().ok()?))
}

/// mmap and parse the feedback format table: 16-byte entries of
/// `{ u32 fourcc; u32 padding; u64 modifier }`.
fn parse_format_table(fd: OwnedFd, size: u32) -> Vec<(u32, u64)> {
    let len = size as usize;
    if len < 16 {
        return Vec::new();
    }
    // SAFETY: mmap a read-only private view of the compositor's table fd.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            fd.as_raw_fd(),
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        tracing::warn!("dmabuf feedback: mmap of format table failed");
        return Vec::new();
    }
    // SAFETY: ptr is valid for `len` bytes until munmap below.
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
    let out: Vec<(u32, u64)> = bytes
        .chunks_exact(16)
        .map(|e| {
            let fourcc = u32::from_ne_bytes(e[0..4].try_into().unwrap());
            let modifier = u64::from_ne_bytes(e[8..16].try_into().unwrap());
            (fourcc, modifier)
        })
        .collect();
    // SAFETY: unmap the view we just created.
    unsafe { libc::munmap(ptr, len) };
    out
}

/// Import a render target's exported dmabuf as a `wl_buffer`. All planes
/// share the single allocation FD at their offsets (the DCC metadata plane
/// included); the compositor dups the FD on `create_immed`.
fn import_dmabuf(
    dmabuf: &DmabufState,
    qh: &QueueHandle<App>,
    target: &RenderTarget,
    available: Arc<AtomicBool>,
) -> WlBuffer {
    let params = dmabuf.proxy.create_params(qh, ());
    let modifier = target.modifier;
    let mod_hi = (modifier >> 32) as u32;
    let mod_lo = (modifier & 0xFFFF_FFFF) as u32;
    for (plane, layout) in target.planes.iter().enumerate() {
        params.add(
            target.fd.as_fd(),
            plane as u32,
            layout.offset as u32,
            layout.stride as u32,
            mod_hi,
            mod_lo,
        );
    }
    let buffer = params.create_immed(
        target.width as i32,
        target.height as i32,
        RENDER_DRM_FOURCC,
        zwp_linux_buffer_params_v1::Flags::empty(),
        qh,
        available,
    );
    params.destroy();
    buffer
}

// ---- Wayland dispatch ----

impl Dispatch<ZwpLinuxDmabufFeedbackV1, FeedbackId> for App {
    fn event(
        state: &mut App,
        _: &ZwpLinuxDmabufFeedbackV1,
        event: zwp_linux_dmabuf_feedback_v1::Event,
        data: &FeedbackId,
        _: &wayland_client::Connection,
        qh: &QueueHandle<App>,
    ) {
        state.on_surface_feedback(qh, &data.0, event);
    }
}

// v4 dmabuf object: format/modifier events are only sent to v<4; nothing here.
wayland_client::delegate_noop!(App: ignore ZwpLinuxDmabufV1);
// create_immed is synchronous; created/failed fire only for async `create`.
wayland_client::delegate_noop!(App: ignore ZwpLinuxBufferParamsV1);

impl Dispatch<WlBuffer, Arc<AtomicBool>> for App {
    fn event(
        _: &mut App,
        _: &WlBuffer,
        event: wayland_client::protocol::wl_buffer::Event,
        available: &Arc<AtomicBool>,
        _: &wayland_client::Connection,
        _: &QueueHandle<App>,
    ) {
        // The compositor is done with this buffer; free it for reuse.
        if matches!(event, wayland_client::protocol::wl_buffer::Event::Release) {
            available.store(true, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::usage_scan_source;

    // The push block is positional, so an iMouse shader must declare `iTime` to
    // reach iMouse — that declaration must not read as a *use* (else the shader
    // would be wrongly flagged animated).
    #[test]
    fn declaration_is_not_a_use() {
        let src = r#"
            layout(push_constant) uniform Push {
                vec2 iResolution;
                float iTime;
                float _pad;
                vec4 iMouse;
            } pc;
            void main() { vec2 m = pc.iMouse.xy; }
        "#;
        let scan = usage_scan_source(src);
        assert!(
            !scan.contains("iTime"),
            "declaration-only iTime must not scan as a use"
        );
        assert!(
            scan.contains("iMouse"),
            "pc.iMouse access must scan as a use"
        );
    }

    #[test]
    fn line_comment_mention_is_not_a_use() {
        let src = "// animate using iTime\nvoid main() {}";
        assert!(!usage_scan_source(src).contains("iTime"));
    }

    #[test]
    fn block_comment_mention_is_not_a_use() {
        let src = "/* an iDate clock idea */ void main() { int f = pc.iFrame; }";
        let scan = usage_scan_source(src);
        assert!(
            !scan.contains("iDate"),
            "iDate only in a comment must not scan"
        );
        assert!(scan.contains("iFrame"), "pc.iFrame access must scan");
    }

    #[test]
    fn access_through_block_instance_scans() {
        let src = "layout(push_constant) uniform P { float iTime; } pc; \
                   void main() { float t = pc.iTime; }";
        assert!(usage_scan_source(src).contains("iTime"));
    }

    // Qualifier variants (`std430`, extra whitespace) must still be stripped, or
    // a declared-but-unused field leaks back into the scan.
    // Exercises the full texture-load path (relative-path join + decode + sRGB
    // quantize) on the real example asset, without a GPU.
    #[test]
    fn loads_example_noise_texture() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/shaders");
        let tex = crate::shadergraph::TextureSpec {
            name: "noise".into(),
            path: "../textures/rgba-noise.png".into(),
            srgb: false,
        };
        let data = super::load_texture(&base, &tex).expect("load example noise texture");
        assert_eq!((data.width, data.height), (256, 256));
        let crate::gpu::TexturePixels::Rgba8 { data: rgba, srgb } = &data.pixels else {
            panic!("expected 8-bit pixels");
        };
        assert_eq!(rgba.len(), 256 * 256 * 4);
        assert!(!srgb, "textures default to raw (UNORM) sampling");
    }

    #[test]
    fn push_constant_qualifier_variants_are_stripped() {
        for layout in [
            "layout(push_constant, std430)",
            "layout(std430, push_constant)",
            "layout( push_constant )",
        ] {
            let src =
                format!("{layout} uniform P {{ float iTime; vec4 iMouse; }} pc; void main() {{}}");
            let scan = usage_scan_source(&src);
            assert!(!scan.contains("iTime"), "{layout}: declaration leaked");
            assert!(!scan.contains("iMouse"), "{layout}: declaration leaked");
        }
    }
}
