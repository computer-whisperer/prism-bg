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
//! surface is occluded); a shader that doesn't use `iTime` renders once.

use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use ash::vk;
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
use crate::gpu::{Gpu, RenderTarget, ShaderRenderer, ShaderUniforms, RENDER_DRM_FOURCC};

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

pub struct ShaderSurface {
    /// GLSL fragment source; compiled per GPU into a `DeviceState`.
    source: String,
    /// Whether the shader samples `iTime` (drives frame-callback animation).
    animated: bool,
    /// iTime origin, set on the first rendered frame.
    started: Option<Instant>,
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
}

impl ShaderSurface {
    /// Prepare a shader surface from `fragment_glsl`. Compilation and target
    /// allocation are deferred until feedback resolves the output's GPU.
    pub fn new(
        qh: &QueueHandle<App>,
        fragment_glsl: &str,
        color: Option<&ColorState>,
        fps: Option<u32>,
    ) -> Result<ShaderSurface> {
        // Validate the shader compiles up front (device-independent), so a
        // bad shader fails at startup rather than silently on the GPU.
        crate::gpu::validate_fragment(fragment_glsl)?;
        let animated = fragment_glsl.contains("iTime");
        if fps.is_some() && !animated {
            tracing::warn!("--fps ignored: shader is static (no iTime), renders a single frame");
        }
        // Cap as a minimum inter-render interval; only meaningful when animated.
        let min_interval = fps
            .filter(|_| animated)
            .map(|n| Duration::from_secs_f64(1.0 / n as f64));
        let description = match color {
            Some(c) => match c.create_description(qh, &SHADER_ENCODING) {
                Ok(d) => Some(d),
                Err(e) => {
                    tracing::warn!(
                        "shader color description unavailable: {e:#}; attaching untagged"
                    );
                    None
                }
            },
            None => None,
        };
        Ok(ShaderSurface {
            source: fragment_glsl.to_string(),
            animated,
            started: None,
            min_interval,
            last_render: None,
            description,
            cm: None,
            tagged: false,
            feedback: None,
            accum: FeedbackAccum::default(),
            resolved: None,
            state: None,
        })
    }

    pub fn animated(&self) -> bool {
        self.animated
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
                renderer: ShaderRenderer::new(gpu, &self.source, RING)?,
                ring: Vec::new(),
                size: (0, 0),
            });
        }
        // (Re)build the ring if the size changed.
        let need_ring = {
            let st = self.state.as_ref().unwrap();
            st.ring.is_empty() || st.size != size
        };
        if need_ring {
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
        let uniforms = ShaderUniforms {
            resolution: [size.0 as f32, size.1 as f32],
            time,
            _pad: 0.0,
        };
        let rt = &st.ring[idx];
        st.renderer
            .render(idx, &rt.target, rt.framebuffer, &uniforms)
            .context("rendering shader frame")?;
        rt.available.store(false, Ordering::Release);
        surface.attach(Some(&rt.buffer), 0, 0);
        self.last_render = Some(Instant::now());
        viewport.set_source(-1.0, -1.0, -1.0, -1.0);
        viewport.set_destination(logical.0 as i32, logical.1 as i32);
        surface.damage_buffer(0, 0, i32::MAX, i32::MAX);
        if self.animated {
            // Request the next callback before committing; delivered when
            // ready (and not while occluded).
            surface.frame(qh, surface.clone());
        }
        surface.commit();
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
        state.on_shader_feedback(qh, &data.0, event);
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
