//! The Wayland application: one background layer surface per matched
//! output, each a solid-color parent (1×1 buffer, viewport-stretched) with
//! an optional image subsurface on top. The compositor does all scaling
//! (viewport) and all color conversion (each image surface is tagged with
//! its source description via `wp_color_management_v1`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
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
use wayland_protocols::wp::color_management::v1::client::wp_color_management_surface_v1::WpColorManagementSurfaceV1;
use wayland_protocols::wp::viewporter::client::{
    wp_viewport::WpViewport, wp_viewporter::WpViewporter,
};

use crate::cli::{Args, Color, Intent, Mode, OutputSpec};
use crate::colormgmt::{ColorState, DescriptionHandle};
use crate::decode::DecodedImage;
use crate::surfaces::{place, upload, upload_tiled};

/// A decoded image plus its (shared) compositor-side description.
pub struct LoadedImage {
    pub image: Arc<DecodedImage>,
    pub description: Option<DescriptionHandle>,
}

struct ImagePart {
    _subsurface: WlSubsurface,
    surface: WlSurface,
    viewport: WpViewport,
    /// Keeps the color-management surface wrapper (and with it the
    /// description binding) alive.
    _cm: Option<WpColorManagementSurfaceV1>,
    image: Arc<DecodedImage>,
    mode: Mode,
}

struct Wallpaper {
    output: wl_output::WlOutput,
    name: String,
    layer: LayerSurface,
    viewport: WpViewport,
    color: Color,
    image_part: Option<ImagePart>,
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
    pub color: Option<ColorState>,
    pub intent: Intent,
    pub specs: Vec<OutputSpec>,
    pub images: HashMap<PathBuf, LoadedImage>,
    /// False until main() has created the image descriptions; gates
    /// wallpaper creation for outputs announced during the setup
    /// roundtrips (main sweeps them once descriptions settle).
    pub bootstrapped: bool,
    wallpapers: Vec<Wallpaper>,
}

pub type SimpleViewporter = smithay_client_toolkit::registry::SimpleGlobal<WpViewporter, 1>;

impl App {
    pub fn new(
        globals: &GlobalList,
        qh: &QueueHandle<App>,
        args: &Args,
        images: HashMap<PathBuf, LoadedImage>,
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
        let viewporter = SimpleViewporter::bind(globals, qh)
            .context("wp_viewporter not available")?;
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
            color,
            intent: args.intent,
            specs: args.specs.clone(),
            images,
            bootstrapped: false,
            wallpapers: Vec::new(),
        })
    }

    /// All image descriptions created and no longer pending?
    pub fn descriptions_settled(&self) -> bool {
        self.images.values().all(|li| {
            li.description
                .as_ref()
                .is_none_or(|d| d.status() != crate::colormgmt::Status::Pending)
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

        let mode = spec.effective_mode();
        let image_part = match (&spec.image, mode) {
            (Some(path), m) if m != Mode::SolidColor => {
                let loaded = &self.images[path];
                let (subsurface, child) = self
                    .subcompositor
                    .create_subsurface(layer.wl_surface().clone(), qh);
                let viewport = self.get_viewport(qh, &child);
                let cm = match (&self.color, &loaded.description) {
                    (Some(color), Some(desc)) => {
                        Some(color.tag_surface(qh, &child, desc, self.intent))
                    }
                    _ => None,
                };
                if !loaded.image.has_alpha {
                    if let Ok(region) = Region::new(&self.compositor) {
                        region.add(0, 0, i32::MAX, i32::MAX);
                        child.set_opaque_region(Some(region.wl_region()));
                    }
                }
                Some(ImagePart {
                    _subsurface: subsurface,
                    surface: child,
                    viewport,
                    _cm: cm,
                    image: loaded.image.clone(),
                    mode,
                })
            }
            _ => None,
        };

        // The parent is always opaque (solid color under an image, or the
        // solid color itself).
        if let Ok(region) = Region::new(&self.compositor) {
            region.add(0, 0, i32::MAX, i32::MAX);
            layer.wl_surface().set_opaque_region(Some(region.wl_region()));
        }

        // Bare commit maps the layer surface; the configure callback draws.
        layer.commit();

        tracing::info!(output = name, ?mode, "wallpaper surface created");
        self.wallpapers.push(Wallpaper {
            output,
            name,
            layer,
            viewport,
            color: spec.color.unwrap_or(Color { r: 0.0, g: 0.0, b: 0.0 }),
            image_part,
            size: (0, 0),
            scale,
            color_buffer: None,
            image_buffer: None,
        });
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
                Some((tw, th)) => upload_tiled(&mut self.pool, &part.image, tw, th)?,
                None => upload(&mut self.pool, &part.image)?,
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
            part._subsurface
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
        wp.layer.wl_surface().damage_buffer(0, 0, i32::MAX, i32::MAX);
        wp.layer.commit();
        wp.color_buffer = Some(buffer);
        Ok(())
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
        // Static content; we never request frame callbacks.
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
        if self.bootstrapped {
            self.add_output(qh, output);
        }
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
