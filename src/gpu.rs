//! Vulkan backend for shader wallpapers.
//!
//! Unlike a typical GL/Vulkan client this does **not** use a WSI swapchain
//! (`VK_KHR_wayland_surface`). The whole point of prism-bg is that the
//! client owns the buffer and tags it with a parametric color description;
//! a swapchain would cede buffer ownership and color-space negotiation to
//! the driver. Instead we render into our own fp16 `VkImage` allocated with
//! an explicit DRM format modifier ([`VK_EXT_image_drm_format_modifier`]),
//! export it as a dmabuf ([`VK_EXT_external_memory_dma_buf`]), and hand that
//! FD to the compositor via `zwp_linux_dmabuf_v1` — the same
//! produce-a-buffer-then-tag-the-surface model as the shm path, with the
//! GPU filling the buffer instead of the CPU. No GPU→CPU readback ever
//! occurs: the CPU only touches the FD, the protocol messages, and fences.
//!
//! This module is the device layer: a [`GpuPool`] owns one Vulkan instance
//! and lazily builds one [`Gpu`] (logical device) per physical GPU, keyed
//! by its DRM render node. A multi-GPU host (several cards, monitors split
//! across them) needs each output's wallpaper rendered on the GPU that
//! actually drives that output — otherwise the compositor pays a per-frame
//! cross-GPU detile+copy to import our buffer. Render targets and pipelines
//! build on a `Gpu`.

use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::os::fd::{FromRawFd, OwnedFd};

use anyhow::{bail, Context, Result};
use ash::vk;

/// Render format: 16-bit float per channel. Matches the `Abgr16161616f`
/// shm format the image path already uses, so the color-management story
/// is identical — the shader authors in extended-linear and the surface is
/// tagged accordingly. The DRM fourcc below is its memory-order twin.
pub const RENDER_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub const RENDER_DRM_FOURCC: u32 = drm_fourcc::DrmFourcc::Abgr16161616f as u32;

/// Device extensions every shader wallpaper needs. Selection rejects any
/// physical device missing one of these — without them the dmabuf-export
/// path is impossible, and we have no fallback worth the complexity.
const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[
    // Export device memory as an FD...
    ash::khr::external_memory_fd::NAME,
    // ...specifically a dmabuf the compositor can import.
    ash::ext::external_memory_dma_buf::NAME,
    // Allocate the image with a DRM tiling modifier the compositor groks,
    // so it can scan out / sample our buffer without an implicit detile.
    ash::ext::image_drm_format_modifier::NAME,
    // Release the image to the compositor's queue family (FOREIGN) at the
    // end of each frame so the layout transition is well-defined.
    ash::ext::queue_family_foreign::NAME,
    // Explicit GPU→compositor sync as a sync_file FD (used later; cheap to
    // require since every modern driver has it).
    ash::khr::external_semaphore_fd::NAME,
];

/// An eligible physical device discovered at pool construction: its DRM
/// render/primary node `dev_t`s (for matching a `wl_output`'s GPU via
/// dmabuf feedback), whether it's discrete, and its graphics queue family.
struct Candidate {
    pd: vk::PhysicalDevice,
    queue_family: u32,
    name: String,
    discrete: bool,
    /// `dev_t` of the render node, when `VK_EXT_physical_device_drm` reports
    /// one. This is what dmabuf feedback's `main_device`/`target_device`
    /// carries, so it's the key we match outputs against.
    render_dev: Option<u64>,
    primary_dev: Option<u64>,
}

/// Owns the Vulkan instance and builds one [`Gpu`] per physical device on
/// demand, matched to the DRM node an output reports. One instance, many
/// logical devices.
pub struct GpuPool {
    _entry: ash::Entry,
    instance: ash::Instance,
    candidates: Vec<Candidate>,
    /// Built logical devices, keyed by candidate index.
    cache: HashMap<usize, Gpu>,
}

impl GpuPool {
    pub fn new() -> Result<GpuPool> {
        // SAFETY: loads the system Vulkan loader; valid for the process.
        let entry = unsafe { ash::Entry::load().context("loading Vulkan loader (libvulkan)")? };

        let app_name = c"prism-bg";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(app_name)
            .api_version(vk::API_VERSION_1_2);

        // Optional validation layer for development (PRISM_BG_VK_VALIDATION=1).
        let mut layer_ptrs: Vec<*const c_char> = Vec::new();
        let validation = c"VK_LAYER_KHRONOS_validation";
        if std::env::var_os("PRISM_BG_VK_VALIDATION").is_some() {
            // SAFETY: enumerate_instance_layer_properties is always valid.
            let available =
                unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
            let present = available.iter().any(|l| {
                let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
                name == validation
            });
            if present {
                layer_ptrs.push(validation.as_ptr());
                tracing::info!("Vulkan validation layer enabled");
            } else {
                tracing::warn!("PRISM_BG_VK_VALIDATION set but validation layer not installed");
            }
        }

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_ptrs);
        // SAFETY: create_info is valid and outlives the call.
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .context("creating Vulkan instance")?;

        let candidates = Self::enumerate(&instance);
        if candidates.is_empty() {
            unsafe { instance.destroy_instance(None) };
            bail!(
                "no Vulkan device supports the dmabuf-export extensions required for \
                 shader wallpapers (need: {})",
                REQUIRED_DEVICE_EXTENSIONS
                    .iter()
                    .map(|e| e.to_string_lossy())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Ok(GpuPool {
            _entry: entry,
            instance,
            candidates,
            cache: HashMap::new(),
        })
    }

    /// Discover every eligible physical device and its DRM node `dev_t`s.
    fn enumerate(instance: &ash::Instance) -> Vec<Candidate> {
        // SAFETY: instance is valid.
        let devices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        let mut out = Vec::new();
        for pd in devices {
            let Some(queue_family) = graphics_queue_family(instance, pd) else {
                continue;
            };
            if !supports_required_extensions(instance, pd) {
                continue;
            }
            // DRM node dev_ts via VK_EXT_physical_device_drm (the struct is
            // ignored by drivers lacking the extension; fields stay zero).
            let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
            // SAFETY: pd is from this instance; props2 outlives the call.
            unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
            let name = unsafe { CStr::from_ptr(props2.properties.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let discrete = props2.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
            let mk = |major: i64, minor: i64| {
                libc::makedev(major as libc::c_uint, minor as libc::c_uint) as u64
            };
            let render_dev = (drm.has_render != 0).then(|| mk(drm.render_major, drm.render_minor));
            let primary_dev =
                (drm.has_primary != 0).then(|| mk(drm.primary_major, drm.primary_minor));
            out.push(Candidate {
                pd,
                queue_family,
                name,
                discrete,
                render_dev,
                primary_dev,
            });
        }
        out
    }

    /// Get (building if needed) the `Gpu` for the GPU identified by DRM
    /// node `dev`, as reported by dmabuf feedback. Falls back to a discrete
    /// device with a warning if no node matches (e.g. driver without
    /// `VK_EXT_physical_device_drm`).
    pub fn get_for_device(&mut self, dev: u64) -> Result<&Gpu> {
        let idx = self
            .candidates
            .iter()
            .position(|c| c.render_dev == Some(dev) || c.primary_dev == Some(dev));
        let idx = match idx {
            Some(i) => i,
            None => {
                let fallback = self.preferred_index();
                tracing::warn!(
                    device = format!("{:#x}", dev),
                    "no Vulkan device matches the output's DRM node; using {}",
                    self.candidates[fallback].name
                );
                fallback
            }
        };
        self.ensure(idx)
    }

    /// Get (building if needed) a sensible default device (discrete first).
    /// Test helper / fallback entry point.
    #[cfg(test)]
    pub fn any(&mut self) -> Result<&Gpu> {
        let idx = self.preferred_index();
        self.ensure(idx)
    }

    /// First discrete device, else the first eligible one.
    fn preferred_index(&self) -> usize {
        self.candidates.iter().position(|c| c.discrete).unwrap_or(0)
    }

    fn ensure(&mut self, idx: usize) -> Result<&Gpu> {
        if !self.cache.contains_key(&idx) {
            let gpu = Gpu::build(&self.instance, &self.candidates[idx])?;
            self.cache.insert(idx, gpu);
        }
        Ok(self.cache.get(&idx).expect("just inserted"))
    }
}

impl Drop for GpuPool {
    fn drop(&mut self) {
        // Destroy every logical device before the instance.
        self.cache.clear();
        // SAFETY: all devices are gone; instance created in new(), once.
        unsafe { self.instance.destroy_instance(None) };
    }
}

fn graphics_queue_family(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Option<u32> {
    // SAFETY: pd is from this instance.
    let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|i| i as u32)
}

fn supports_required_extensions(instance: &ash::Instance, pd: vk::PhysicalDevice) -> bool {
    // SAFETY: pd is from this instance.
    let available = match unsafe { instance.enumerate_device_extension_properties(pd) } {
        Ok(a) => a,
        Err(_) => return false,
    };
    REQUIRED_DEVICE_EXTENSIONS.iter().all(|req| {
        available.iter().any(|ext| {
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            name == *req
        })
    })
}

/// A logical Vulkan device on one physical GPU: a graphics queue and a
/// command pool. Built and owned by [`GpuPool`].
pub struct Gpu {
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family: u32,
    pub queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    /// Cached memory properties for [`Self::find_memory_type`].
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Loader for `VK_EXT_image_drm_format_modifier` device functions.
    pub drm_modifier_fn: ash::ext::image_drm_format_modifier::Device,
    /// Loader for `VK_KHR_external_memory_fd` device functions.
    pub external_memory_fd_fn: ash::khr::external_memory_fd::Device,
}

impl Gpu {
    fn build(instance: &ash::Instance, cand: &Candidate) -> Result<Gpu> {
        let physical_device = cand.pd;
        let queue_family = cand.queue_family;
        tracing::info!(device = %cand.name, queue_family, "Vulkan device selected");

        let ext_ptrs: Vec<*const c_char> = REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .map(|e| e.as_ptr())
            .collect();
        let priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities);
        let queue_infos = [queue_info];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&ext_ptrs);
        // SAFETY: all referenced slices outlive the call.
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .context("creating logical device")?;

        // SAFETY: queue_family/index 0 were requested above.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);
        // SAFETY: pool_info is valid.
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .context("creating command pool")?;

        // SAFETY: physical_device came from this instance.
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let drm_modifier_fn = ash::ext::image_drm_format_modifier::Device::new(instance, &device);
        let external_memory_fd_fn = ash::khr::external_memory_fd::Device::new(instance, &device);

        Ok(Gpu {
            instance: instance.clone(),
            device,
            physical_device,
            queue_family,
            queue,
            command_pool,
            memory_properties,
            drm_modifier_fn,
            external_memory_fd_fn,
        })
    }

    /// DRM format modifiers this device can render into (COLOR_ATTACHMENT)
    /// for [`RENDER_FORMAT`], each with its plane count. The Wayland side
    /// intersects these with the compositor's advertised set so the buffer
    /// is scanned out / sampled with no implicit detile.
    pub fn renderable_modifiers(&self) -> Vec<(u64, u32)> {
        // Two-call idiom: first fills the count, second fills the array.
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
        {
            let mut props2 = vk::FormatProperties2::default().push_next(&mut list);
            // SAFETY: physical_device is from self.instance.
            unsafe {
                self.instance.get_physical_device_format_properties2(
                    self.physical_device,
                    RENDER_FORMAT,
                    &mut props2,
                );
            }
        }
        let count = list.drm_format_modifier_count as usize;
        let mut storage = vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
        list.p_drm_format_modifier_properties = storage.as_mut_ptr();
        {
            let mut props2 = vk::FormatProperties2::default().push_next(&mut list);
            // SAFETY: storage outlives this call; count matches its length.
            unsafe {
                self.instance.get_physical_device_format_properties2(
                    self.physical_device,
                    RENDER_FORMAT,
                    &mut props2,
                );
            }
        }
        storage
            .iter()
            .take(list.drm_format_modifier_count as usize)
            .filter(|m| {
                m.drm_format_modifier_tiling_features
                    .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT)
            })
            .map(|m| (m.drm_format_modifier, m.drm_format_modifier_plane_count))
            .collect()
    }

    /// Find a memory type index satisfying `type_bits` (from a memory
    /// requirements query) with all of `flags`. Returns `None` if the
    /// device exposes no such type.
    pub fn find_memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        let mp = &self.memory_properties;
        (0..mp.memory_type_count).find(|&i| {
            let supported = type_bits & (1 << i) != 0;
            let has_flags = mp.memory_types[i as usize].property_flags.contains(flags);
            supported && has_flags
        })
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: handles created from this device, dropped once. wait_idle
        // ensures nothing is in flight. The instance is owned by GpuPool and
        // outlives every Gpu (the pool destroys devices before the instance).
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

/// One plane of an exported dmabuf: byte offset and row pitch within the
/// shared FD. Single-plane for the fp16 format, but the modifier could in
/// principle request more, so we carry a list.
#[derive(Debug, Clone, Copy)]
pub struct DmabufPlane {
    pub offset: u64,
    pub stride: u64,
}

/// An fp16 `VkImage` allocated with a DRM format modifier and exported as a
/// dmabuf. Rendered into directly (no intermediate, no resolve), then the
/// FD + plane layout are imported by the compositor as a `wl_buffer`. The
/// `VkImage`/memory stay alive here for the lifetime of the surface; the
/// compositor dups the FD on import.
pub struct RenderTarget {
    /// Cloned device handle for `Drop` (ash's idiom — cheap handle copy,
    /// destroyed exactly once here).
    device: ash::Device,
    pub image: vk::Image,
    pub view: vk::ImageView,
    memory: vk::DeviceMemory,
    pub width: u32,
    pub height: u32,
    /// The modifier the driver actually chose from the candidate list.
    pub modifier: u64,
    pub planes: Vec<DmabufPlane>,
    /// The exported dmabuf. Handed to `zwp_linux_dmabuf` at import; kept
    /// here until then.
    pub fd: OwnedFd,
}

impl RenderTarget {
    /// Allocate an exportable fp16 target of `width`×`height`. `candidates`
    /// is the modifier list the driver may choose from — pass the
    /// intersection of [`Gpu::renderable_modifiers`] and the compositor's
    /// advertised set, with each modifier's plane count for the chosen-one
    /// layout query.
    pub fn new(
        gpu: &Gpu,
        width: u32,
        height: u32,
        candidates: &[(u64, u32)],
    ) -> Result<RenderTarget> {
        if candidates.is_empty() {
            bail!("no DRM format modifier is both GPU-renderable and compositor-importable");
        }
        let modifier_list: Vec<u64> = candidates.iter().map(|(m, _)| *m).collect();
        let device = &gpu.device;

        let mut ext_mem = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut mod_list = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
            .drm_format_modifiers(&modifier_list);
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(RENDER_FORMAT)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut ext_mem)
            .push_next(&mut mod_list);
        // SAFETY: image_info and its chained structs outlive the call.
        let image =
            unsafe { device.create_image(&image_info, None) }.context("creating dmabuf image")?;

        // Allocate dedicated, exportable device-local memory and bind it.
        // Dedicated allocation is the safe choice for exported images on
        // RADV (and required by some drivers).
        // SAFETY: image was just created on this device.
        let reqs = unsafe { device.get_image_memory_requirements(image) };
        let mem_type = gpu
            .find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .context("no device-local memory type for render target")?;
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(mem_type)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        // SAFETY: alloc_info outlives the call; freed in Drop.
        let memory = match unsafe { device.allocate_memory(&alloc_info, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_image(image, None) };
                return Err(e).context("allocating exportable image memory");
            }
        };
        // SAFETY: image and memory are this device's; offset 0, dedicated.
        if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
            unsafe {
                device.free_memory(memory, None);
                device.destroy_image(image, None);
            }
            return Err(e).context("binding image memory");
        }

        // Build the rest; on failure, tear down image+memory.
        match Self::finish(gpu, image, memory, width, height, candidates) {
            Ok(rt) => Ok(rt),
            Err(e) => {
                // SAFETY: nothing else references these yet.
                unsafe {
                    device.free_memory(memory, None);
                    device.destroy_image(image, None);
                }
                Err(e)
            }
        }
    }

    fn finish(
        gpu: &Gpu,
        image: vk::Image,
        memory: vk::DeviceMemory,
        width: u32,
        height: u32,
        candidates: &[(u64, u32)],
    ) -> Result<RenderTarget> {
        let device = &gpu.device;

        // Which modifier did the driver pick?
        let mut mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
        // SAFETY: image is a DRM-modifier image on this device.
        unsafe {
            gpu.drm_modifier_fn
                .get_image_drm_format_modifier_properties(image, &mut mod_props)
        }
        .context("querying chosen DRM format modifier")?;
        let modifier = mod_props.drm_format_modifier;
        let plane_count = candidates
            .iter()
            .find(|(m, _)| *m == modifier)
            .map(|(_, n)| *n)
            .context("driver chose a modifier outside the candidate list (bug)")?;

        // Per-plane offset/stride within the shared allocation.
        const PLANE_ASPECTS: [vk::ImageAspectFlags; 4] = [
            vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_2_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_3_EXT,
        ];
        let mut planes = Vec::with_capacity(plane_count as usize);
        for &aspect in PLANE_ASPECTS.iter().take(plane_count as usize) {
            let sub = vk::ImageSubresource::default()
                .aspect_mask(aspect)
                .mip_level(0)
                .array_layer(0);
            // SAFETY: image is a DRM-modifier image; aspect is a memory plane.
            let layout = unsafe { device.get_image_subresource_layout(image, sub) };
            planes.push(DmabufPlane {
                offset: layout.offset,
                stride: layout.row_pitch,
            });
        }

        // Export the dmabuf FD (one FD; planes reference it at offsets).
        let get_fd = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        // SAFETY: memory is exportable (allocated with ExportMemoryAllocateInfo).
        let raw_fd = unsafe { gpu.external_memory_fd_fn.get_memory_fd(&get_fd) }
            .context("exporting dmabuf FD")?;
        // SAFETY: get_memory_fd transfers ownership of a fresh FD to us.
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(RENDER_FORMAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        // SAFETY: view_info outlives the call; destroyed in Drop.
        let view = unsafe { device.create_image_view(&view_info, None) }
            .context("creating render-target image view")?;

        Ok(RenderTarget {
            device: device.clone(),
            image,
            view,
            memory,
            width,
            height,
            modifier,
            planes,
            fd,
        })
    }
}

impl Drop for RenderTarget {
    fn drop(&mut self) {
        // SAFETY: created from self.device, destroyed once. The owning Gpu
        // (and thus the device) outlives every RenderTarget by construction.
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// Shadertoy-ish uniforms handed to the fragment shader as push constants
/// (no descriptor sets needed for the common case).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderUniforms {
    /// Output size in pixels (`iResolution.xy`).
    pub resolution: [f32; 2],
    /// Seconds since the wallpaper started (`iTime`).
    pub time: f32,
    /// Padding to 16 bytes (std430 / push-constant alignment).
    pub _pad: f32,
}

/// Fullscreen-triangle vertex shader: three vertices covering the viewport,
/// emitting pixel-space `fragCoord` with a bottom-left origin (Shadertoy
/// convention; Vulkan's framebuffer y is flipped back in the shader).
const VERTEX_GLSL: &str = r#"#version 450
layout(location = 0) out vec2 fragCoord;
layout(push_constant) uniform Push { vec2 iResolution; float iTime; float _pad; } pc;
void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    // Flip y so fragCoord.y grows upward like Shadertoy.
    fragCoord = vec2(uv.x, 1.0 - uv.y) * pc.iResolution;
}
"#;

/// Animated gradient used as the render test subject (kept in sync with
/// `examples/shaders/gradient.frag`). Outputs extended-linear values; the
/// surface is tagged linear / sRGB-primaries.
#[cfg(test)]
pub const DEFAULT_FRAGMENT_GLSL: &str = r#"#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 iResolution; float iTime; float _pad; } pc;
void main() {
    vec2 uv = fragCoord / pc.iResolution;
    vec3 col = 0.5 + 0.5 * cos(pc.iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
    outColor = vec4(col, 1.0);
}
"#;

/// Compile a fragment shader to SPIR-V and discard it — a device-independent
/// validity check so a bad `--shader` fails at startup with the GLSL
/// compiler's diagnostic, before any GPU work or output mapping.
pub fn validate_fragment(source: &str) -> Result<()> {
    compile_glsl(source, shaderc::ShaderKind::Fragment, "wallpaper.frag")?;
    Ok(())
}

/// Compile GLSL to SPIR-V at load time. Returns the validator's error
/// messages on failure so a bad user shader gets a useful diagnostic.
fn compile_glsl(source: &str, kind: shaderc::ShaderKind, name: &str) -> Result<Vec<u32>> {
    let compiler = shaderc::Compiler::new().context("initializing shaderc")?;
    let artifact = compiler
        .compile_into_spirv(source, kind, name, "main", None)
        .with_context(|| format!("compiling {name}"))?;
    Ok(artifact.as_binary().to_vec())
}

/// The render pass + graphics pipeline for one fragment shader. Renders a
/// fullscreen triangle into an fp16 [`RenderTarget`]; viewport/scissor are
/// dynamic so a resize reuses the pipeline.
pub struct ShaderRenderer {
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl ShaderRenderer {
    /// Build the pipeline for `fragment_glsl` (a fragment shader providing
    /// `main` and the `Push` uniform block — see [`DEFAULT_FRAGMENT_GLSL`]).
    pub fn new(gpu: &Gpu, fragment_glsl: &str) -> Result<ShaderRenderer> {
        let device = gpu.device.clone();
        let vert_spv = compile_glsl(VERTEX_GLSL, shaderc::ShaderKind::Vertex, "fullscreen.vert")?;
        let frag_spv = compile_glsl(
            fragment_glsl,
            shaderc::ShaderKind::Fragment,
            "wallpaper.frag",
        )?;

        let render_pass = Self::create_render_pass(&device)?;
        match Self::build(gpu, device.clone(), render_pass, &vert_spv, &frag_spv) {
            Ok(r) => Ok(r),
            Err(e) => {
                // SAFETY: render_pass created just above, nothing else owns it.
                unsafe { device.destroy_render_pass(render_pass, None) };
                Err(e)
            }
        }
    }

    fn create_render_pass(device: &ash::Device) -> Result<vk::RenderPass> {
        // initialLayout UNDEFINED every frame: we redraw the whole surface,
        // so prior contents are discarded — no acquire-from-FOREIGN needed.
        // finalLayout GENERAL: the explicit queue-family release barrier
        // afterward (→ FOREIGN) hands the buffer to the compositor.
        let attachment = vk::AttachmentDescription::default()
            .format(RENDER_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::GENERAL);
        let attachments = [attachment];
        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_refs = [color_ref];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs);
        let subpasses = [subpass];
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        let dependencies = [dependency];
        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        // SAFETY: info outlives the call.
        unsafe { device.create_render_pass(&info, None) }.context("creating render pass")
    }

    fn build(
        gpu: &Gpu,
        device: ash::Device,
        render_pass: vk::RenderPass,
        vert_spv: &[u32],
        frag_spv: &[u32],
    ) -> Result<ShaderRenderer> {
        // SAFETY: SPIR-V slices are valid for the duration of module creation.
        let vert = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(vert_spv), None)
        }
        .context("creating vertex shader module")?;
        let frag = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(frag_spv), None)
        };
        let frag = match frag {
            Ok(f) => f,
            Err(e) => {
                unsafe { device.destroy_shader_module(vert, None) };
                return Err(e).context("creating fragment shader module");
            }
        };

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<ShaderUniforms>() as u32);
        let push_ranges = [push_range];
        let layout_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges);
        // SAFETY: layout_info outlives the call.
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .context("creating pipeline layout")?;

        let pipeline_result = Self::create_pipeline(&device, render_pass, layout, vert, frag);
        // Shader modules are consumed by pipeline creation; destroy regardless.
        // SAFETY: vert/frag are this device's, no longer referenced.
        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
        }
        let pipeline = match pipeline_result {
            Ok(p) => p,
            Err(e) => {
                unsafe { device.destroy_pipeline_layout(layout, None) };
                return Err(e);
            }
        };

        // One reusable command buffer + fence (RESET_COMMAND_BUFFER pool).
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(gpu.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: alloc references gpu.command_pool, valid for this device.
        let command_buffer = unsafe { device.allocate_command_buffers(&alloc) }
            .context("allocating command buffer")?[0];
        // Pre-signaled: render() waits at the top of each frame, so the
        // first frame must find it already signaled or it blocks forever.
        let fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .context("creating fence")?;

        Ok(ShaderRenderer {
            device,
            queue: gpu.queue,
            queue_family: gpu.queue_family,
            render_pass,
            layout,
            pipeline,
            command_buffer,
            fence,
        })
    }

    fn create_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        layout: vk::PipelineLayout,
        vert: vk::ShaderModule,
        frag: vk::ShaderModule,
    ) -> Result<vk::Pipeline> {
        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(entry),
        ];
        // No vertex buffers — the VS generates positions from gl_VertexIndex.
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let blend_attachments = [blend_attachment];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        // SAFETY: all referenced state outlives the call.
        let pipelines =
            unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
                .map_err(|(_, e)| e)
                .context("creating graphics pipeline")?;
        Ok(pipelines[0])
    }

    /// Create a framebuffer binding `target`'s view to this render pass.
    /// The caller owns it and destroys it via the device.
    pub fn create_framebuffer(&self, target: &RenderTarget) -> Result<vk::Framebuffer> {
        let attachments = [target.view];
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(self.render_pass)
            .attachments(&attachments)
            .width(target.width)
            .height(target.height)
            .layers(1);
        // SAFETY: info outlives the call.
        unsafe { self.device.create_framebuffer(&info, None) }.context("creating framebuffer")
    }

    /// Render one frame into `target` via `framebuffer`, then block until
    /// the GPU is done (so the dmabuf is safe to commit). A CPU fence wait
    /// is simple and correct for the prototype; explicit GPU→compositor
    /// sync (`wp_linux_drm_syncobj`) is the later optimization. No readback.
    pub fn render(
        &self,
        target: &RenderTarget,
        framebuffer: vk::Framebuffer,
        uniforms: &ShaderUniforms,
    ) -> Result<()> {
        let device = &self.device;
        let cb = self.command_buffer;
        // SAFETY: all handles are this renderer's; the previous submission
        // (if any) is awaited via the fence before reuse below.
        unsafe {
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .context("waiting on prior frame fence")?;
            device
                .reset_fences(&[self.fence])
                .context("resetting fence")?;
            device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .context("resetting command buffer")?;

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(cb, &begin)
                .context("begin cb")?;

            let clear = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            };
            let clears = [clear];
            let area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: target.width,
                    height: target.height,
                },
            };
            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(framebuffer)
                .render_area(area)
                .clear_values(&clears);
            device.cmd_begin_render_pass(cb, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: target.width as f32,
                height: target.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[area]);
            device.cmd_push_constants(
                cb,
                self.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(uniforms),
            );
            device.cmd_draw(cb, 3, 1, 0, 0);
            device.cmd_end_render_pass(cb);

            // Release ownership to the compositor's (foreign) queue. For the
            // DCC modifier this also lets the driver flush compression
            // metadata so the compositor reads correct pixels.
            let barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(self.queue_family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .image(target.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            device.end_command_buffer(cb).context("end cb")?;

            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            device
                .queue_submit(self.queue, &[submit], self.fence)
                .context("submitting render")?;
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .context("waiting on render fence")?;
        }
        Ok(())
    }
}

impl Drop for ShaderRenderer {
    fn drop(&mut self) {
        // SAFETY: all handles are this renderer's; nothing is in flight after
        // the fence wait in render(). The command buffer is freed with the pool.
        unsafe {
            let _ = self.device.wait_for_fences(&[self.fence], true, u64::MAX);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brings up Vulkan on the local machine. Ignored by default so CI
    /// (no GPU) is unaffected; run with `cargo test --ignored gpu_init`.
    #[test]
    #[ignore]
    fn gpu_init() {
        let mut pool = GpuPool::new().expect("Vulkan bring-up failed");
        let gpu = pool.any().expect("no usable device");
        // A graphics queue family must exist and memory types be queryable.
        assert!(gpu
            .find_memory_type(u32::MAX, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .is_some());
    }

    /// Allocate an fp16 dmabuf target and export it. Verifies modifier
    /// negotiation, plane layout, and FD export end to end on the local GPU.
    #[test]
    #[ignore]
    fn render_target_export() {
        use std::os::fd::AsRawFd;
        let mut pool = GpuPool::new().expect("Vulkan bring-up failed");
        let gpu = pool.any().expect("no usable device");
        let mods = gpu.renderable_modifiers();
        assert!(!mods.is_empty(), "no renderable fp16 modifiers");
        let rt = RenderTarget::new(gpu, 256, 128, &mods).expect("render target");
        assert!(rt.fd.as_raw_fd() >= 0);
        assert!(!rt.planes.is_empty());
        // Plane 0 must hold at least one fp16 RGBA row (8 bytes/px); tiled
        // modifiers can pad beyond that, never under.
        assert!(rt.planes[0].stride >= 256 * 8);
        eprintln!(
            "modifier={:#x} planes={} stride0={}",
            rt.modifier,
            rt.planes.len(),
            rt.planes[0].stride
        );
    }

    /// Compile the default shader, build the pipeline, and render one frame
    /// into a dmabuf target. Run with PRISM_BG_VK_VALIDATION=1 to surface
    /// any validation errors. Pixel correctness is verified on-screen later.
    #[test]
    #[ignore]
    fn render_frame() {
        let mut pool = GpuPool::new().expect("Vulkan bring-up failed");
        let gpu = pool.any().expect("no usable device");
        let mods = gpu.renderable_modifiers();
        let rt = RenderTarget::new(gpu, 320, 200, &mods).expect("render target");
        let renderer = ShaderRenderer::new(gpu, DEFAULT_FRAGMENT_GLSL).expect("renderer");
        let fb = renderer.create_framebuffer(&rt).expect("framebuffer");
        let uniforms = ShaderUniforms {
            resolution: [rt.width as f32, rt.height as f32],
            time: 1.25,
            _pad: 0.0,
        };
        renderer.render(&rt, fb, &uniforms).expect("render frame");
        // SAFETY: fb came from this renderer; the render fence was awaited
        // inside render(), so nothing references it.
        unsafe {
            let _ = gpu.device.device_wait_idle();
            gpu.device.destroy_framebuffer(fb, None);
        }
    }
}
