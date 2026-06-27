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
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

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
    // Export the render-completion semaphore as a sync_file FD, attached to
    // the dmabuf for implicit GPU→compositor sync (see ShaderRenderer::render).
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
    /// Loader for `VK_KHR_external_semaphore_fd` device functions (exports the
    /// render-completion semaphore as a sync_file for implicit dmabuf sync).
    pub external_semaphore_fd_fn: ash::khr::external_semaphore_fd::Device,
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
        let external_semaphore_fd_fn =
            ash::khr::external_semaphore_fd::Device::new(instance, &device);

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
            external_semaphore_fd_fn,
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

/// Create one internal F16 feedback texture (optimal tiling, color-attachment +
/// sampled, device-local) and its view. Unlike [`RenderTarget`] it isn't a
/// dmabuf — it's sampled and rendered entirely within our queue. The caller
/// owns the returned handles.
fn create_feedback_image(gpu: &Gpu, device: &ash::Device, w: u32, h: u32) -> Result<FeedbackImage> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(RENDER_FORMAT)
        .extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: image_info outlives the call.
    let image = unsafe { device.create_image(&image_info, None) }
        .context("creating feedback image")?;

    // SAFETY: image is this device's.
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let Some(mem_type) =
        gpu.find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
    else {
        unsafe { device.destroy_image(image, None) };
        anyhow::bail!("no device-local memory for feedback image");
    };
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    // SAFETY: alloc outlives the call; image freed on error.
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(e) => {
            unsafe { device.destroy_image(image, None) };
            return Err(e).context("allocating feedback image memory");
        }
    };
    // SAFETY: image and memory are this device's, bound once.
    if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_image(image, None);
        }
        return Err(e).context("binding feedback image memory");
    }

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
    // SAFETY: view_info outlives the call; freed via FeedbackImage::destroy.
    let view = match unsafe { device.create_image_view(&view_info, None) } {
        Ok(v) => v,
        Err(e) => {
            unsafe {
                device.free_memory(memory, None);
                device.destroy_image(image, None);
            }
            return Err(e).context("creating feedback image view");
        }
    };
    Ok(FeedbackImage {
        image,
        view,
        memory,
    })
}

/// Clear both feedback textures to black and transition them to
/// `SHADER_READ_ONLY_OPTIMAL`, so the first frame samples a clean previous
/// frame. A one-shot command buffer, awaited before returning.
fn clear_feedback_textures(
    gpu: &Gpu,
    device: &ash::Device,
    textures: &[FeedbackImage; 2],
) -> Result<()> {
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(gpu.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: alloc references gpu.command_pool.
    let cb = unsafe { device.allocate_command_buffers(&alloc) }
        .context("allocating feedback clear command buffer")?[0];
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let black = vk::ClearColorValue {
        float32: [0.0, 0.0, 0.0, 1.0],
    };
    // SAFETY: cb is valid; all images/barriers are this device's; the queue
    // submit + wait below drains the work before the cb is freed.
    let result = unsafe {
        device
            .begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .context("begin feedback clear cb")?;
        for img in textures {
            let to_dst = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(img.image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
            device.cmd_clear_color_image(
                cb,
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &black,
                &[range],
            );
            let to_read = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(img.image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
        }
        device.end_command_buffer(cb).context("end feedback clear cb")?;
        let cbs = [cb];
        let submit = vk::SubmitInfo::default().command_buffers(&cbs);
        device
            .queue_submit(gpu.queue, &[submit], vk::Fence::null())
            .context("submitting feedback clear")?;
        device
            .queue_wait_idle(gpu.queue)
            .context("waiting on feedback clear")
    };
    // SAFETY: the submission has drained (queue_wait_idle), so the cb is free.
    unsafe { device.free_command_buffers(gpu.command_pool, &[cb]) };
    result
}

/// Shadertoy-ish uniforms handed to the fragment shader as push constants
/// (no descriptor sets needed for the common case).
///
/// The first 16 bytes are the classic Shadertoy subset (`iResolution`,
/// `iTime`); a shader that declares only those keeps working. The trailing
/// fields describe this output's place in the multi-monitor *cluster* so a
/// shader can tile continuously across the whole workspace — see the `Push`
/// block documented in the module. They are y-up logical pixels with the
/// origin at the cluster's bottom-left, matching the y-up `fragCoord` the
/// vertex stage emits. Layout (std430): every field is `vec2`/`float` so the
/// natural `repr(C)` order matches the GLSL block with no implicit padding.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderUniforms {
    /// This output's size in device pixels (`iResolution.xy`).
    pub resolution: [f32; 2],
    /// Seconds since the wallpaper started (`iTime`).
    pub time: f32,
    /// Padding to a vec2 boundary (std430 / push-constant alignment).
    pub _pad: f32,
    /// This output's bottom-left corner in cluster space, logical px
    /// (`iOutputOffset`). `(0,0)` for a lone output.
    pub output_offset: [f32; 2],
    /// This output's logical size (`iOutputSize`); maps a `0..1` uv into
    /// cluster space: `g = iOutputOffset + uv * iOutputSize`.
    pub output_size: [f32; 2],
    /// The whole cluster's logical size (`iGlobalResolution`); normalize a
    /// cluster coord to `0..1` across the workspace with `g / iGlobalResolution`.
    pub global_resolution: [f32; 2],
}

/// Number of spectrum bins handed to audio-reactive shaders (log-spaced,
/// low→high). Packed four per `vec4` for std140, so keep this a multiple of 4.
pub const AUDIO_BINS: usize = 32;

/// Audio-reactivity uniforms, uploaded each frame to a fragment-stage UBO
/// (set 0, binding 0) — separate from the push constants because the bin
/// array alone would blow the 128-byte push-constant budget. Zeroed when no
/// audio is captured, so a shader reads silence rather than garbage. std140
/// layout: `bins` is a `vec4[AUDIO_BINS/4]` (16-byte stride), then four
/// trailing floats; the `repr(C)` order matches with no implicit padding.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AudioUniforms {
    /// `AUDIO_BINS` magnitudes in `0..1`, low→high, packed 4 per `vec4`
    /// (`iAudioBins[i/4][i%4]`).
    pub bins: [[f32; 4]; AUDIO_BINS / 4],
    /// Overall loudness `0..1` (`iAudioLevel`).
    pub level: f32,
    /// Low/mid/high band energy `0..1` (`iAudioBass`/`iAudioMid`/`iAudioTreble`),
    /// for cheap reactive effects that don't want the full spectrum.
    pub bass: f32,
    pub mid: f32,
    pub treble: f32,
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

/// Blit vertex shader for the feedback present pass: fullscreen triangle that
/// emits a plain `0..1` uv (no y-flip — the feedback texture is copied to the
/// dmabuf 1:1, preserving orientation).
const BLIT_VERTEX_GLSL: &str = r#"#version 450
layout(location = 0) out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    uv = p;
}
"#;

/// Blit fragment shader: sample the feedback buffer and present it verbatim
/// (it's already in the shader's extended-linear output space).
const BLIT_FRAGMENT_GLSL: &str = r#"#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D feedbackTex;
void main() { outColor = texture(feedbackTex, uv); }
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

/// Per-ring-slot command resources. One set per [`RenderTarget`] in the ring
/// so frames pipeline: while the compositor reads slot N's buffer, we can
/// already record and submit slot N+1 without waiting on N's GPU work.
struct Frame {
    command_buffer: vk::CommandBuffer,
    /// Signaled when this slot's submission completes. Awaited (cheaply — the
    /// ring's `wl_buffer.release` recycle already implies completion) before
    /// the command buffer is reset for reuse.
    fence: vk::Fence,
    /// Signaled by the same submission; exported as a sync_file and attached
    /// to the dmabuf as its implicit write fence. Unused in CPU-wait mode.
    semaphore: vk::Semaphore,
    /// Per-slot spectrum UBO (set 0, binding 0), host-visible and persistently
    /// mapped. One per slot so writing the next frame's audio never races the
    /// in-flight read of the previous one (the fence wait at the top of
    /// `render` gates the overwrite). `mapped` points into `ubo_memory`.
    ubo: vk::Buffer,
    ubo_memory: vk::DeviceMemory,
    ubo_mapped: *mut u8,
    descriptor_set: vk::DescriptorSet,
}

/// One internal F16 feedback texture (not a dmabuf — it never leaves our
/// queue). Two of these ping-pong so a feedback shader can sample the previous
/// frame while writing the next.
struct FeedbackImage {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
}

impl FeedbackImage {
    // SAFETY: handles are the owner's device's; called once in teardown.
    unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_image_view(self.view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}

/// Everything a feedback (`iPrevFrame`) shader needs beyond the base renderer:
/// a ping-pong pair of internal textures, the off-screen render pass that
/// writes them, and the built-in blit pipeline that presents the latest into
/// the dmabuf. The user shader's pipeline itself lives in [`ShaderRenderer`]
/// (`pipeline`/`layout`), built against [`Self::render_pass`] with the extra
/// sampler set; this holds the rest.
struct Feedback {
    /// Linear sampler for `iPrevFrame` and the blit.
    sampler: vk::Sampler,
    /// Off-screen pass writing a feedback texture (F16, → SHADER_READ).
    render_pass: vk::RenderPass,
    /// Layout of the single combined-image-sampler set, used as set 1 of the
    /// user pipeline (`iPrevFrame`) and set 0 of the blit.
    sampler_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    /// `tex_sets[i]` samples `textures[i]`; re-pointed when textures resize.
    tex_sets: [vk::DescriptorSet; 2],
    blit_layout: vk::PipelineLayout,
    blit_pipeline: vk::Pipeline,
    /// The ping-pong textures and their framebuffers; rebuilt on resize.
    textures: Option<[FeedbackImage; 2]>,
    framebuffers: [vk::Framebuffer; 2],
    size: (u32, u32),
    /// Which texture is the current write target this frame; toggles each
    /// render. `Cell` so [`ShaderRenderer::render`] can stay `&self`.
    parity: std::cell::Cell<usize>,
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
    /// Spectrum-UBO descriptor layout (set 0, binding 0) and the pool the
    /// per-slot sets are allocated from. Always present so every shader's
    /// pipeline layout is uniform; a shader that ignores the UBO is fine.
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    /// One [`Frame`] per ring slot; indexed by the slot passed to [`Self::render`].
    frames: Vec<Frame>,
    /// Loader to export the render semaphore as a sync_file FD.
    external_semaphore_fd_fn: ash::khr::external_semaphore_fd::Device,
    /// True when the driver can export a SYNC_FD semaphore: we hand the
    /// compositor an implicit fence and never block. False (rare) falls back
    /// to a CPU fence wait before presenting.
    sync_file: bool,
    /// Feedback machinery, present only when the shader samples `iPrevFrame`.
    /// When set, `pipeline` renders the user shader into a feedback texture
    /// (via `feedback.render_pass`) and the blit presents it; when `None`, the
    /// shader renders straight into the dmabuf as before.
    feedback: Option<Feedback>,
}

impl ShaderRenderer {
    /// Build the pipeline for `fragment_glsl` (a fragment shader providing
    /// `main` and the `Push` uniform block — see [`DEFAULT_FRAGMENT_GLSL`]),
    /// with `frames` per-slot command resources (one per ring buffer). When
    /// `feedback` is set, the shader samples `iPrevFrame` (its previous output):
    /// it's rendered into a ping-pong texture and presented via a built-in blit.
    pub fn new(
        gpu: &Gpu,
        fragment_glsl: &str,
        frames: usize,
        feedback: bool,
    ) -> Result<ShaderRenderer> {
        let device = gpu.device.clone();
        let vert_spv = compile_glsl(VERTEX_GLSL, shaderc::ShaderKind::Vertex, "fullscreen.vert")?;
        let frag_spv = compile_glsl(
            fragment_glsl,
            shaderc::ShaderKind::Fragment,
            "wallpaper.frag",
        )?;

        let render_pass = Self::create_render_pass(&device)?;
        match Self::build(
            gpu,
            device.clone(),
            render_pass,
            &vert_spv,
            &frag_spv,
            frames,
            feedback,
        ) {
            Ok(r) => Ok(r),
            Err(e) => {
                // SAFETY: render_pass created just above, nothing else owns it.
                unsafe { device.destroy_render_pass(render_pass, None) };
                Err(e)
            }
        }
    }

    /// Whether `gpu` can export a binary semaphore as a SYNC_FD sync_file —
    /// the basis of the implicit-sync present path. Practically always true on
    /// modern drivers; if not, [`Self::render`] falls back to a CPU wait.
    fn sync_file_exportable(gpu: &Gpu) -> bool {
        let info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let mut props = vk::ExternalSemaphoreProperties::default();
        // SAFETY: physical_device came from gpu.instance; props is owned here.
        unsafe {
            gpu.instance.get_physical_device_external_semaphore_properties(
                gpu.physical_device,
                &info,
                &mut props,
            )
        };
        props
            .external_semaphore_features
            .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE)
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
        frame_count: usize,
        feedback: bool,
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

        // Spectrum UBO at set 0, binding 0 (fragment stage). Created even for
        // shaders that don't sample audio, so the pipeline layout is uniform.
        let ubo_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let ubo_bindings = [ubo_binding];
        let set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&ubo_bindings);
        // SAFETY: set_layout_info outlives the call.
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }
                .context("creating descriptor set layout")?;

        // Feedback machinery first (when requested): the user pipeline must be
        // built against its off-screen render pass and gain the sampler set.
        let feedback = if feedback {
            match Self::build_feedback(&device, render_pass) {
                Ok(f) => Some(f),
                Err(e) => {
                    // SAFETY: only these three exist so far.
                    unsafe {
                        device.destroy_shader_module(vert, None);
                        device.destroy_shader_module(frag, None);
                        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    }
                    return Err(e);
                }
            }
        } else {
            None
        };

        // Helper to unwind everything created above on a later failure (no
        // ShaderRenderer exists yet, so Drop won't run). render_pass is freed
        // by the caller (`new`).
        let cleanup = |layout: Option<vk::PipelineLayout>, pipeline: Option<vk::Pipeline>| {
            // SAFETY: each handle was created above and is freed once here.
            unsafe {
                if let Some(p) = pipeline {
                    device.destroy_pipeline(p, None);
                }
                if let Some(l) = layout {
                    device.destroy_pipeline_layout(l, None);
                }
                if let Some(f) = &feedback {
                    f.destroy(&device);
                }
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
        };

        // User pipeline layout: set 0 = audio UBO; set 1 = `iPrevFrame` sampler
        // (feedback only).
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<ShaderUniforms>() as u32);
        let push_ranges = [push_range];
        let mut set_layouts = vec![descriptor_set_layout];
        if let Some(f) = &feedback {
            set_layouts.push(f.sampler_set_layout);
        }
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        // SAFETY: layout_info outlives the call.
        let layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
            Ok(l) => l,
            Err(e) => {
                // SAFETY: shader modules still live here.
                unsafe {
                    device.destroy_shader_module(vert, None);
                    device.destroy_shader_module(frag, None);
                }
                cleanup(None, None);
                return Err(e).context("creating pipeline layout");
            }
        };

        // Feedback shaders render into the off-screen feedback pass; otherwise
        // straight into the dmabuf render pass.
        let user_render_pass = feedback
            .as_ref()
            .map(|f| f.render_pass)
            .unwrap_or(render_pass);
        let pipeline_result = Self::create_pipeline(&device, user_render_pass, layout, vert, frag);
        // Shader modules are consumed by pipeline creation; destroy regardless.
        // SAFETY: vert/frag are this device's, no longer referenced.
        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
        }
        let pipeline = match pipeline_result {
            Ok(p) => p,
            Err(e) => {
                cleanup(Some(layout), None);
                return Err(e);
            }
        };

        let sync_file = Self::sync_file_exportable(gpu);
        if !sync_file {
            tracing::warn!(
                "driver can't export a SYNC_FD semaphore; shader present falls back to a \
                 blocking CPU fence wait each frame"
            );
        }

        // The descriptor pool and per-slot resources, with self-contained
        // cleanup: on any failure here `build_frames` frees what it created and
        // we tear down everything else above.
        let (descriptor_pool, frames) =
            match Self::build_frames(gpu, &device, descriptor_set_layout, sync_file, frame_count) {
                Ok(v) => v,
                Err(e) => {
                    cleanup(Some(layout), Some(pipeline));
                    return Err(e);
                }
            };

        Ok(ShaderRenderer {
            device,
            queue: gpu.queue,
            queue_family: gpu.queue_family,
            render_pass,
            layout,
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            frames,
            external_semaphore_fd_fn: gpu.external_semaphore_fd_fn.clone(),
            sync_file,
            feedback,
        })
    }

    /// Create the descriptor pool and one [`Frame`] per ring slot (command
    /// buffer + fence + export semaphore + spectrum UBO/descriptor). On any
    /// failure, frees everything it allocated so far so `build` can fail
    /// cleanly without a half-built renderer to drop.
    fn build_frames(
        gpu: &Gpu,
        device: &ash::Device,
        set_layout: vk::DescriptorSetLayout,
        sync_file: bool,
        frame_count: usize,
    ) -> Result<(vk::DescriptorPool, Vec<Frame>)> {
        // Pool sized for one spectrum UBO descriptor per ring slot.
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(frame_count as u32);
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(frame_count as u32)
            .pool_sizes(&pool_sizes);
        // SAFETY: pool_info outlives the call. Nothing else allocated yet, so a
        // failure here needs no cleanup.
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .context("creating descriptor pool")?;

        // One command buffer + fence + (export) semaphore per ring slot, so
        // frames pipeline instead of serializing through a single fence.
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(gpu.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(frame_count as u32);
        // SAFETY: alloc references gpu.command_pool, valid for this device.
        let command_buffers = match unsafe { device.allocate_command_buffers(&alloc) } {
            Ok(c) => c,
            Err(e) => {
                // SAFETY: pool created just above, nothing references it.
                unsafe { device.destroy_descriptor_pool(descriptor_pool, None) };
                return Err(e).context("allocating command buffers");
            }
        };

        let mut frames: Vec<Frame> = Vec::with_capacity(frame_count);
        for &command_buffer in &command_buffers {
            match Self::build_one_frame(gpu, device, descriptor_pool, set_layout, sync_file) {
                Ok((fence, semaphore, ubo, ubo_memory, ubo_mapped, descriptor_set)) => {
                    frames.push(Frame {
                        command_buffer,
                        fence,
                        semaphore,
                        ubo,
                        ubo_memory,
                        ubo_mapped,
                        descriptor_set,
                    });
                }
                Err(e) => {
                    // Free the slots already built, then the pool (which frees
                    // their descriptor sets and the command buffers).
                    // SAFETY: every handle below was created by this function.
                    unsafe {
                        for f in &frames {
                            device.destroy_fence(f.fence, None);
                            device.destroy_semaphore(f.semaphore, None);
                            device.destroy_buffer(f.ubo, None);
                            device.free_memory(f.ubo_memory, None);
                        }
                        device.destroy_descriptor_pool(descriptor_pool, None);
                    }
                    return Err(e);
                }
            }
        }

        Ok((descriptor_pool, frames))
    }

    /// Create one slot's fence, export semaphore, and spectrum UBO. The caller
    /// owns the returned handles (and cleans them up on a later slot's failure).
    #[allow(clippy::type_complexity)]
    fn build_one_frame(
        gpu: &Gpu,
        device: &ash::Device,
        pool: vk::DescriptorPool,
        set_layout: vk::DescriptorSetLayout,
        sync_file: bool,
    ) -> Result<(
        vk::Fence,
        vk::Semaphore,
        vk::Buffer,
        vk::DeviceMemory,
        *mut u8,
        vk::DescriptorSet,
    )> {
        // Pre-signaled: render() waits on the fence before reusing the slot, so
        // the first use must find it already signaled.
        let fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .context("creating frame fence")?;
        // Exportable as a sync_file when supported.
        let mut export = vk::ExportSemaphoreCreateInfo::default().handle_types(if sync_file {
            vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD
        } else {
            vk::ExternalSemaphoreHandleTypeFlags::empty()
        });
        let semaphore = match unsafe {
            device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut export), None)
        } {
            Ok(s) => s,
            Err(e) => {
                // SAFETY: fence created just above.
                unsafe { device.destroy_fence(fence, None) };
                return Err(e).context("creating frame semaphore");
            }
        };
        let (ubo, ubo_memory, ubo_mapped, descriptor_set) =
            match Self::create_ubo(gpu, device, pool, set_layout) {
                Ok(v) => v,
                Err(e) => {
                    // SAFETY: fence + semaphore created just above.
                    unsafe {
                        device.destroy_semaphore(semaphore, None);
                        device.destroy_fence(fence, None);
                    }
                    return Err(e);
                }
            };
        Ok((fence, semaphore, ubo, ubo_memory, ubo_mapped, descriptor_set))
    }

    /// Create one slot's spectrum UBO: a host-visible, persistently-mapped
    /// buffer sized for [`AudioUniforms`], plus a descriptor set pointing at
    /// it. The mapping is coherent so [`Self::render`] can write the audio
    /// snapshot with a plain `copy` and no flush. Zero-initialized so a shader
    /// reads silence before any audio arrives.
    fn create_ubo(
        gpu: &Gpu,
        device: &ash::Device,
        pool: vk::DescriptorPool,
        set_layout: vk::DescriptorSetLayout,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8, vk::DescriptorSet)> {
        let size = std::mem::size_of::<AudioUniforms>() as u64;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: buffer_info outlives the call.
        let buffer = unsafe { device.create_buffer(&buffer_info, None) }
            .context("creating spectrum UBO buffer")?;

        // From here, free what's been created on any failure: this fn owns the
        // buffer/memory until it returns Ok. SAFETY: each handle freed is one
        // this fn created, exactly once.
        let cleanup_buffer = || unsafe { device.destroy_buffer(buffer, None) };
        let cleanup_buffer_memory = |memory: vk::DeviceMemory| unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        };

        // SAFETY: buffer is this device's.
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let Some(mem_type) = gpu.find_memory_type(
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) else {
            cleanup_buffer();
            anyhow::bail!("no host-visible coherent memory for spectrum UBO");
        };
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(mem_type);
        // SAFETY: alloc is valid.
        let memory = match unsafe { device.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                cleanup_buffer();
                return Err(e).context("allocating spectrum UBO memory");
            }
        };
        // SAFETY: buffer and memory are this device's, bound once.
        if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            cleanup_buffer_memory(memory);
            return Err(e).context("binding spectrum UBO memory");
        }
        // SAFETY: memory is host-visible; mapped for the buffer's whole life.
        let mapped = match unsafe {
            device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        } {
            Ok(p) => p as *mut u8,
            Err(e) => {
                cleanup_buffer_memory(memory);
                return Err(e).context("mapping spectrum UBO");
            }
        };
        // SAFETY: `mapped` is valid for `size` bytes; zero = silence.
        unsafe { std::ptr::write_bytes(mapped, 0, size as usize) };

        let set_layouts = [set_layout];
        let set_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts);
        // SAFETY: set_alloc outlives the call; pool has room (sized per slot).
        let descriptor_set = match unsafe { device.allocate_descriptor_sets(&set_alloc) } {
            Ok(sets) => sets[0],
            Err(e) => {
                cleanup_buffer_memory(memory);
                return Err(e).context("allocating spectrum descriptor set");
            }
        };

        let buffer_descriptor = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(size);
        let buffer_descriptors = [buffer_descriptor];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_descriptors);
        // SAFETY: write references live handles for the call's duration.
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        Ok((buffer, memory, mapped, descriptor_set))
    }

    /// Build the size-independent feedback machinery: sampler, off-screen
    /// render pass, the combined-image-sampler set layout + pool + two sets,
    /// and the blit pipeline (presents a feedback texture into the dmabuf via
    /// `present_render_pass`). Textures/framebuffers come later, on resize.
    /// Cleans up on any internal failure.
    fn build_feedback(
        device: &ash::Device,
        present_render_pass: vk::RenderPass,
    ) -> Result<Feedback> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        // SAFETY: sampler_info outlives the call.
        let sampler = unsafe { device.create_sampler(&sampler_info, None) }
            .context("creating feedback sampler")?;

        let render_pass = match Self::create_feedback_render_pass(device) {
            Ok(r) => r,
            Err(e) => {
                unsafe { device.destroy_sampler(sampler, None) };
                return Err(e);
            }
        };

        // Combined image sampler at binding 0 (fragment): set 1 of the user
        // pipeline (iPrevFrame), set 0 of the blit.
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let bindings = [binding];
        let sl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: sl_info outlives the call.
        let sampler_set_layout =
            match unsafe { device.create_descriptor_set_layout(&sl_info, None) } {
                Ok(l) => l,
                Err(e) => {
                    unsafe {
                        device.destroy_render_pass(render_pass, None);
                        device.destroy_sampler(sampler, None);
                    }
                    return Err(e).context("creating feedback sampler set layout");
                }
            };

        // Cleanup helper for the remaining steps.
        let undo = |pool: Option<vk::DescriptorPool>, blit_layout: Option<vk::PipelineLayout>| {
            // SAFETY: each handle was created in this fn and is freed once.
            unsafe {
                if let Some(l) = blit_layout {
                    device.destroy_pipeline_layout(l, None);
                }
                if let Some(p) = pool {
                    device.destroy_descriptor_pool(p, None);
                }
                device.destroy_descriptor_set_layout(sampler_set_layout, None);
                device.destroy_render_pass(render_pass, None);
                device.destroy_sampler(sampler, None);
            }
        };

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(2);
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&pool_sizes);
        // SAFETY: pool_info outlives the call.
        let descriptor_pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(e) => {
                undo(None, None);
                return Err(e).context("creating feedback descriptor pool");
            }
        };

        let layouts = [sampler_set_layout, sampler_set_layout];
        let set_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        // SAFETY: set_alloc outlives the call; pool sized for 2 sets.
        let tex_sets = match unsafe { device.allocate_descriptor_sets(&set_alloc) } {
            Ok(s) => [s[0], s[1]],
            Err(e) => {
                undo(Some(descriptor_pool), None);
                return Err(e).context("allocating feedback descriptor sets");
            }
        };

        // Blit pipeline layout: just the sampler set (no push constants).
        let blit_set_layouts = [sampler_set_layout];
        let blit_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&blit_set_layouts);
        // SAFETY: blit_layout_info outlives the call.
        let blit_layout = match unsafe { device.create_pipeline_layout(&blit_layout_info, None) } {
            Ok(l) => l,
            Err(e) => {
                undo(Some(descriptor_pool), None);
                return Err(e).context("creating blit pipeline layout");
            }
        };

        // Blit pipeline: compile the built-in blit shaders and build against the
        // dmabuf (present) render pass.
        let blit_pipeline = match Self::build_blit_pipeline(device, present_render_pass, blit_layout)
        {
            Ok(p) => p,
            Err(e) => {
                undo(Some(descriptor_pool), Some(blit_layout));
                return Err(e);
            }
        };

        Ok(Feedback {
            sampler,
            render_pass,
            sampler_set_layout,
            descriptor_pool,
            tex_sets,
            blit_layout,
            blit_pipeline,
            textures: None,
            framebuffers: [vk::Framebuffer::null(); 2],
            size: (0, 0),
            parity: std::cell::Cell::new(0),
        })
    }

    /// Compile and build the built-in blit pipeline (textured fullscreen
    /// triangle) for `render_pass`.
    fn build_blit_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline> {
        let vert_spv = compile_glsl(BLIT_VERTEX_GLSL, shaderc::ShaderKind::Vertex, "blit.vert")?;
        let frag_spv = compile_glsl(BLIT_FRAGMENT_GLSL, shaderc::ShaderKind::Fragment, "blit.frag")?;
        // SAFETY: SPIR-V slices live across module creation.
        let vert = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_spv), None)
        }
        .context("creating blit vertex module")?;
        let frag = match unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_spv), None)
        } {
            Ok(f) => f,
            Err(e) => {
                unsafe { device.destroy_shader_module(vert, None) };
                return Err(e).context("creating blit fragment module");
            }
        };
        let result = Self::create_pipeline(device, render_pass, layout, vert, frag);
        // SAFETY: modules consumed by pipeline creation.
        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
        }
        result
    }

    /// Off-screen render pass writing a feedback texture (F16, → SHADER_READ).
    /// The subpass dependencies make this frame's write visible to later
    /// fragment reads (blit + next frame's sample), and order this write after
    /// the previous frame's read of the same ping-pong texture (a WAR hazard the
    /// 3-deep per-slot fence doesn't cover, since parity cycles every 2 frames).
    fn create_feedback_render_pass(device: &ash::Device) -> Result<vk::RenderPass> {
        let attachment = vk::AttachmentDescription::default()
            .format(RENDER_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            // Fully overwritten each frame; old contents (this texture's prior
            // generation) are irrelevant.
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let attachments = [attachment];
        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_refs = [color_ref];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs);
        let subpasses = [subpass];
        let deps = [
            // WAR: the previous frame's sample of this texture (as iPrevFrame)
            // must finish before we overwrite it.
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            // RAW: this write must be visible to later fragment samples (the
            // blit in this command buffer, and the next frame's iPrevFrame).
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags::SHADER_READ),
        ];
        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&deps);
        // SAFETY: info outlives the call.
        unsafe { device.create_render_pass(&info, None) }.context("creating feedback render pass")
    }

    /// (Re)create the feedback ping-pong textures + framebuffers at `w`×`h`,
    /// re-point the descriptor sets at them, and clear both to black. No-op for
    /// non-feedback renderers or an unchanged size. Called from the ring rebuild.
    pub fn resize_feedback(&mut self, gpu: &Gpu, w: u32, h: u32) -> Result<()> {
        let Some(fb) = self.feedback.as_mut() else {
            return Ok(());
        };
        if fb.textures.is_some() && fb.size == (w, h) {
            return Ok(());
        }
        let device = &self.device;
        // SAFETY: wait for any in-flight use before freeing the old textures.
        unsafe {
            let _ = device.device_wait_idle();
            if let Some(t) = &fb.textures {
                for img in t {
                    img.destroy(device);
                }
            }
            for f in fb.framebuffers {
                if f != vk::Framebuffer::null() {
                    device.destroy_framebuffer(f, None);
                }
            }
        }
        fb.textures = None;
        fb.framebuffers = [vk::Framebuffer::null(); 2];

        // Create the two textures + their framebuffers. Track both in Vecs so a
        // mid-loop allocation failure can free everything built so far.
        let mut images: Vec<FeedbackImage> = Vec::with_capacity(2);
        let mut fbs: Vec<vk::Framebuffer> = Vec::with_capacity(2);
        // SAFETY: frees every handle built so far in this loop, each once.
        let undo = |device: &ash::Device, images: &[FeedbackImage], fbs: &[vk::Framebuffer]| unsafe {
            for f in fbs {
                device.destroy_framebuffer(*f, None);
            }
            for i in images {
                i.destroy(device);
            }
        };
        for _ in 0..2 {
            let img = match create_feedback_image(gpu, device, w, h) {
                Ok(i) => i,
                Err(e) => {
                    undo(device, &images, &fbs);
                    return Err(e);
                }
            };
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(fb.render_pass)
                .attachments(std::slice::from_ref(&img.view))
                .width(w)
                .height(h)
                .layers(1);
            // SAFETY: fb_info outlives the call.
            let framebuffer = match unsafe { device.create_framebuffer(&fb_info, None) } {
                Ok(f) => f,
                Err(e) => {
                    // SAFETY: this image isn't tracked yet; free it + the rest.
                    unsafe { img.destroy(device) };
                    undo(device, &images, &fbs);
                    return Err(e).context("creating feedback framebuffer");
                }
            };
            images.push(img);
            fbs.push(framebuffer);
        }
        let framebuffers = [fbs[0], fbs[1]];

        // Point each descriptor set at its texture.
        for (set, img) in fb.tex_sets.iter().zip(images.iter()) {
            let info = vk::DescriptorImageInfo::default()
                .sampler(fb.sampler)
                .image_view(img.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            let infos = [info];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&infos);
            // SAFETY: write references live handles for the call.
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }

        // Clear both to black and move them to SHADER_READ so the first frame
        // samples a clean previous frame (no NaNs to poison the feedback loop).
        let textures = [images.remove(0), images.remove(0)];
        if let Err(e) = clear_feedback_textures(gpu, device, &textures) {
            // SAFETY: textures/framebuffers not yet stored; free them.
            unsafe {
                for img in &textures {
                    img.destroy(device);
                }
                for f in framebuffers {
                    device.destroy_framebuffer(f, None);
                }
            }
            return Err(e);
        }

        fb.textures = Some(textures);
        fb.framebuffers = framebuffers;
        fb.size = (w, h);
        fb.parity.set(0);
        Ok(())
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

    /// Render one frame into ring slot `slot`'s `target` via `framebuffer`.
    ///
    /// On the fast path (`sync_file`) the GPU work is submitted and its
    /// completion semaphore is exported as a sync_file and attached to the
    /// dmabuf as its implicit write fence — so the caller may `commit`
    /// immediately and the *compositor* waits on the GPU (implicit sync),
    /// never the calloop thread. If the driver can't export a sync_file we
    /// fall back to a CPU fence wait before returning. No readback.
    pub fn render(
        &self,
        slot: usize,
        target: &RenderTarget,
        framebuffer: vk::Framebuffer,
        uniforms: &ShaderUniforms,
        audio: &AudioUniforms,
    ) -> Result<()> {
        let device = &self.device;
        let frame = &self.frames[slot];
        let cb = frame.command_buffer;
        // SAFETY: all handles are this renderer's; this slot's previous
        // submission is awaited via its fence before reuse below.
        unsafe {
            device
                .wait_for_fences(&[frame.fence], true, u64::MAX)
                .context("waiting on prior frame fence")?;
            device
                .reset_fences(&[frame.fence])
                .context("resetting fence")?;

            // Safe to overwrite now: the fence wait above means this slot's
            // previous read of the UBO is complete. Coherent mapping → no flush.
            std::ptr::copy_nonoverlapping(
                bytemuck::bytes_of(audio).as_ptr(),
                frame.ubo_mapped,
                std::mem::size_of::<AudioUniforms>(),
            );
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
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: target.width as f32,
                height: target.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };

            if let Some(fb) = &self.feedback {
                // Two-pass feedback. Pass 1: the user shader renders into the
                // ping-pong write target, sampling the previous frame as
                // `iPrevFrame`. Pass 2: a built-in blit presents that result
                // into the dmabuf. The feedback render pass's own dependencies
                // order the cross-frame WAR and the pass1→pass2 RAW.
                let textures = fb
                    .textures
                    .as_ref()
                    .context("feedback render before textures built")?;
                let _ = textures; // presence checked; framebuffers index in lockstep
                let curr = fb.parity.get();
                let prev = 1 - curr;

                // Pass 1: user shader → feedback[curr]. loadOp is DONT_CARE, so
                // no clear values.
                let rp1 = vk::RenderPassBeginInfo::default()
                    .render_pass(fb.render_pass)
                    .framebuffer(fb.framebuffers[curr])
                    .render_area(area);
                device.cmd_begin_render_pass(cb, &rp1, vk::SubpassContents::INLINE);
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    &[frame.descriptor_set, fb.tex_sets[prev]],
                    &[],
                );
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

                // Pass 2: blit feedback[curr] → dmabuf (clears, then samples).
                let rp2 = vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(framebuffer)
                    .render_area(area)
                    .clear_values(&clears);
                device.cmd_begin_render_pass(cb, &rp2, vk::SubpassContents::INLINE);
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, fb.blit_pipeline);
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    fb.blit_layout,
                    0,
                    &[fb.tex_sets[curr]],
                    &[],
                );
                device.cmd_set_viewport(cb, 0, &[viewport]);
                device.cmd_set_scissor(cb, 0, &[area]);
                device.cmd_draw(cb, 3, 1, 0, 0);
                device.cmd_end_render_pass(cb);

                fb.parity.set(prev);
            } else {
                // Single pass: the shader renders straight into the dmabuf.
                let rp_begin = vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(framebuffer)
                    .render_area(area)
                    .clear_values(&clears);
                device.cmd_begin_render_pass(cb, &rp_begin, vk::SubpassContents::INLINE);
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    &[frame.descriptor_set],
                    &[],
                );
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
            }

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
            let signal = [frame.semaphore];
            let mut submit = vk::SubmitInfo::default().command_buffers(&cbs);
            if self.sync_file {
                submit = submit.signal_semaphores(&signal);
            }
            device
                .queue_submit(self.queue, &[submit], frame.fence)
                .context("submitting render")?;
        }

        if !self.sync_file {
            // No exportable sync_file: block until the GPU is done so the
            // committed dmabuf holds finished pixels.
            // SAFETY: the fence was just submitted with this frame's work.
            unsafe { device.wait_for_fences(&[frame.fence], true, u64::MAX) }
                .context("waiting on render fence")?;
            return Ok(());
        }

        // Export the completion semaphore as a sync_file and attach it to the
        // dmabuf as its implicit write fence; the compositor's implicit-sync
        // read then waits on the GPU. A -1 FD means the work already finished
        // (nothing to wait on). If the attach fails, degrade to a CPU wait so
        // we never present a half-rendered buffer.
        if let Some(sync) = self.export_sync_file(frame.semaphore)? {
            if let Err(e) = import_sync_file_to_dmabuf(target.fd.as_raw_fd(), sync) {
                tracing::warn!("dmabuf sync_file import failed ({e:#}); blocking instead");
                // SAFETY: this frame's fence was submitted above.
                unsafe { device.wait_for_fences(&[frame.fence], true, u64::MAX) }
                    .context("waiting on render fence (import fallback)")?;
            }
        }
        Ok(())
    }

    /// Export `semaphore`'s pending payload as a sync_file FD. Returns `None`
    /// for the special -1 FD (semaphore already signaled — no fence to wait).
    fn export_sync_file(&self, semaphore: vk::Semaphore) -> Result<Option<OwnedFd>> {
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        // SAFETY: semaphore is this renderer's, created exportable; the loader
        // matches this device.
        let raw = unsafe { self.external_semaphore_fd_fn.get_semaphore_fd(&info) }
            .context("exporting render semaphore as sync_file")?;
        if raw < 0 {
            Ok(None)
        } else {
            // SAFETY: get_semaphore_fd transferred ownership of this FD to us.
            Ok(Some(unsafe { OwnedFd::from_raw_fd(raw) }))
        }
    }
}

/// Attach `sync_file` to `dmabuf_fd` as an implicit *write* fence, so any
/// later implicit-sync reader (the compositor) waits for it before sampling.
/// `sync_file` is closed on return; the kernel holds its own reference.
fn import_sync_file_to_dmabuf(dmabuf_fd: RawFd, sync_file: OwnedFd) -> Result<()> {
    // linux/dma-buf.h: DMA_BUF_SYNC_WRITE, and
    // DMA_BUF_IOCTL_IMPORT_SYNC_FILE = _IOW('b', 3, struct dma_buf_import_sync_file).
    const DMA_BUF_SYNC_WRITE: u32 = 2;
    const DMA_BUF_IOCTL_IMPORT_SYNC_FILE: libc::c_ulong = 0x4008_6203;
    #[repr(C)]
    struct DmaBufImportSyncFile {
        flags: u32,
        fd: i32,
    }
    let mut data = DmaBufImportSyncFile {
        flags: DMA_BUF_SYNC_WRITE,
        fd: sync_file.as_raw_fd(),
    };
    // SAFETY: dmabuf_fd is a live dma-buf; data matches the kernel's struct
    // and outlives the call; the ioctl only reads from it.
    let rc = unsafe {
        libc::ioctl(
            dmabuf_fd,
            DMA_BUF_IOCTL_IMPORT_SYNC_FILE,
            &mut data as *mut DmaBufImportSyncFile,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error())
            .context("DMA_BUF_IOCTL_IMPORT_SYNC_FILE");
    }
    Ok(())
}

impl Drop for ShaderRenderer {
    fn drop(&mut self) {
        // SAFETY: all handles are this renderer's. On the sync_file path a
        // submission may still be in flight, so wait every slot's fence
        // before tearing down. Command buffers are freed with the pool.
        unsafe {
            let fences: Vec<vk::Fence> = self.frames.iter().map(|f| f.fence).collect();
            if !fences.is_empty() {
                let _ = self.device.wait_for_fences(&fences, true, u64::MAX);
            }
            for frame in &self.frames {
                self.device.destroy_fence(frame.fence, None);
                self.device.destroy_semaphore(frame.semaphore, None);
                // Unmap is implicit on free; destroy buffer then its memory.
                self.device.destroy_buffer(frame.ubo, None);
                self.device.free_memory(frame.ubo_memory, None);
            }
            // Sets are freed with the pool.
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            if let Some(fb) = &self.feedback {
                fb.destroy(&self.device);
            }
            self.device.destroy_render_pass(self.render_pass, None);
        }
    }
}

impl Feedback {
    /// Free every owned object. SAFETY: called once (Drop or a build error
    /// path), after in-flight work has drained; handles are `device`'s.
    unsafe fn destroy(&self, device: &ash::Device) {
        if let Some(textures) = &self.textures {
            for img in textures {
                img.destroy(device);
            }
        }
        for f in self.framebuffers {
            if f != vk::Framebuffer::null() {
                device.destroy_framebuffer(f, None);
            }
        }
        device.destroy_pipeline(self.blit_pipeline, None);
        device.destroy_pipeline_layout(self.blit_layout, None);
        // tex_sets are freed with the pool.
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.sampler_set_layout, None);
        device.destroy_render_pass(self.render_pass, None);
        device.destroy_sampler(self.sampler, None);
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
        let renderer = ShaderRenderer::new(gpu, DEFAULT_FRAGMENT_GLSL, 1, false).expect("renderer");
        let fb = renderer.create_framebuffer(&rt).expect("framebuffer");
        let uniforms = ShaderUniforms {
            resolution: [rt.width as f32, rt.height as f32],
            time: 1.25,
            _pad: 0.0,
            output_offset: [0.0, 0.0],
            output_size: [rt.width as f32, rt.height as f32],
            global_resolution: [rt.width as f32, rt.height as f32],
        };
        // Exercises the sync_file export + dmabuf import attach on this GPU,
        // plus the spectrum UBO upload + descriptor bind.
        let audio = <AudioUniforms as bytemuck::Zeroable>::zeroed();
        renderer
            .render(0, &rt, fb, &uniforms, &audio)
            .expect("render frame");
        // SAFETY: fb came from this renderer; device_wait_idle drains the
        // submission (the sync_file path does not block in render()).
        unsafe {
            let _ = gpu.device.device_wait_idle();
            gpu.device.destroy_framebuffer(fb, None);
        }
    }

    /// Build a feedback shader, allocate its ping-pong textures, and render two
    /// frames (so the second samples the first). Exercises the off-screen pass,
    /// the blit, the iPrevFrame descriptor, and parity. Run with
    /// PRISM_BG_VK_VALIDATION=1 to surface validation errors.
    #[test]
    #[ignore]
    fn render_feedback() {
        const FEEDBACK_GLSL: &str = r#"#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 iResolution; float iTime; float _pad;
    vec2 iOutputOffset; vec2 iOutputSize; vec2 iGlobalResolution; } pc;
layout(set = 1, binding = 0) uniform sampler2D iPrevFrame;
void main() {
    vec2 uv = fragCoord / pc.iResolution;
    vec3 prev = texture(iPrevFrame, vec2(uv.x, 1.0 - uv.y)).rgb;
    outColor = vec4(prev * 0.95 + vec3(uv, 0.5) * 0.05, 1.0);
}
"#;
        let mut pool = GpuPool::new().expect("Vulkan bring-up failed");
        let gpu = pool.any().expect("no usable device");
        let mods = gpu.renderable_modifiers();
        let rt = RenderTarget::new(gpu, 320, 200, &mods).expect("render target");
        let mut renderer = ShaderRenderer::new(gpu, FEEDBACK_GLSL, 2, true).expect("renderer");
        renderer
            .resize_feedback(gpu, rt.width, rt.height)
            .expect("feedback textures");
        let fb = renderer.create_framebuffer(&rt).expect("framebuffer");
        let uniforms = ShaderUniforms {
            resolution: [rt.width as f32, rt.height as f32],
            time: 0.0,
            _pad: 0.0,
            output_offset: [0.0, 0.0],
            output_size: [rt.width as f32, rt.height as f32],
            global_resolution: [rt.width as f32, rt.height as f32],
        };
        let audio = <AudioUniforms as bytemuck::Zeroable>::zeroed();
        // Two frames: the second samples the first via iPrevFrame (parity flip).
        for slot in 0..2 {
            renderer
                .render(slot, &rt, fb, &uniforms, &audio)
                .expect("render feedback frame");
            // SAFETY: drain before reusing/reading on the next iteration.
            unsafe {
                let _ = gpu.device.device_wait_idle();
            }
        }
        // SAFETY: fb came from this renderer; work is drained above.
        unsafe {
            gpu.device.destroy_framebuffer(fb, None);
        }
    }
}
