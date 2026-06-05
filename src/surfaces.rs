//! Mode geometry and pixel upload.
//!
//! prism-bg never resamples an image itself: for everything except `tile`
//! the buffer holds the image at native resolution and a `wp_viewport`
//! source-crop + destination tells the compositor where and how big to
//! draw it. prism composites in linear fp16, so compositor-side scaling is
//! gamma-correct — better than the cairo resampling swaybg does. `tile`
//! assembles an output-sized buffer (pure pixel copies, no resampling).

use anyhow::{Context, Result};
use smithay_client_toolkit::shm::slot::{Buffer, SlotPool};
use wayland_client::protocol::wl_shm;

use crate::cli::Mode;
use crate::decode::{DecodedImage, Pixels};

/// Where the image subsurface goes and what the viewport says.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    /// Subsurface position in the parent's logical coordinates.
    pub pos: (i32, i32),
    /// Viewport destination (logical size of the presented image).
    pub dest: (i32, i32),
    /// Viewport source rect in buffer coordinates, `None` = whole buffer.
    pub src: Option<(f64, f64, f64, f64)>,
    /// Buffer must be assembled by tiling at `(width, height)` device px.
    pub tile: Option<(u32, u32)>,
}

/// Compute the placement for `mode` on an output of `out` logical pixels
/// at integer `scale`, for an image of `img` pixels.
pub fn place(mode: Mode, out: (u32, u32), scale: i32, img: (u32, u32)) -> Placement {
    let (ow, oh) = (out.0 as f64, out.1 as f64);
    let (iw, ih) = (img.0 as f64, img.1 as f64);
    let s = scale.max(1) as f64;
    match mode {
        Mode::SolidColor => unreachable!("solid_color has no image part"),
        Mode::Stretch => Placement {
            pos: (0, 0),
            dest: (out.0 as i32, out.1 as i32),
            src: None,
            tile: None,
        },
        Mode::Fill => {
            // Scale to cover, crop the overflow axis (centered).
            let r = (ow / iw).max(oh / ih);
            let (crop_w, crop_h) = (ow / r, oh / r);
            Placement {
                pos: (0, 0),
                dest: (out.0 as i32, out.1 as i32),
                src: Some(((iw - crop_w) / 2.0, (ih - crop_h) / 2.0, crop_w, crop_h)),
                tile: None,
            }
        }
        Mode::Fit => {
            // Scale to contain, letterbox shows the background color.
            let r = (ow / iw).min(oh / ih);
            let (dw, dh) = ((iw * r).round().max(1.0), (ih * r).round().max(1.0));
            Placement {
                pos: (((ow - dw) / 2.0) as i32, ((oh - dh) / 2.0) as i32),
                dest: (dw as i32, dh as i32),
                src: None,
                tile: None,
            }
        }
        Mode::Center => {
            // 1:1 device pixels, centered; crop if the image is larger
            // than the output.
            let (vis_w, vis_h) = (iw.min(ow * s), ih.min(oh * s));
            let (dw, dh) = ((vis_w / s).round().max(1.0), (vis_h / s).round().max(1.0));
            Placement {
                pos: (((ow - dw) / 2.0) as i32, ((oh - dh) / 2.0) as i32),
                dest: (dw as i32, dh as i32),
                src: Some(((iw - vis_w) / 2.0, (ih - vis_h) / 2.0, vis_w, vis_h)),
                tile: None,
            }
        }
        Mode::Tile => Placement {
            pos: (0, 0),
            dest: (out.0 as i32, out.1 as i32),
            src: None,
            tile: Some(((ow * s) as u32, (oh * s) as u32)),
        },
    }
}

fn shm_format(img: &DecodedImage) -> wl_shm::Format {
    match img.pixels {
        // RGBA byte order == DRM/shm ABGR little-endian.
        Pixels::Rgba8(_) => wl_shm::Format::Abgr8888,
        Pixels::RgbaF16(_) => wl_shm::Format::Abgr16161616f,
    }
}

fn bytes_per_pixel(img: &DecodedImage) -> usize {
    match img.pixels {
        Pixels::Rgba8(_) => 4,
        Pixels::RgbaF16(_) => 8,
    }
}

fn pixel_bytes(img: &DecodedImage) -> &[u8] {
    match &img.pixels {
        Pixels::Rgba8(d) => d,
        Pixels::RgbaF16(d) => bytemuck::cast_slice(d),
    }
}

/// Upload the image at native size.
pub fn upload(pool: &mut SlotPool, img: &DecodedImage) -> Result<Buffer> {
    let bpp = bytes_per_pixel(img);
    let stride = img.width as usize * bpp;
    let (buffer, canvas) = pool
        .create_buffer(
            img.width as i32,
            img.height as i32,
            stride as i32,
            shm_format(img),
        )
        .context("creating shm buffer")?;
    canvas.copy_from_slice(pixel_bytes(img));
    Ok(buffer)
}

/// Assemble a `(width, height)` device-pixel buffer by repeating the image
/// from its top-left corner (the swaybg tiling origin).
pub fn upload_tiled(
    pool: &mut SlotPool,
    img: &DecodedImage,
    width: u32,
    height: u32,
) -> Result<Buffer> {
    let bpp = bytes_per_pixel(img);
    let stride = width as usize * bpp;
    let (buffer, canvas) = pool
        .create_buffer(width as i32, height as i32, stride as i32, shm_format(img))
        .context("creating tiled shm buffer")?;
    let src = pixel_bytes(img);
    let src_stride = img.width as usize * bpp;
    for y in 0..height as usize {
        let src_row = &src[(y % img.height as usize) * src_stride..][..src_stride];
        let dst_row = &mut canvas[y * stride..][..stride];
        for chunk in dst_row.chunks_mut(src_stride) {
            chunk.copy_from_slice(&src_row[..chunk.len()]);
        }
    }
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretch_covers_output() {
        let p = place(Mode::Stretch, (1920, 1080), 1, (800, 600));
        assert_eq!(p.dest, (1920, 1080));
        assert_eq!(p.pos, (0, 0));
        assert!(p.src.is_none() && p.tile.is_none());
    }

    #[test]
    fn fill_crops_the_overflow_axis() {
        // 1:1 image on a 2:1 output: full width used, half the height
        // cropped (centered).
        let p = place(Mode::Fill, (2000, 1000), 1, (1000, 1000));
        assert_eq!(p.dest, (2000, 1000));
        let (x, y, w, h) = p.src.unwrap();
        assert_eq!((x, w), (0.0, 1000.0));
        assert_eq!((y, h), (250.0, 500.0));
    }

    #[test]
    fn fit_letterboxes_and_centers() {
        // 1:1 image on a 2:1 output: height-limited, centered horizontally.
        let p = place(Mode::Fit, (2000, 1000), 1, (500, 500));
        assert_eq!(p.dest, (1000, 1000));
        assert_eq!(p.pos, (500, 0));
        assert!(p.src.is_none());
    }

    #[test]
    fn center_is_one_to_one_device_pixels() {
        // Small image on a 2x output: 200px image = 100 logical px.
        let p = place(Mode::Center, (1920, 1080), 2, (200, 100));
        assert_eq!(p.dest, (100, 50));
        assert_eq!(p.pos, ((1920 - 100) / 2, (1080 - 50) / 2));
        // Oversized image: cropped to the output, still 1:1.
        let p = place(Mode::Center, (100, 100), 1, (300, 100));
        assert_eq!(p.dest, (100, 100));
        assert_eq!(p.pos, (0, 0));
        let (x, _, w, _) = p.src.unwrap();
        assert_eq!((x, w), (100.0, 100.0));
    }

    #[test]
    fn tile_assembles_at_device_pixels() {
        let p = place(Mode::Tile, (1920, 1080), 2, (512, 512));
        assert_eq!(p.tile, Some((3840, 2160)));
        assert_eq!(p.dest, (1920, 1080));
    }
}
