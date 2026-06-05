//! Image loading: file → pixels + [`ColorEncoding`].
//!
//! Decoders produce a [`RawImage`] — straight-alpha pixels in the source's
//! own encoding, plus what we know about that encoding (a resolved
//! [`ColorEncoding`], or an ICC profile still to be resolved). [`finish`]
//! then runs ICC resolution (see [`crate::cms`]), premultiplies alpha, and
//! packs into one of the two wire formats we ship to the compositor:
//! 8-bit RGBA for ordinary SDR content, fp16 RGBA for everything that
//! needs more range or precision (>8-bit sources, linear/PQ encodings).
//!
//! Format dispatch is by magic bytes, not extension.

mod generic;
mod jxl;
mod jxr;
mod png;

use std::path::Path;

use anyhow::{bail, Context, Result};
use half::f16;

use crate::cms;
use crate::color::{ColorEncoding, Tf};

/// Wire pixel formats. Decoding produces `Rgba8` (`Abgr8888`) or `RgbaF16`
/// (`Abgr16161616f`); `Rgba16` (`Abgr16161616`, 16-bit unorm) only appears
/// via capability adaptation, for compositors with deep integer buffers
/// but no fp16 shm (KWin). Premultiplied alpha, tightly packed, RGBA
/// memory order.
#[derive(Debug, Clone)]
pub enum Pixels {
    Rgba8(Vec<u8>),
    Rgba16(Vec<u16>),
    RgbaF16(Vec<f16>),
}

#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Pixels,
    pub encoding: ColorEncoding,
    /// Whether any pixel has alpha < 1 (drives the surface opaque region).
    pub has_alpha: bool,
}

impl DecodedImage {
    /// Lossy fallback for compositors that don't accept fp16 shm buffers:
    /// quantize to 8-bit, encoding linear-light content with `target_tf`
    /// first (8-bit linear bands horribly in the darks) and retagging.
    /// Values above 1.0 clip to reference white. Display-referred TFs just
    /// clamp and keep their tag.
    pub fn quantized_to_8bit(&self, target_tf: Tf) -> DecodedImage {
        if let Pixels::Rgba16(d) = &self.pixels {
            // Already display-referred unorm (adaptation output); just
            // drop precision.
            return DecodedImage {
                pixels: Pixels::Rgba8(d.iter().map(|&v| (v >> 8) as u8).collect()),
                ..self.clone()
            };
        }
        let Pixels::RgbaF16(d) = &self.pixels else {
            return self.clone();
        };
        let linear = self.encoding.tf == Tf::Linear;
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let mut out = Vec::with_capacity(d.len());
        for px in d.chunks_exact(4) {
            let a = px[3].to_f32().clamp(0.0, 1.0);
            for &c in &px[..3] {
                let mut v = c.to_f32();
                if linear {
                    // Pixels are premultiplied; the OETF applies to the
                    // straight value, premultiplication to the encoded one.
                    if a > 0.0 {
                        v = target_tf.oetf((v / a).clamp(0.0, 1.0)) * a;
                    }
                } else {
                    v = v.clamp(0.0, 1.0);
                }
                out.push(q(v));
            }
            out.push(q(a));
        }
        let encoding = if linear {
            ColorEncoding {
                tf: target_tf,
                primaries: self.encoding.primaries,
                luminances: None,
            }
        } else {
            self.encoding
        };
        DecodedImage {
            pixels: Pixels::Rgba8(out),
            encoding,
            ..self.clone()
        }
    }

    /// Re-encode through a different display-referred TF (for compositors
    /// whose named-TF vocabulary lacks the source's — e.g. KWin dropped the
    /// protocol-deprecated `srgb`). Pixels are decoded with the source EOTF
    /// and re-encoded with the target OETF, through straight alpha; linear
    /// sources clip above reference white and drop their luminances.
    pub fn reencoded_tf(&self, target_tf: Tf) -> DecodedImage {
        let src_tf = self.encoding.tf;
        let convert = |v: f32| target_tf.oetf(src_tf.eotf(v.clamp(0.0, 1.0)));
        let pixels = match &self.pixels {
            Pixels::Rgba8(d) => {
                // Opaque pixels go through a per-channel LUT; translucent
                // ones need straight-alpha math.
                let lut: Vec<u8> = (0..=255u16)
                    .map(|v| (convert(v as f32 / 255.0) * 255.0 + 0.5) as u8)
                    .collect();
                let mut out = Vec::with_capacity(d.len());
                for px in d.chunks_exact(4) {
                    let a = px[3];
                    if a == 255 {
                        out.extend_from_slice(&[
                            lut[px[0] as usize],
                            lut[px[1] as usize],
                            lut[px[2] as usize],
                            255,
                        ]);
                    } else {
                        let af = a as f32 / 255.0;
                        for &c in &px[..3] {
                            let v = if a == 0 {
                                0.0
                            } else {
                                convert(c as f32 / 255.0 / af) * af
                            };
                            out.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                        }
                        out.push(a);
                    }
                }
                Pixels::Rgba8(out)
            }
            Pixels::Rgba16(d) => {
                let mut out = Vec::with_capacity(d.len());
                for px in d.chunks_exact(4) {
                    let a = px[3] as f32 / 65535.0;
                    for &c in &px[..3] {
                        let v = if a > 0.0 {
                            convert(c as f32 / 65535.0 / a) * a
                        } else {
                            0.0
                        };
                        out.push((v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16);
                    }
                    out.push(px[3]);
                }
                Pixels::Rgba16(out)
            }
            Pixels::RgbaF16(d) => {
                let mut out = Vec::with_capacity(d.len());
                for px in d.chunks_exact(4) {
                    let a = px[3].to_f32().clamp(0.0, 1.0);
                    for &c in &px[..3] {
                        let v = if a > 0.0 {
                            convert(c.to_f32() / a) * a
                        } else {
                            0.0
                        };
                        out.push(f16::from_f32(v));
                    }
                    out.push(f16::from_f32(a));
                }
                Pixels::RgbaF16(out)
            }
        };
        DecodedImage {
            pixels,
            encoding: ColorEncoding {
                tf: target_tf,
                primaries: self.encoding.primaries,
                // Display-referred either side; luminance declarations
                // don't survive the conversion (linear sources clipped).
                luminances: None,
            },
            ..self.clone()
        }
    }

    /// Repack fp16 pixels into 16-bit unorm (`Abgr16161616`) for
    /// compositors with deep integer shm but no fp16 (KWin). Electrical
    /// content ([0,1] by definition) converts losslessly-for-display;
    /// linear HDR content PQ-encodes (perceptually transparent at 16 bits,
    /// preserving the full luminance range — this is the path that keeps
    /// HDR alive without fp16) or, without compositor PQ support, falls
    /// back to `sdr_tf` with highlights clipped at reference white.
    pub fn repacked_unorm16(&self, pq_supported: bool, sdr_tf: Tf) -> DecodedImage {
        let Pixels::RgbaF16(d) = &self.pixels else {
            return self.clone();
        };
        let q = |v: f32| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        let linear = self.encoding.tf == Tf::Linear;
        // 1.0 in linear content = this many cd/m² (scRGB 80, scene-linear
        // 203 — always set by the decoders that produce Tf::Linear).
        let ref_nits = self
            .encoding
            .luminances
            .map(|l| l.reference)
            .unwrap_or(80.0) as f32;

        let (convert, encoding): (Box<dyn Fn(f32) -> f32>, _) = if !linear {
            // Electrical signal already; just requantize.
            (Box::new(|v: f32| v), self.encoding)
        } else if pq_supported {
            (
                Box::new(move |v: f32| crate::color::pq_oetf(v.max(0.0) * ref_nits / 10000.0)),
                ColorEncoding {
                    tf: Tf::Pq,
                    primaries: self.encoding.primaries,
                    luminances: None,
                },
            )
        } else {
            (
                Box::new(move |v: f32| sdr_tf.oetf(v.clamp(0.0, 1.0))),
                ColorEncoding {
                    tf: sdr_tf,
                    primaries: self.encoding.primaries,
                    luminances: None,
                },
            )
        };

        let mut out = Vec::with_capacity(d.len());
        for px in d.chunks_exact(4) {
            let a = px[3].to_f32().clamp(0.0, 1.0);
            for &c in &px[..3] {
                // Through straight alpha: transform the straight value,
                // re-premultiply the encoded result.
                let v = if a > 0.0 { convert(c.to_f32() / a) * a } else { 0.0 };
                out.push(q(v));
            }
            out.push(q(a));
        }
        DecodedImage {
            pixels: Pixels::Rgba16(out),
            encoding,
            ..self.clone()
        }
    }
}

/// Straight-alpha RGBA pixels as decoded, before color resolution.
#[derive(Debug)]
pub enum RawPixels {
    Rgba8(Vec<u8>),
    /// 16-bit unorm (from 16-bit PNG, 10/12-bit AVIF upshifted by dav1d).
    Rgba16(Vec<u16>),
    /// Float values; electrical or linear depending on the encoding.
    RgbaF32(Vec<f32>),
}

/// What the decoder learned about the encoding.
#[derive(Debug)]
pub enum RawColor {
    /// Fully resolved (cICP, format convention, or known-default sRGB).
    Encoding(ColorEncoding),
    /// An ICC profile that still needs resolving against prism's
    /// parametric vocabulary.
    Icc(Vec<u8>),
}

#[derive(Debug)]
pub struct RawImage {
    pub width: u32,
    pub height: u32,
    pub pixels: RawPixels,
    pub color: RawColor,
}

pub fn load(path: &Path) -> Result<DecodedImage> {
    let data = std::fs::read(path)
        .with_context(|| format!("reading image {}", path.display()))?;
    let raw = decode(&data).with_context(|| format!("decoding {}", path.display()))?;
    finish(raw)
}

fn decode(data: &[u8]) -> Result<RawImage> {
    match sniff(data) {
        Some(Format::Png) => png::decode(data),
        Some(Format::Jxl) => jxl::decode(data),
        Some(Format::Jxr) => jxr::decode(data),
        Some(f) => generic::decode(data, f),
        None => bail!("unrecognized image format"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Png,
    Jpeg,
    WebP,
    Jxl,
    Jxr,
    Avif,
    Exr,
    Hdr,
}

fn sniff(data: &[u8]) -> Option<Format> {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some(Format::Png)
    } else if data.starts_with(&[0xff, 0xd8, 0xff]) {
        Some(Format::Jpeg)
    } else if data.len() > 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        Some(Format::WebP)
    } else if data.starts_with(&[0xff, 0x0a])
        || data.starts_with(b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a")
    {
        Some(Format::Jxl)
    } else if data.starts_with(&[0x49, 0x49, 0xbc]) {
        // TIFF-style little-endian header with the JXR signature byte.
        Some(Format::Jxr)
    } else if data.len() > 12 && &data[4..8] == b"ftyp" && &data[8..12] == b"avif" {
        Some(Format::Avif)
    } else if data.starts_with(&[0x76, 0x2f, 0x31, 0x01]) {
        Some(Format::Exr)
    } else if data.starts_with(b"#?RADIANCE") || data.starts_with(b"#?RGBE") {
        Some(Format::Hdr)
    } else {
        None
    }
}

/// Resolve color (ICC → parametric, possibly rewriting pixels), premultiply,
/// and pack to a wire format.
fn finish(raw: RawImage) -> Result<DecodedImage> {
    let RawImage {
        width,
        height,
        pixels,
        color,
    } = raw;

    let (pixels, encoding) = match color {
        RawColor::Encoding(enc) => (pixels, enc),
        RawColor::Icc(icc) => cms::resolve_icc(&icc, pixels)?,
    };

    // Pack. 8-bit stays 8-bit only for plain display-referred TFs; linear
    // and PQ data, and anything wider than 8 bits, rides fp16.
    let keep_8bit = matches!(encoding.tf, Tf::Srgb | Tf::Gamma22 | Tf::Bt1886);
    let (pixels, has_alpha) = match pixels {
        RawPixels::Rgba8(data) if keep_8bit => {
            let mut data = data;
            let has_alpha = premultiply_u8(&mut data);
            (Pixels::Rgba8(data), has_alpha)
        }
        other => {
            let mut data = to_f16(other);
            let has_alpha = premultiply_f16(&mut data);
            (Pixels::RgbaF16(data), has_alpha)
        }
    };

    Ok(DecodedImage {
        width,
        height,
        pixels,
        encoding,
        has_alpha,
    })
}

fn to_f16(pixels: RawPixels) -> Vec<f16> {
    match pixels {
        RawPixels::Rgba8(d) => d
            .iter()
            .map(|&v| f16::from_f32(v as f32 / 255.0))
            .collect(),
        RawPixels::Rgba16(d) => d
            .iter()
            .map(|&v| f16::from_f32(v as f32 / 65535.0))
            .collect(),
        RawPixels::RgbaF32(d) => d.iter().map(|&v| f16::from_f32(v)).collect(),
    }
}

/// Premultiply straight alpha in place (electrical values — the Wayland
/// convention without wp_color_representation is premultiplied-electrical).
/// Returns whether any pixel was actually translucent.
fn premultiply_u8(data: &mut [u8]) -> bool {
    let mut has_alpha = false;
    for px in data.chunks_exact_mut(4) {
        let a = px[3];
        if a == 255 {
            continue;
        }
        has_alpha = true;
        let a16 = a as u16;
        px[0] = ((px[0] as u16 * a16 + 127) / 255) as u8;
        px[1] = ((px[1] as u16 * a16 + 127) / 255) as u8;
        px[2] = ((px[2] as u16 * a16 + 127) / 255) as u8;
    }
    has_alpha
}

fn premultiply_f16(data: &mut [f16]) -> bool {
    let mut has_alpha = false;
    for px in data.chunks_exact_mut(4) {
        let a = px[3].to_f32();
        if a >= 1.0 {
            continue;
        }
        has_alpha = true;
        px[0] = f16::from_f32(px[0].to_f32() * a);
        px[1] = f16::from_f32(px[1].to_f32() * a);
        px[2] = f16::from_f32(px[2].to_f32() * a);
    }
    has_alpha
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{PrimaryVolume, Tf};

    /// Synthesize an in-memory PNG with the given chunk setup.
    fn make_png(set_info: impl FnOnce(&mut ::png::Encoder<&mut Vec<u8>>)) -> Vec<u8> {
        let mut out = Vec::new();
        let mut enc = ::png::Encoder::new(&mut out, 2, 2);
        enc.set_color(::png::ColorType::Rgba);
        enc.set_depth(::png::BitDepth::Eight);
        set_info(&mut enc);
        let mut writer = enc.write_header().unwrap();
        writer
            .write_image_data(&[255u8; 16])
            .unwrap();
        drop(writer);
        out
    }

    #[test]
    fn untagged_png_is_srgb_8bit() {
        let img = finish(decode(&make_png(|_| {})).unwrap()).unwrap();
        assert_eq!(img.encoding, crate::color::ColorEncoding::SRGB);
        assert!(matches!(img.pixels, Pixels::Rgba8(_)));
        assert!(!img.has_alpha);
    }

    #[test]
    fn png_chrm_gama_yields_display_p3() {
        let data = make_png(|enc| {
            enc.set_source_gamma(::png::ScaledFloat::new(1.0 / 2.2));
            enc.set_source_chromaticities(::png::SourceChromaticities::new(
                (0.3127, 0.3290),
                (0.680, 0.320),
                (0.265, 0.690),
                (0.150, 0.060),
            ));
        });
        let img = finish(decode(&data).unwrap()).unwrap();
        assert_eq!(img.encoding.tf, Tf::Gamma22);
        assert_eq!(img.encoding.primaries, PrimaryVolume::DisplayP3);
    }

    #[test]
    fn premultiply_is_electrical_and_flags_alpha() {
        let mut px = vec![200u8, 100, 50, 128];
        assert!(premultiply_u8(&mut px));
        assert_eq!(&px, &[100, 50, 25, 128]);
        let mut opaque = vec![200u8, 100, 50, 255];
        assert!(!premultiply_u8(&mut opaque));
        assert_eq!(&opaque, &[200, 100, 50, 255]);
    }

    /// Hand-rolled minimal AVIF box structure: meta(fullbox){iprp{ipco{colr}}}.
    fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn avif_colr_nclx_parses() {
        // nclx: primaries=9 (BT.2020), tf=16 (PQ), matrix=9, full-range bit.
        let mut nclx = b"nclx".to_vec();
        nclx.extend_from_slice(&9u16.to_be_bytes());
        nclx.extend_from_slice(&16u16.to_be_bytes());
        nclx.extend_from_slice(&9u16.to_be_bytes());
        nclx.push(0x80);
        let colr = boxed(b"colr", &nclx);
        let ipco = boxed(b"ipco", &colr);
        let iprp = boxed(b"iprp", &ipco);
        let mut meta_payload = vec![0u8; 4]; // FullBox version/flags
        meta_payload.extend_from_slice(&iprp);
        let mut file = boxed(b"ftyp", b"avifpayload");
        file.extend_from_slice(&boxed(b"meta", &meta_payload));

        match generic::test_parse_avif_colr(&file) {
            Some((9, 16, true)) => {}
            other => panic!("unexpected colr parse: {other:?}"),
        }
    }
}

#[cfg(test)]
mod jxr_probe {
    #[test]
    #[ignore]
    fn probe_file() {
        let path = std::env::var("PROBE_IMAGE").unwrap();
        let img = super::load(std::path::Path::new(&path)).unwrap();
        let (mut mn, mut mx, mut sum) = (f32::MAX, f32::MIN, 0f64);
        let mut n = 0u64;
        match &img.pixels {
            super::Pixels::Rgba8(d) => {
                for px in d.chunks_exact(4) {
                    for &c in &px[..3] {
                        let v = c as f32 / 255.0;
                        mn = mn.min(v); mx = mx.max(v); sum += v as f64; n += 1;
                    }
                }
            }
            super::Pixels::Rgba16(d) => {
                for px in d.chunks_exact(4) {
                    for &c in &px[..3] {
                        let v = c as f32 / 65535.0;
                        mn = mn.min(v); mx = mx.max(v); sum += v as f64; n += 1;
                    }
                }
            }
            super::Pixels::RgbaF16(d) => {
                for px in d.chunks_exact(4) {
                    for &c in &px[..3] {
                        let v = c.to_f32();
                        mn = mn.min(v); mx = mx.max(v); sum += v as f64; n += 1;
                    }
                }
            }
        }
        println!(
            "{}x{} encoding={:?} has_alpha={} rgb min={mn} max={mx} mean={}",
            img.width, img.height, img.encoding, img.has_alpha, sum / n as f64
        );
    }
}

#[cfg(test)]
mod quantize_tests {
    use super::*;
    use crate::color::{ColorEncoding, Luminances, PrimaryVolume, Tf};

    #[test]
    fn linear_fp16_quantizes_to_srgb_encoded_8bit() {
        // 0.5 linear → ~0.7354 sRGB-encoded → 188; 2.0 clips to 255.
        let d: Vec<f16> = [0.5f32, 0.0, 2.0, 1.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let img = DecodedImage {
            width: 1,
            height: 1,
            pixels: Pixels::RgbaF16(d),
            encoding: ColorEncoding {
                tf: Tf::Linear,
                primaries: PrimaryVolume::Srgb,
                luminances: Some(Luminances {
                    min: 0.0,
                    max: 10000.0,
                    reference: 80.0,
                }),
            },
            has_alpha: false,
        };
        let q = img.quantized_to_8bit(Tf::Srgb);
        assert_eq!(q.encoding.tf, Tf::Srgb);
        assert!(q.encoding.luminances.is_none());
        let Pixels::Rgba8(d) = &q.pixels else {
            panic!("expected 8-bit")
        };
        assert_eq!(&d[..], &[188, 0, 255, 255]);
    }

    #[test]
    fn display_referred_fp16_just_clamps() {
        let d: Vec<f16> = [0.5f32, 1.5, 0.25, 1.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let img = DecodedImage {
            width: 1,
            height: 1,
            pixels: Pixels::RgbaF16(d),
            encoding: ColorEncoding {
                tf: Tf::Pq,
                primaries: PrimaryVolume::Bt2020,
                luminances: None,
            },
            has_alpha: false,
        };
        let q = img.quantized_to_8bit(Tf::Srgb);
        assert_eq!(q.encoding.tf, Tf::Pq);
        let Pixels::Rgba8(d) = &q.pixels else {
            panic!("expected 8-bit")
        };
        assert_eq!(&d[..], &[128, 255, 64, 255]);
    }
}

#[cfg(test)]
mod reencode_tests {
    use super::*;
    use crate::color::{ColorEncoding, PrimaryVolume, Tf};

    fn srgb_image(pixels: Pixels) -> DecodedImage {
        DecodedImage {
            width: 1,
            height: 1,
            pixels,
            encoding: ColorEncoding::SRGB,
            has_alpha: false,
        }
    }

    #[test]
    fn srgb_to_gamma22_reencodes_opaque_8bit() {
        // sRGB 188 → linear ≈0.5029 → gamma2.2 ≈ 0.7316 → 187.
        let img = srgb_image(Pixels::Rgba8(vec![188, 0, 255, 255]));
        let out = img.reencoded_tf(Tf::Gamma22);
        assert_eq!(out.encoding.tf, Tf::Gamma22);
        let Pixels::Rgba8(d) = &out.pixels else { panic!() };
        assert_eq!(&d[..], &[187, 0, 255, 255]);
    }

    #[test]
    fn reencode_respects_premultiplied_alpha() {
        // Straight value 188 premultiplied by a=0.5 → 94 in the buffer.
        // Conversion must go through the straight value: 188→187, then
        // re-premultiply → round(187 * 128/255) = 94.
        let img = srgb_image(Pixels::Rgba8(vec![94, 0, 128, 128]));
        let out = img.reencoded_tf(Tf::Gamma22);
        let Pixels::Rgba8(d) = &out.pixels else { panic!() };
        // straight 94/(128/255)=187.3→ converted ≈186.5 → ×0.502 ≈ 94
        assert!((d[0] as i32 - 94).abs() <= 1, "got {}", d[0]);
        assert_eq!(d[3], 128);
    }

    #[test]
    fn gamma22_quantize_target() {
        // linear 0.5 → 0.5^(1/2.2) ≈ 0.7297 → 186
        let d: Vec<f16> = [0.5f32, 0.0, 1.0, 1.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let img = DecodedImage {
            width: 1,
            height: 1,
            pixels: Pixels::RgbaF16(d),
            encoding: ColorEncoding {
                tf: Tf::Linear,
                primaries: PrimaryVolume::Srgb,
                luminances: None,
            },
            has_alpha: false,
        };
        let q = img.quantized_to_8bit(Tf::Gamma22);
        assert_eq!(q.encoding.tf, Tf::Gamma22);
        let Pixels::Rgba8(d) = &q.pixels else { panic!() };
        assert_eq!(&d[..], &[186, 0, 255, 255]);
    }
}

#[cfg(test)]
mod repack_tests {
    use super::*;
    use crate::color::{pq_eotf, ColorEncoding, Luminances, PrimaryVolume, Tf};

    fn scrgb(vals: [f32; 4]) -> DecodedImage {
        DecodedImage {
            width: 1,
            height: 1,
            pixels: Pixels::RgbaF16(vals.iter().map(|&v| f16::from_f32(v)).collect()),
            encoding: ColorEncoding {
                tf: Tf::Linear,
                primaries: PrimaryVolume::Srgb,
                luminances: Some(Luminances {
                    min: 0.0,
                    max: 10000.0,
                    reference: 80.0,
                }),
            },
            has_alpha: false,
        }
    }

    #[test]
    fn linear_with_pq_keeps_absolute_luminance() {
        // scRGB 1.0 = 80 nits; 5.0 = 400 nits. PQ round-trip must recover
        // the nits (16-bit unorm + PQ is ~perceptually lossless).
        let img = scrgb([1.0, 5.0, 0.0, 1.0]);
        let out = img.repacked_unorm16(true, Tf::Gamma22);
        assert_eq!(out.encoding.tf, Tf::Pq);
        assert_eq!(out.encoding.primaries, PrimaryVolume::Srgb);
        let Pixels::Rgba16(d) = &out.pixels else { panic!() };
        let nits = |v: u16| pq_eotf(v as f32 / 65535.0) * 10000.0;
        assert!((nits(d[0]) - 80.0).abs() < 0.1, "got {}", nits(d[0]));
        assert!((nits(d[1]) - 400.0).abs() < 0.5, "got {}", nits(d[1]));
        assert_eq!(d[2], 0);
        assert_eq!(d[3], 65535);
    }

    #[test]
    fn linear_without_pq_clips_to_sdr() {
        // 0.5 linear → gamma2.2-encoded; 5.0 clips to 1.0.
        let img = scrgb([0.5, 5.0, 0.0, 1.0]);
        let out = img.repacked_unorm16(false, Tf::Gamma22);
        assert_eq!(out.encoding.tf, Tf::Gamma22);
        let Pixels::Rgba16(d) = &out.pixels else { panic!() };
        let expect = (0.5f32.powf(1.0 / 2.2) * 65535.0 + 0.5) as u16;
        assert_eq!(d[0], expect);
        assert_eq!(d[1], 65535);
    }

    #[test]
    fn electrical_fp16_requantizes_verbatim() {
        // PQ-encoded fp16 (HDR AVIF/JXL path): signal is [0,1] electrical,
        // repack must not touch the values or the tag.
        let img = DecodedImage {
            width: 1,
            height: 1,
            pixels: Pixels::RgbaF16(
                [0.25f32, 0.5, 0.75, 1.0]
                    .iter()
                    .map(|&v| f16::from_f32(v))
                    .collect(),
            ),
            encoding: ColorEncoding {
                tf: Tf::Pq,
                primaries: PrimaryVolume::Bt2020,
                luminances: None,
            },
            has_alpha: false,
        };
        let out = img.repacked_unorm16(true, Tf::Gamma22);
        assert_eq!(out.encoding.tf, Tf::Pq);
        let Pixels::Rgba16(d) = &out.pixels else { panic!() };
        assert_eq!(d[0], (0.25 * 65535.0 + 0.5) as u16);
        assert_eq!(d[1], (0.5 * 65535.0 + 0.5) as u16);
        assert_eq!(d[3], 65535);
    }

    #[test]
    fn pq_oetf_eotf_roundtrip() {
        for nits in [0.0f32, 0.1, 1.0, 80.0, 203.0, 1000.0, 10000.0] {
            let e = crate::color::pq_oetf(nits / 10000.0);
            let back = pq_eotf(e) * 10000.0;
            assert!(
                (back - nits).abs() < nits.max(1.0) * 1e-3,
                "{nits} -> {e} -> {back}"
            );
        }
    }
}
