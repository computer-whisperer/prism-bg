//! prism-bg — swaybg, in Rust, color-managed.
//!
//! A wallpaper client for prism (and any compositor with wlr-layer-shell,
//! wp_viewporter and wp_color_management_v1). Decodes the image's actual
//! color encoding (cICP, ICC, format conventions — HDR included), tags the
//! surface with a parametric image description, and lets the compositor's
//! calibrated pipeline do every conversion. No client-side resampling, no
//! silent sRGB assumption.

mod app;
mod cli;
mod cms;
mod color;
mod colormgmt;
mod decode;
mod surfaces;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use wayland_client::{globals::registry_queue_init, Connection};

use app::{App, LoadedImage};
use colormgmt::Status;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "prism_bg=info".into()),
        )
        .init();

    let args = cli::parse(std::env::args().skip(1))?;

    // Decode every referenced image up front (deduplicated by path), before
    // touching the display — a bad file should fail fast.
    let mut images: HashMap<_, _> = HashMap::new();
    for spec in &args.specs {
        let Some(path) = &spec.image else { continue };
        if images.contains_key(path) {
            continue;
        }
        let img = decode::load(path)?;
        tracing::info!(
            path = %path.display(),
            width = img.width,
            height = img.height,
            tf = ?img.encoding.tf,
            primaries = ?img.encoding.primaries,
            luminances = ?img.encoding.luminances,
            fp16 = matches!(img.pixels, decode::Pixels::RgbaF16(_)),
            "image loaded"
        );
        images.insert(
            path.clone(),
            LoadedImage {
                image: Arc::new(img),
                description: None,
            },
        );
    }

    let conn = Connection::connect_to_env().context("connecting to Wayland display")?;
    let (globals, mut queue) = registry_queue_init::<App>(&conn)?;
    let qh = queue.handle();

    let mut app = App::new(&globals, &qh, &args, images)?;

    // First roundtrip: output enumeration, wl_shm formats, and the color
    // manager's supported_* events (terminated by done).
    queue.roundtrip(&mut app).context("initial roundtrip")?;
    if app.color.as_ref().is_some_and(|c| !c.done) {
        queue.roundtrip(&mut app).context("waiting for color manager caps")?;
    }

    adapt_images_to_caps(&mut app)?;

    // Create image descriptions and wait for ready/failed. The protocol
    // forbids attaching a description before its ready event.
    if app.color.is_some() {
        let color = app.color.as_ref().unwrap();
        let mut descriptions = Vec::new();
        for (path, loaded) in &app.images {
            let desc = color
                .create_description(&qh, &loaded.image.encoding)
                .with_context(|| format!("describing {}", path.display()))?;
            descriptions.push((path.clone(), desc));
        }
        for (path, desc) in descriptions {
            app.images.get_mut(&path).unwrap().description = Some(desc);
        }
        while !app.descriptions_settled() {
            queue.roundtrip(&mut app).context("waiting for image descriptions")?;
        }
        for (path, loaded) in &app.images {
            if let Some(desc) = &loaded.description {
                if let Status::Failed(msg) = desc.status() {
                    bail!(
                        "compositor rejected image description for {}: {msg}",
                        path.display()
                    );
                }
            }
        }
    }

    // Descriptions are settled; create wallpapers for the outputs we
    // already know about. Hotplugged outputs arrive via new_output.
    app.bootstrapped = true;
    for output in app.output_state.outputs().collect::<Vec<_>>() {
        app.add_output(&qh, output);
    }

    loop {
        queue.blocking_dispatch(&mut app).context("event loop")?;
    }
}

/// Adapt loaded images to what the compositor can actually take.
///
/// Two independent axes, in order:
/// 1. **TF vocabulary** — when an image's TF isn't advertised, re-encode
///    the pixels to one that is (KWin dropped the protocol-deprecated
///    `srgb`, so plain sRGB re-encodes to gamma 2.2 there). Linear sources
///    clip at reference white when `ext_linear` is missing; PQ without
///    compositor PQ fails loudly rather than tone-mapping client-side.
/// 2. **Buffer container** — fp16 shm (`Abgr16161616f`) when advertised;
///    else 16-bit unorm (`Abgr16161616`), PQ-encoding linear HDR content
///    so the luminance range survives (the KWin path — real HDR, no
///    fp16); else quantize to 8-bit.
fn adapt_images_to_caps(app: &mut App) -> Result<()> {
    use crate::color::Tf;
    use smithay_client_toolkit::shm::ShmHandler as _;
    use wayland_client::protocol::wl_shm::Format;

    let formats = app.shm_state().formats().to_vec();
    // PRISM_BG_FORCE_NO_FP16=1 pretends fp16 shm is missing, to exercise
    // the unorm16/8-bit ladder on compositors that do support it.
    let fp16_ok = formats.contains(&Format::Abgr16161616f)
        && std::env::var_os("PRISM_BG_FORCE_NO_FP16").is_none();
    let unorm16_ok = formats.contains(&Format::Abgr16161616);

    // The display-referred TF we re-encode into when we must. Preference
    // order keeps pixels untouched on compositors that accept sRGB.
    let sdr_tf = match &app.color {
        Some(c) => [Tf::Srgb, Tf::Gamma22, Tf::Bt1886]
            .into_iter()
            .find(|&t| c.supports_tf(t)),
        // No color management: surfaces are untagged, compositor assumes
        // sRGB; nothing to negotiate.
        None => Some(Tf::Srgb),
    };

    for (path, loaded) in app.images.iter_mut() {
        // Axis 1: TF vocabulary (before any container change, while the
        // pixels still carry full precision).
        if let Some(color) = &app.color {
            let tf = loaded.image.encoding.tf;
            if !color.supports_tf(tf) {
                match tf {
                    Tf::Srgb | Tf::Gamma22 | Tf::Bt1886 | Tf::Linear => {
                        let target =
                            sdr_tf.context("compositor supports no display-referred TF")?;
                        if tf == Tf::Linear {
                            tracing::warn!(
                                path = %path.display(),
                                ?target,
                                "compositor lacks ext_linear; re-encoding (HDR clips at \
                                 reference white)"
                            );
                        } else {
                            tracing::info!(
                                path = %path.display(),
                                from = ?tf,
                                to = ?target,
                                "re-encoding to a TF the compositor supports"
                            );
                        }
                        loaded.image = Arc::new(loaded.image.reencoded_tf(target));
                    }
                    Tf::Pq => anyhow::bail!(
                        "{}: compositor does not support the PQ transfer function",
                        path.display()
                    ),
                }
            }
        }

        // Axis 2: buffer container.
        if fp16_ok || !matches!(loaded.image.pixels, decode::Pixels::RgbaF16(_)) {
            continue;
        }
        if unorm16_ok {
            let pq_ok = app
                .color
                .as_ref()
                .is_some_and(|c| c.supports_tf(Tf::Pq));
            let target = sdr_tf.context("compositor supports no display-referred TF")?;
            let hdr = loaded.image.encoding.tf == Tf::Linear;
            tracing::info!(
                path = %path.display(),
                pq = pq_ok && hdr,
                "compositor lacks fp16 shm; repacking as 16-bit unorm"
            );
            loaded.image = Arc::new(loaded.image.repacked_unorm16(pq_ok, target));
        } else {
            let target = sdr_tf.context("compositor supports no display-referred TF")?;
            tracing::warn!(
                path = %path.display(),
                ?target,
                "compositor lacks fp16 and 16-bit shm; quantizing to 8-bit"
            );
            loaded.image = Arc::new(loaded.image.quantized_to_8bit(target));
        }
    }
    Ok(())
}
