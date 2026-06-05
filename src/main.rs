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

/// Adapt loaded images to what the compositor can actually take:
/// - without fp16 shm (`Abgr16161616f`), quantize to 8-bit;
/// - when an image's TF isn't in the compositor's named-TF vocabulary,
///   re-encode the pixels to one that is. KWin notably dropped the
///   protocol-deprecated `srgb`, so plain sRGB images re-encode to
///   gamma 2.2 there; linear sources clip at reference white when
///   `ext_linear` is missing. PQ is never converted — failing loudly
///   beats silently tone-mapping HDR.
fn adapt_images_to_caps(app: &mut App) -> Result<()> {
    use crate::color::Tf;
    use smithay_client_toolkit::shm::ShmHandler as _;

    let fp16_ok = app
        .shm_state()
        .formats()
        .contains(&wayland_client::protocol::wl_shm::Format::Abgr16161616f);

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
        if !fp16_ok && matches!(loaded.image.pixels, decode::Pixels::RgbaF16(_)) {
            let target = sdr_tf.context("compositor supports no display-referred TF")?;
            tracing::warn!(
                path = %path.display(),
                ?target,
                "compositor lacks fp16 shm buffers; quantizing to 8-bit"
            );
            loaded.image = Arc::new(loaded.image.quantized_to_8bit(target));
        }
        let Some(color) = &app.color else { continue };
        let tf = loaded.image.encoding.tf;
        if color.supports_tf(tf) {
            continue;
        }
        match tf {
            Tf::Srgb | Tf::Gamma22 | Tf::Bt1886 | Tf::Linear => {
                let target = sdr_tf.context("compositor supports no display-referred TF")?;
                if tf == Tf::Linear {
                    tracing::warn!(
                        path = %path.display(),
                        ?target,
                        "compositor lacks ext_linear; re-encoding (HDR clips at reference white)"
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
    Ok(())
}
