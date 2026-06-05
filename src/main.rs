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

    // First roundtrip: output enumeration + the color manager's
    // supported_* events (terminated by done).
    queue.roundtrip(&mut app).context("initial roundtrip")?;

    // Create image descriptions and wait for ready/failed. The protocol
    // forbids attaching a description before its ready event.
    if let Some(color) = &app.color {
        if !color.done {
            queue.roundtrip(&mut app).context("waiting for color manager caps")?;
        }
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
