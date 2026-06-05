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

use app::App;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "prism_bg=info".into()),
        )
        .init();

    let args = cli::parse(std::env::args().skip(1))?;

    // Decode every referenced image up front, before touching the display —
    // a bad file should fail fast. Luminance treatment and capability
    // adaptation happen lazily per output (App::service): `--tone-map
    // auto` targets only resolve once the compositor tells us about each
    // output, and the same file may need different pixels per output.
    let mut raw_images: HashMap<std::path::PathBuf, Arc<decode::DecodedImage>> = HashMap::new();
    for spec in &args.specs {
        let Some(path) = &spec.image else { continue };
        if raw_images.contains_key(path) {
            continue;
        }
        let img = Arc::new(decode::load(path)?);
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
        raw_images.insert(path.clone(), img);
    }

    let conn = Connection::connect_to_env().context("connecting to Wayland display")?;
    let (globals, mut queue) = registry_queue_init::<App>(&conn)?;
    let qh = queue.handle();

    let mut app = App::new(&globals, &qh, &args, raw_images)?;

    // Setup roundtrips: output enumeration, wl_shm formats, and the color
    // manager's supported_* events (terminated by done). Image treatment
    // waits on these (capability adaptation needs the caps).
    queue.roundtrip(&mut app).context("initial roundtrip")?;
    if app.color.as_ref().is_some_and(|c| !c.done) {
        queue.roundtrip(&mut app).context("waiting for color manager caps")?;
    }

    // Wallpapers for the outputs we already know about (hotplug arrives
    // via new_output). Images attach via service() as their tone targets
    // and descriptions resolve.
    for output in app.output_state.outputs().collect::<Vec<_>>() {
        app.add_output(&qh, output);
    }
    app.service(&qh);

    loop {
        queue.blocking_dispatch(&mut app).context("event loop")?;
        app.service(&qh);
    }
}
