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
mod playlist;
mod surfaces;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use smithay_client_toolkit::reexports::{
    calloop::{
        timer::{TimeoutAction, Timer},
        EventLoop,
    },
    calloop_wayland_source::WaylandSource,
};
use wayland_client::{globals::registry_queue_init, Connection};

use app::App;

/// Decode `path` into the raw-image cache (no-op if already present),
/// logging the source characteristics.
fn load_raw(
    raw_images: &mut HashMap<std::path::PathBuf, Arc<decode::DecodedImage>>,
    path: &std::path::Path,
) -> Result<()> {
    if raw_images.contains_key(path) {
        return Ok(());
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
    raw_images.insert(path.to_path_buf(), img);
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "prism_bg=info".into()),
        )
        .init();

    let mut args = cli::parse(std::env::args().skip(1))?;

    // Load playlists (--image-list): one rotation state per spec group.
    // Entries decode lazily at rotation time (App::rotate skips broken
    // files); only the initial entry is decoded fail-fast below.
    let mut playlists: Vec<playlist::Playlist> = Vec::new();
    for spec in &mut args.specs {
        let Some(list) = &spec.image_list else {
            continue;
        };
        let pl = playlist::Playlist::load(
            list,
            spec.rotate_every.unwrap_or(cli::DEFAULT_ROTATE_EVERY),
            spec.randomize,
        )?;
        tracing::info!(
            list = %list.display(),
            entries = pl.len(),
            period = ?pl.period,
            randomize = spec.randomize,
            "playlist loaded"
        );
        spec.playlist = Some(playlists.len());
        playlists.push(pl);
    }

    // Decode every initially-referenced image up front, before touching
    // the display — a bad `-i` file should fail fast. Luminance treatment
    // and capability adaptation happen lazily per output (App::service):
    // `--tone-map auto` targets only resolve once the compositor tells us
    // about each output, and the same file may need different pixels per
    // output.
    let mut raw_images: HashMap<std::path::PathBuf, Arc<decode::DecodedImage>> = HashMap::new();
    for path in args.specs.iter().filter_map(|s| s.image.clone()) {
        load_raw(&mut raw_images, &path)?;
    }
    // Playlists tolerate broken entries (a deleted file must not kill the
    // daemon — rotation skips them too): seek each list to its first
    // decodable entry, failing only if none decodes.
    for (i, pl) in playlists.iter_mut().enumerate() {
        let loaded = (0..pl.len()).any(|_| match load_raw(&mut raw_images, pl.current()) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(
                    path = %pl.current().display(),
                    "skipping playlist entry: {e:#}"
                );
                pl.advance();
                false
            }
        });
        if !loaded {
            let list = args.specs.iter().find(|s| s.playlist == Some(i));
            bail!(
                "image list {} has no loadable entries",
                list.and_then(|s| s.image_list.as_deref())
                    .unwrap_or(std::path::Path::new("?"))
                    .display()
            );
        }
    }

    let conn = Connection::connect_to_env().context("connecting to Wayland display")?;
    let (globals, mut queue) = registry_queue_init::<App>(&conn)?;
    let qh = queue.handle();

    let mut app = App::new(&globals, &qh, &args, raw_images, playlists)?;

    // Setup roundtrips: output enumeration, wl_shm formats, and the color
    // manager's supported_* events (terminated by done). Image treatment
    // waits on these (capability adaptation needs the caps).
    queue.roundtrip(&mut app).context("initial roundtrip")?;
    if app.color.as_ref().is_some_and(|c| !c.done) {
        queue
            .roundtrip(&mut app)
            .context("waiting for color manager caps")?;
    }

    // Wallpapers for the outputs we already know about (hotplug arrives
    // via new_output). Images attach via service() as their tone targets
    // and descriptions resolve.
    for output in app.output_state.outputs().collect::<Vec<_>>() {
        app.add_output(&qh, output);
    }
    app.service(&qh);

    // Event loop: the wayland socket plus one rotation timer per
    // playlist. service() runs after every wake-up (cheap when nothing is
    // pending), same as the old dispatch loop.
    let mut event_loop: EventLoop<App> = EventLoop::try_new().context("creating event loop")?;
    WaylandSource::new(conn, queue)
        .insert(event_loop.handle())
        .map_err(|e| anyhow!("inserting wayland source: {e}"))?;
    for idx in 0..app.playlists.len() {
        let period = app.playlists[idx].period;
        let qh = qh.clone();
        event_loop
            .handle()
            .insert_source(Timer::from_duration(period), move |_, _, app: &mut App| {
                app.rotate(&qh, idx);
                TimeoutAction::ToDuration(period)
            })
            .map_err(|e| anyhow!("inserting rotation timer: {e}"))?;
    }
    event_loop
        .run(None, &mut app, |app| app.service(&qh))
        .context("event loop")?;
    bail!("event loop exited unexpectedly");
}
