//! Hot-reload for `--list` playlist files: a debounced filesystem watch that
//! signals the event loop whenever a watched list is edited, so it can reload
//! and re-roll the wallpaper. Editors save via atomic rename and multi-write
//! bursts; the debouncer coalesces those into a single notification, and we
//! watch each list's *parent directory* (not the file inode) so a
//! rename-over-original is still caught.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use notify_debouncer_mini::{
    new_debouncer,
    notify::{RecommendedWatcher, RecursiveMode},
    DebounceEventResult, Debouncer,
};
use smithay_client_toolkit::reexports::calloop;

/// Debounce window: long enough to coalesce an editor's save burst, short
/// enough that a manual edit feels immediate.
const DEBOUNCE: Duration = Duration::from_millis(250);

/// Start watching every distinct `--list` file in `lists`. Returns the live
/// [`Debouncer`] — which the caller must keep alive, as dropping it stops the
/// watch — and a calloop channel that yields the path of a list each time it
/// changes (matching the path as it appeared in `lists`).
pub fn watch_lists(
    lists: &[PathBuf],
) -> Result<(
    Debouncer<RecommendedWatcher>,
    calloop::channel::Channel<PathBuf>,
)> {
    let (tx, rx) = calloop::channel::channel::<PathBuf>();

    // Map each watched file — resolved to the canonical full path inotify will
    // report (canonical dir + filename) — back to the list path we hand the
    // loop. Each parent directory is watched once.
    let mut by_full: HashMap<PathBuf, PathBuf> = HashMap::new();
    let mut dirs: Vec<PathBuf> = Vec::new();
    // Last modification time we forwarded for each list, seeded with the
    // file's current mtime. We watch each list's *parent directory*, so our
    // own reload — which reads the list file — produces open/access/close
    // events that the watch reports back. The debouncer can't tell a read
    // from a write, and forwarding them would loop forever. Reads leave mtime
    // untouched, so we only forward when it actually advances.
    let mut last_mtime: HashMap<PathBuf, SystemTime> = HashMap::new();
    for list in lists {
        let parent = list.parent().filter(|p| !p.as_os_str().is_empty());
        let parent = parent.unwrap_or(Path::new("."));
        let canon_parent = std::fs::canonicalize(parent)
            .with_context(|| format!("resolving list directory {}", parent.display()))?;
        let Some(name) = list.file_name() else {
            continue;
        };
        by_full.insert(canon_parent.join(name), list.clone());
        if let Ok(mtime) = std::fs::metadata(list).and_then(|m| m.modified()) {
            last_mtime.insert(list.clone(), mtime);
        }
        if !dirs.contains(&canon_parent) {
            dirs.push(canon_parent);
        }
    }

    let mut debouncer = new_debouncer(DEBOUNCE, move |res: DebounceEventResult| {
        let events = match res {
            Ok(events) => events,
            Err(e) => {
                tracing::warn!("list watch error: {e:?}");
                return;
            }
        };
        // A single edit can touch the file more than once; decide each changed
        // list at most once per batch.
        let mut seen: Vec<&PathBuf> = Vec::new();
        for ev in &events {
            let Some(list) = by_full.get(&ev.path) else {
                continue;
            };
            if seen.contains(&list) {
                continue;
            }
            seen.push(list);
            // Skip self-induced read events: only forward when mtime advanced
            // past the last value we forwarded (see `last_mtime` above).
            if let Ok(mtime) = std::fs::metadata(list).and_then(|m| m.modified()) {
                if last_mtime.get(list) == Some(&mtime) {
                    continue;
                }
                last_mtime.insert(list.clone(), mtime);
            }
            let _ = tx.send(list.clone());
        }
    })
    .context("creating list watcher")?;

    for dir in &dirs {
        debouncer
            .watcher()
            .watch(dir, RecursiveMode::NonRecursive)
            .with_context(|| format!("watching list directory {}", dir.display()))?;
    }
    Ok((debouncer, rx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use smithay_client_toolkit::reexports::calloop::EventLoop;
    use std::io::Write;

    /// Live smoke test (filesystem + debounce timing, hence ignored): editing a
    /// watched list signals the calloop channel with that list's path, and a
    /// subsequent *read* of the file (as our own reload does) does not — the
    /// mtime guard that breaks the reload→read→reload loop.
    #[test]
    #[ignore]
    fn edit_signals_the_loop() {
        let dir = std::env::temp_dir().join(format!("prism-bg-watch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let list = dir.join("list.txt");
        std::fs::write(&list, "a.png\n").unwrap();

        let (debouncer, rx) = watch_lists(std::slice::from_ref(&list)).unwrap();
        let mut ev: EventLoop<Option<PathBuf>> = EventLoop::try_new().unwrap();
        ev.handle()
            .insert_source(rx, |event, _, got: &mut Option<PathBuf>| {
                if let calloop::channel::Event::Msg(p) = event {
                    *got = Some(p);
                }
            })
            .unwrap();

        let edited = list.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&edited)
                .unwrap();
            writeln!(f, "b.png").unwrap();
        });

        let mut got = None;
        for _ in 0..20 {
            ev.dispatch(Duration::from_millis(100), &mut got).unwrap();
            if got.is_some() {
                break;
            }
        }
        assert_eq!(got.as_deref(), Some(list.as_path()), "edit should signal");

        // Reading the file (what reload does) must not re-signal: a read leaves
        // mtime untouched, so the watch's self-induced events are dropped.
        got = None;
        let _ = std::fs::read_to_string(&list).unwrap();
        for _ in 0..10 {
            ev.dispatch(Duration::from_millis(100), &mut got).unwrap();
        }
        drop(debouncer);
        assert_eq!(got, None, "a pure read must not re-trigger the watch");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
