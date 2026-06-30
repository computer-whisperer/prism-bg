//! Wallpaper playlists (`--list`): a text file with one entry per
//! line, stepped through on a timer. Each entry is either a still image or
//! a live GLSL shader (`.frag`/`.glsl`), classified by extension — a single
//! list may interleave both.
//!
//! One `Playlist` per CLI output group that asked for one — every output
//! the group matches shows the same entry and advances in lockstep, so
//! each image is decoded once regardless of monitor count. Entries decode
//! lazily at rotation time (see `App::rotate`); only each list's first
//! entry is decoded fail-fast at startup.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};

use crate::cli::Luminance;

/// Scan a shader's source for its `//!luminance dark|bright` self-declaration.
/// `None` if absent (the entry is then eligible at any time of day).
pub fn classify_shader_source(source: &str) -> Option<Luminance> {
    for line in source.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("//!luminance") {
            return match rest.trim() {
                "dark" => Some(Luminance::Dark),
                "bright" => Some(Luminance::Bright),
                _ => None,
            };
        }
    }
    None
}

/// Whether an entry of class `class` may show when `desired` luminance is
/// preferred. Unclassified entries (`None`) are always eligible.
fn eligible(class: Option<Luminance>, desired: Luminance) -> bool {
    class.is_none_or(|c| c == desired)
}

/// A playlist entry: a still image or a live GLSL shader. The kind is
/// decided from the file extension when the list is parsed, so the
/// presentation code can route each entry to the right pipeline without
/// re-sniffing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    Image(PathBuf),
    Shader(PathBuf),
}

impl Source {
    /// Classify a path by extension: `.frag`/`.glsl` (case-insensitive) are
    /// shaders, anything else is an image (the decoder sniffs the actual
    /// format from the file's bytes).
    pub fn from_path(path: PathBuf) -> Source {
        let is_shader = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("frag") || e.eq_ignore_ascii_case("glsl"));
        if is_shader {
            Source::Shader(path)
        } else {
            Source::Image(path)
        }
    }

    pub fn path(&self) -> &Path {
        match self {
            Source::Image(p) | Source::Shader(p) => p,
        }
    }

    pub fn is_shader(&self) -> bool {
        matches!(self, Source::Shader(_))
    }
}

pub struct Playlist {
    entries: Vec<Source>,
    /// Average-luminance class per entry, parallel to `entries`, for
    /// `--dark-hours` filtering. Shaders are classified from their
    /// `//!luminance` directive at load; images are filled in lazily once
    /// decoded (`App` calls [`Self::set_class`]). `None` = eligible at any time.
    classes: Vec<Option<Luminance>>,
    /// Presentation order, indices into `entries`; identity unless
    /// `randomize`.
    order: Vec<usize>,
    pos: usize,
    pub period: Duration,
    randomize: bool,
}

impl Playlist {
    /// Parse the list file. Blank lines and `#` comments are skipped;
    /// relative entries resolve against the list file's directory; a
    /// leading `~/` expands to `$HOME`.
    pub fn load(list: &Path, period: Duration, randomize: bool) -> Result<Playlist> {
        let text = std::fs::read_to_string(list)
            .with_context(|| format!("reading image list {}", list.display()))?;
        let dir = list.parent().unwrap_or(Path::new("."));
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let mut entries = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let path = match (line.strip_prefix("~/"), &home) {
                (Some(rest), Some(home)) => home.join(rest),
                // join() ignores `dir` when `line` is absolute.
                _ => dir.join(line),
            };
            entries.push(Source::from_path(path));
        }
        if entries.is_empty() {
            bail!("image list {} contains no entries", list.display());
        }
        // Classify shaders up front (cheap text scan; a missing/unreadable file
        // is left unclassified — the display path reports the real error).
        // Images are classified lazily once decoded.
        let classes = entries
            .iter()
            .map(|src| match src {
                Source::Shader(p) => std::fs::read_to_string(p)
                    .ok()
                    .and_then(|s| classify_shader_source(&s)),
                Source::Image(_) => None,
            })
            .collect();
        let mut order: Vec<usize> = (0..entries.len()).collect();
        if randomize {
            shuffle(&mut order);
        }
        Ok(Playlist {
            entries,
            classes,
            order,
            pos: 0,
            period,
            randomize,
        })
    }

    pub fn current(&self) -> &Source {
        &self.entries[self.order[self.pos]]
    }

    /// The current entry's luminance class (`None` if unknown / unclassified).
    pub fn current_class(&self) -> Option<Luminance> {
        self.classes[self.order[self.pos]]
    }

    /// Record an image entry's class once it has been decoded and classified.
    /// Matched by path (an entry can appear more than once), so the caller
    /// needn't know presentation order.
    pub fn set_class(&mut self, path: &Path, class: Luminance) {
        for (i, src) in self.entries.iter().enumerate() {
            if src.path() == path {
                self.classes[i] = Some(class);
            }
        }
    }

    /// Copy known luminance classes from `old` onto matching entries (by path).
    /// Used on hot-reload so entries that survived the edit keep their
    /// classification instead of being re-decoded to re-learn it.
    pub fn inherit_classes(&mut self, old: &Playlist) {
        for (src, &class) in old.entries.iter().zip(&old.classes) {
            if let Some(c) = class {
                self.set_class(src.path(), c);
            }
        }
    }

    /// Re-seat the cursor on an entry eligible for `desired`, preferring one
    /// whose path differs from `avoid` (the wallpaper currently on screen) so a
    /// re-roll visibly changes something. Falls back to the first eligible entry
    /// even if it equals `avoid` (e.g. a single-entry pool), and leaves the
    /// position untouched when nothing is eligible.
    pub fn reseat(&mut self, desired: Option<Luminance>, avoid: Option<&Path>) {
        let n = self.order.len();
        let mut first_eligible = None;
        for step in 0..n {
            let p = (self.pos + step) % n;
            let entry = self.order[p];
            if desired.is_some_and(|d| !eligible(self.classes[entry], d)) {
                continue;
            }
            if first_eligible.is_none() {
                first_eligible = Some(p);
            }
            if avoid.is_none_or(|a| self.entries[entry].path() != a) {
                self.pos = p;
                return;
            }
        }
        if let Some(p) = first_eligible {
            self.pos = p;
        }
    }

    /// Move to the first entry from the current position (inclusive) that is
    /// eligible for `desired`, without reshuffling. No-op when filtering is off
    /// (`desired` is `None`) or nothing is eligible (leaves the position as-is,
    /// so the caller still shows *something*).
    pub fn seek_eligible(&mut self, desired: Option<Luminance>) {
        let Some(d) = desired else { return };
        let n = self.order.len();
        for step in 0..n {
            let p = (self.pos + step) % n;
            if eligible(self.classes[self.order[p]], d) {
                self.pos = p;
                return;
            }
        }
    }

    /// Whether the current entry is eligible for `desired` (always true when
    /// `desired` is `None`).
    pub fn current_eligible(&self, desired: Option<Luminance>) -> bool {
        desired.is_none_or(|d| eligible(self.current_class(), d))
    }

    /// Whether *any* entry could satisfy `desired` — one already known to match,
    /// or one still unclassified (which might, once decoded). Guards the re-roll
    /// in [`crate::app::App::on_image_prepared`]: with no eligible alternative we
    /// accept a violating wallpaper rather than spin forever looking for one.
    pub fn has_eligible(&self, desired: Luminance) -> bool {
        self.classes.iter().any(|&c| eligible(c, desired))
    }

    /// Step to the next entry, wrapping. A randomized list reshuffles each
    /// pass, avoiding an immediate repeat of the last shown image.
    pub fn advance(&mut self) {
        self.pos += 1;
        if self.pos == self.order.len() {
            self.pos = 0;
            if self.randomize && self.order.len() > 1 {
                let last = self.order[self.order.len() - 1];
                while {
                    shuffle(&mut self.order);
                    self.order[0] == last
                } {}
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Fisher–Yates.
fn shuffle(v: &mut [usize]) {
    for i in (1..v.len()).rev() {
        v.swap(i, fastrand::usize(..=i));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn write_list(name: &str, contents: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("prism-bg-test-{}-{name}", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn classifies_shader_directives() {
        assert_eq!(
            classify_shader_source("//!luminance dark\n#version 450\n"),
            Some(Luminance::Dark)
        );
        assert_eq!(
            classify_shader_source("  //!luminance bright\nvoid main(){}"),
            Some(Luminance::Bright)
        );
        assert_eq!(classify_shader_source("void main(){}"), None);
        // Unknown value is ignored, not an error.
        assert_eq!(classify_shader_source("//!luminance teal"), None);
    }

    #[test]
    fn playlist_filters_by_luminance() {
        let dark = write_list("dk.frag", "//!luminance dark\nvoid main(){}");
        let bright = write_list("br.frag", "//!luminance bright\nvoid main(){}");
        let plain = write_list("pl.frag", "void main(){}");
        let list = write_list(
            "lum.txt",
            &format!(
                "{}\n{}\n{}\na.png\n",
                dark.display(),
                bright.display(),
                plain.display()
            ),
        );
        let mut pl = Playlist::load(&list, Duration::from_secs(60), false).unwrap();
        // Identity order: dark, bright, plain(None), image(None).
        assert_eq!(pl.current_class(), Some(Luminance::Dark));
        // Seeking bright skips the dark entry to the bright one.
        pl.seek_eligible(Some(Luminance::Bright));
        assert_eq!(pl.current_class(), Some(Luminance::Bright));
        // From there, seeking dark lands on the unclassified plain shader
        // (None is eligible for any preference) before wrapping to the dark one.
        pl.seek_eligible(Some(Luminance::Dark));
        assert_eq!(pl.current_class(), None);
        assert!(pl.current_eligible(Some(Luminance::Dark)));
        // No filter → every entry is eligible.
        assert!(pl.current_eligible(None));
        // An unclassified entry (the image) keeps both classes "available".
        assert!(pl.has_eligible(Luminance::Dark));
        assert!(pl.has_eligible(Luminance::Bright));
        for f in [&dark, &bright, &plain, &list] {
            let _ = std::fs::remove_file(f);
        }
    }

    #[test]
    fn has_eligible_false_only_when_all_known_other_class() {
        let bright = write_list("brt.frag", "//!luminance bright\nvoid main(){}");
        let list = write_list("allbright.txt", &format!("{0}\n{0}\n", bright.display()));
        let pl = Playlist::load(&list, Duration::from_secs(60), false).unwrap();
        // Every entry is known-bright → nothing eligible for a dark preference,
        // so the re-roll guard yields to the fallback instead of looping.
        assert!(!pl.has_eligible(Luminance::Dark));
        assert!(pl.has_eligible(Luminance::Bright));
        for f in [&bright, &list] {
            let _ = std::fs::remove_file(f);
        }
    }

    #[test]
    fn reseat_rerolls_off_current_entry() {
        let list = write_list("reseat.txt", "a.png\nb.png\nc.png\n");
        let mut pl = Playlist::load(&list, Duration::from_secs(60), false).unwrap();
        let cur = pl.current().path().to_path_buf();
        // Re-roll avoiding the on-screen entry lands on a different one.
        pl.reseat(None, Some(&cur));
        assert_ne!(pl.current().path(), cur.as_path());
        // Single-entry pool: no alternative exists, so it keeps the only entry
        // rather than leaving nothing on screen.
        let solo = write_list("solo.txt", "only.png\n");
        let mut sp = Playlist::load(&solo, Duration::from_secs(60), false).unwrap();
        let only = sp.current().path().to_path_buf();
        sp.reseat(None, Some(&only));
        assert_eq!(sp.current().path(), only.as_path());
        for f in [&list, &solo] {
            let _ = std::fs::remove_file(f);
        }
    }

    #[test]
    fn reseat_honors_luminance_filter() {
        let dark = write_list("d2.frag", "//!luminance dark\nvoid main(){}");
        let bright = write_list("b2.frag", "//!luminance bright\nvoid main(){}");
        let list = write_list(
            "rl.txt",
            &format!("{}\n{}\n", bright.display(), dark.display()),
        );
        let mut pl = Playlist::load(&list, Duration::from_secs(60), false).unwrap();
        assert_eq!(pl.current_class(), Some(Luminance::Bright));
        // A dark preference seats past the bright entry onto the dark one.
        pl.reseat(Some(Luminance::Dark), None);
        assert_eq!(pl.current_class(), Some(Luminance::Dark));
        for f in [&dark, &bright, &list] {
            let _ = std::fs::remove_file(f);
        }
    }

    #[test]
    fn inherit_classes_carries_known_classes() {
        let list = write_list("inh.txt", "x.png\ny.png\n");
        let mut old = Playlist::load(&list, Duration::from_secs(60), false).unwrap();
        let xpath = old.current().path().to_path_buf();
        old.set_class(&xpath, Luminance::Bright);
        // A freshly reloaded playlist starts unclassified; inheriting restores
        // what the prior load already learned.
        let mut fresh = Playlist::load(&list, Duration::from_secs(60), false).unwrap();
        assert_eq!(fresh.current_class(), None);
        fresh.inherit_classes(&old);
        assert_eq!(fresh.current_class(), Some(Luminance::Bright));
        let _ = std::fs::remove_file(&list);
    }

    #[test]
    fn parses_comments_blanks_and_relative_paths() {
        let list = write_list(
            "parse.txt",
            "# header\n\n  a.png  \n/abs/b.jpg\n\n# trailing\n",
        );
        let pl = Playlist::load(&list, Duration::from_secs(60), false).unwrap();
        assert_eq!(pl.len(), 2);
        assert_eq!(pl.current().path(), list.parent().unwrap().join("a.png"));
        let _ = std::fs::remove_file(&list);
    }

    #[test]
    fn entries_are_classified_by_extension() {
        let list = write_list("kinds.txt", "a.png\nb.frag\nc.JPG\nd.GLSL\n");
        let mut pl = Playlist::load(&list, Duration::from_secs(60), false).unwrap();
        assert!(matches!(pl.current(), Source::Image(_))); // a.png
        pl.advance();
        assert!(matches!(pl.current(), Source::Shader(_))); // b.frag
        pl.advance();
        assert!(matches!(pl.current(), Source::Image(_))); // c.JPG
        pl.advance();
        assert!(matches!(pl.current(), Source::Shader(_))); // d.GLSL (case-insensitive)
        let _ = std::fs::remove_file(&list);
    }

    #[test]
    fn empty_list_is_an_error() {
        let list = write_list("empty.txt", "# nothing\n\n");
        assert!(Playlist::load(&list, Duration::from_secs(60), false).is_err());
        let _ = std::fs::remove_file(&list);
    }

    #[test]
    fn sequential_advance_wraps() {
        let list = write_list("seq.txt", "a.png\nb.png\nc.png\n");
        let mut pl = Playlist::load(&list, Duration::from_secs(60), false).unwrap();
        let first = pl.current().clone();
        pl.advance();
        pl.advance();
        pl.advance();
        assert_eq!(pl.current(), &first);
        let _ = std::fs::remove_file(&list);
    }

    #[test]
    fn shuffle_covers_all_entries_without_immediate_repeat() {
        let list = write_list("rand.txt", "a.png\nb.png\nc.png\nd.png\n");
        let mut pl = Playlist::load(&list, Duration::from_secs(60), true).unwrap();
        for _ in 0..10 {
            let mut seen = HashSet::new();
            for _ in 0..pl.len() {
                let prev = pl.current().path().to_path_buf();
                seen.insert(prev.clone());
                pl.advance();
                assert_ne!(pl.current().path(), prev, "immediate repeat across shuffle");
            }
            assert_eq!(seen.len(), pl.len(), "a pass must cover every entry");
        }
        let _ = std::fs::remove_file(&list);
    }

    #[test]
    fn single_entry_list_advances_to_itself() {
        let list = write_list("single.txt", "only.png\n");
        let mut pl = Playlist::load(&list, Duration::from_secs(60), true).unwrap();
        let only = pl.current().path().to_path_buf();
        pl.advance();
        assert_eq!(pl.current().path(), only);
        let _ = std::fs::remove_file(&list);
    }
}
