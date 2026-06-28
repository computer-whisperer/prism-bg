//! Wallpaper playlists (`--image-list`): a text file with one entry per
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
        let mut order: Vec<usize> = (0..entries.len()).collect();
        if randomize {
            shuffle(&mut order);
        }
        Ok(Playlist {
            entries,
            order,
            pos: 0,
            period,
            randomize,
        })
    }

    pub fn current(&self) -> &Source {
        &self.entries[self.order[self.pos]]
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
