//! swaybg-compatible command line.
//!
//! The flags are order-sensitive the same way swaybg's are: `-i`/`-m`/`-c`
//! apply to the most recent `-o`; flags before any `-o` configure the
//! default (`*`) spec, which applies to outputs no explicit spec matches.
//! Hand-parsed — clap can't express the positional grouping.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};

use crate::color::LuminanceControl;

/// Rotation period when `--list` is given without `--rotate-every`.
pub const DEFAULT_ROTATE_EVERY: Duration = Duration::from_secs(15 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Stretch,
    Fit,
    Fill,
    Center,
    Tile,
    SolidColor,
}

impl Mode {
    fn parse(s: &str) -> Result<Mode> {
        Ok(match s {
            "stretch" => Mode::Stretch,
            "fit" => Mode::Fit,
            "fill" => Mode::Fill,
            "center" => Mode::Center,
            "tile" => Mode::Tile,
            "solid_color" => Mode::SolidColor,
            _ => bail!("invalid mode {s:?} (expected stretch|fit|fill|center|tile|solid_color)"),
        })
    }
}

/// Rendering intent passed to `set_image_description` (prism honors
/// perceptual, relative and absolute).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    Perceptual,
    Relative,
    Absolute,
}

/// sRGB-encoded solid color, 0..=1 per channel.
#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    fn parse(s: &str) -> Result<Color> {
        let hex = s.strip_prefix('#').unwrap_or(s);
        if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!("invalid color {s:?} (expected [#]RRGGBB)");
        }
        let chan =
            |i: usize| -> f64 { u8::from_str_radix(&hex[i..i + 2], 16).unwrap() as f64 / 255.0 };
        Ok(Color {
            r: chan(0),
            g: chan(2),
            b: chan(4),
        })
    }
}

/// `--tone-map` argument: an explicit display peak, or "ask the
/// compositor" (resolved per output from the preferred image
/// description's target luminance before pixels are treated).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneMap {
    Nits(f64),
    Auto,
}

/// Per-output wallpaper spec. `output == "*"` is the fallback.
#[derive(Debug, Clone)]
pub struct OutputSpec {
    pub output: String,
    pub image: Option<PathBuf>,
    /// Playlist file (`--list`): one entry path per line, rotated
    /// on a timer. Entries may be images or `.frag`/`.glsl` shaders; a
    /// single list interleaves both. Mutually exclusive with `image`.
    pub image_list: Option<PathBuf>,
    /// Index into `App::playlists`, assigned by `main` after the list
    /// file is loaded. `None` until then (and always for `-i` specs).
    pub playlist: Option<usize>,
    /// GLSL fragment shader file (`--shader`): rendered on the GPU into an
    /// fp16 dmabuf and presented as a live wallpaper. Mutually exclusive
    /// with `image`/`image_list`.
    pub shader: Option<PathBuf>,
    /// `--fps`: cap an animated shader's render rate (frames/second). `None`
    /// renders at the compositor's vsync cadence. Ignored by static shaders.
    pub fps: Option<u32>,
    /// `--rotate-every`; `None` means [`DEFAULT_ROTATE_EVERY`].
    pub rotate_every: Option<Duration>,
    /// `--randomize`: shuffle the playlist order.
    pub randomize: bool,
    /// `--fade`: crossfade duration on rotation; `None` is a hard cut.
    pub fade: Option<Duration>,
    pub mode: Option<Mode>,
    pub color: Option<Color>,
    /// HDR luminance shaping (`--cap-luminance` / `--scale-luminance`).
    pub luminance: Option<LuminanceControl>,
    /// BT.2390 tone mapping (`--tone-map`); `Auto` resolves per output.
    pub tone_map: Option<ToneMap>,
}

impl OutputSpec {
    fn new(output: String) -> Self {
        OutputSpec {
            output,
            image: None,
            image_list: None,
            shader: None,
            fps: None,
            playlist: None,
            rotate_every: None,
            randomize: false,
            fade: None,
            mode: None,
            color: None,
            luminance: None,
            tone_map: None,
        }
    }

    /// Effective mode: explicit, else stretch with an image (swaybg's
    /// default), else solid color.
    pub fn effective_mode(&self) -> Mode {
        self.mode
            .unwrap_or(if self.image.is_some() || self.image_list.is_some() {
                Mode::Stretch
            } else {
                Mode::SolidColor
            })
    }
}

/// GPU render-time profiling. Off by default — the timestamp query pool is only
/// created when this is enabled, so normal operation pays nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileMode {
    Off,
    /// One averaged per-output report a few seconds after each shader (re)load,
    /// then quiet — for a 24/7 daemon that shouldn't spam logs.
    OnLoad,
    /// A per-output report every interval, continuously.
    Every(Duration),
}

impl ProfileMode {
    pub fn enabled(self) -> bool {
        !matches!(self, ProfileMode::Off)
    }
}

/// A wallpaper's rough average-luminance class, for time-of-day filtering.
/// Shaders self-declare it (`//!luminance dark|bright`); images are classified
/// from their mean luminance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Luminance {
    Dark,
    Bright,
}

/// Local-clock window during which playlists prefer dark wallpapers (and,
/// outside it, bright ones). Times are minutes since local midnight; a window
/// that wraps past midnight (start > end, e.g. 19:00–07:00) is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DarkHours {
    start_min: u32,
    end_min: u32,
}

impl DarkHours {
    /// Whether `minute` (0..1440, minutes since local midnight) is in the window.
    pub fn contains(self, minute: u32) -> bool {
        if self.start_min <= self.end_min {
            (self.start_min..self.end_min).contains(&minute)
        } else {
            minute >= self.start_min || minute < self.end_min
        }
    }

    /// The luminance class to prefer at `minute`: dark inside the window, bright
    /// outside.
    pub fn preference(self, minute: u32) -> Luminance {
        if self.contains(minute) {
            Luminance::Dark
        } else {
            Luminance::Bright
        }
    }
}

#[derive(Debug)]
pub struct Args {
    pub specs: Vec<OutputSpec>,
    pub intent: Intent,
    /// Escape hatch: skip wp_color_management_v1 entirely (untagged
    /// surfaces, compositor assumes sRGB). For debugging color issues.
    pub no_color_management: bool,
    /// GPU render-time profiling mode (global, not per-output).
    pub profile: ProfileMode,
    /// When set, playlists prefer dark wallpapers inside this local-clock window
    /// and bright ones outside it. `None` disables time-of-day filtering.
    pub dark_hours: Option<DarkHours>,
}

const USAGE: &str = "\
Usage: prism-bg <options...>

  -c, --color RRGGBB     Set the background color.
  -i, --image <path>     Set the image to display.
      --shader <file>    Render a GLSL fragment shader as a live wallpaper
                         (GPU, fp16, color-managed). Shadertoy-style: define
                         main() with iResolution/iTime; a shader that uses
                         iTime animates (vsync-paced, paused when occluded),
                         otherwise it renders a single frame. Mutually
                         exclusive with --image/--list.
      --fps <n>          Cap an animated shader's render rate (1..=1000).
                         Default is the compositor's vsync cadence. Cuts GPU
                         cost; iTime stays real-time. Requires --shader or a
                         --list containing shaders.
  -m, --mode <mode>      Set the mode to use for the image
                         (stretch|fit|fill|center|tile|solid_color).
  -o, --output <name>    Set the output to operate on or * for all,
                         starting a new per-output group.
      --list <file>      Rotate through the entries listed in <file>, one
                         path per line (relative to the file's directory;
                         blank lines and # comments ignored). Entries may be
                         images or .frag/.glsl shaders, interleaved. Outputs
                         sharing the group rotate in lockstep.
      --rotate-every <duration>
                         Rotation period for --list, e.g. 90s, 15m,
                         1h (bare number = seconds). Default: 15m.
      --randomize        Shuffle the playlist order; reshuffles each pass
                         without immediate repeats.
      --fade <duration>  Blur-dissolve transition on rotation instead of a
                         hard cut, e.g. 500ms, 2s. Requires --list.
      --cap-luminance <nits>
                         Hard-clip HDR content above this luminance.
      --scale-luminance <nits>
                         Scale HDR content linearly so its peak luminance
                         is at most this (preserves highlight structure).
                         Combines with --cap-luminance: scale first, then
                         clip — tames overall level and white outliers
                         independently.
      --tone-map <nits|auto>
                         Remaster HDR content to a display peak via the
                         BT.2390 EETF (knee + roll-off, hue-preserving).
                         'auto' asks the compositor for the output's
                         target luminance. Runs between --scale-luminance
                         and --cap-luminance.
      --intent <intent>  Rendering intent (perceptual|relative|absolute).
                         Default: perceptual.
      --no-color-management
                         Do not tag surfaces with color descriptions.
      --profile-gpu      Report per-output GPU render time (timestamp queries)
                         once, a few seconds after each shader load. Off by
                         default; near-zero overhead when enabled.
      --profile-gpu-every <duration>
                         Instead, report continuously every <duration>
                         (e.g. 30s, 5m). Implies --profile-gpu.
      --dark-hours <HH:MM-HH:MM>
                         Prefer dark wallpapers within this local-clock window
                         and bright ones outside it (e.g. 19:00-07:00).
                         Filters --list playlists; shaders self-declare
                         via `//!luminance dark|bright`, images by mean
                         luminance.
  -h, --help             Show help message and quit.
  -v, --version          Show the version number and quit.

Like swaybg, -i/-m/-c apply to the most recent -o (or to all outputs if
given before any -o). Color management is automatic: the image's cICP/ICC
metadata is honored and passed to the compositor parametrically.";

pub fn parse<I: Iterator<Item = String>>(mut argv: I) -> Result<Args> {
    let mut specs: Vec<OutputSpec> = Vec::new();
    let mut intent = Intent::Perceptual;
    let mut no_color_management = false;
    let mut profile = ProfileMode::Off;
    let mut dark_hours = None;

    // The implicit "*" spec; only kept if any flag touched it.
    let mut current = OutputSpec::new("*".to_string());
    let mut current_touched = false;

    while let Some(arg) = argv.next() {
        let mut value = |flag: &str| -> Result<String> {
            argv.next()
                .with_context(|| format!("{flag} requires a value"))
        };
        match arg.as_str() {
            "-o" | "--output" => {
                let name = value("--output")?;
                if current_touched || current.output != "*" {
                    specs.push(current.clone());
                }
                current = OutputSpec::new(name);
                current_touched = false;
            }
            "-i" | "--image" => {
                current.image = Some(PathBuf::from(value("--image")?));
                current_touched = true;
            }
            "--list" => {
                current.image_list = Some(PathBuf::from(value("--list")?));
                current_touched = true;
            }
            "--shader" => {
                current.shader = Some(PathBuf::from(value("--shader")?));
                current_touched = true;
            }
            "--fps" => {
                let v = value("--fps")?;
                let n: u32 = v
                    .parse()
                    .ok()
                    .filter(|n| (1..=1000).contains(n))
                    .with_context(|| format!("--fps expects an integer 1..=1000, got {v:?}"))?;
                current.fps = Some(n);
                current_touched = true;
            }
            "--rotate-every" => {
                current.rotate_every = Some(parse_duration(&value("--rotate-every")?)?);
                current_touched = true;
            }
            "--randomize" => {
                current.randomize = true;
                current_touched = true;
            }
            "--fade" => {
                current.fade = Some(parse_fade(&value("--fade")?)?);
                current_touched = true;
            }
            "-m" | "--mode" => {
                current.mode = Some(Mode::parse(&value("--mode")?)?);
                current_touched = true;
            }
            "-c" | "--color" => {
                current.color = Some(Color::parse(&value("--color")?)?);
                current_touched = true;
            }
            "--cap-luminance" => {
                current
                    .luminance
                    .get_or_insert(LuminanceControl::default())
                    .cap = Some(parse_nits(&value("--cap-luminance")?)?);
                current_touched = true;
            }
            "--scale-luminance" => {
                current
                    .luminance
                    .get_or_insert(LuminanceControl::default())
                    .scale_max = Some(parse_nits(&value("--scale-luminance")?)?);
                current_touched = true;
            }
            "--tone-map" => {
                let v = value("--tone-map")?;
                current.tone_map = Some(if v == "auto" {
                    ToneMap::Auto
                } else {
                    ToneMap::Nits(parse_nits(&v)?)
                });
                current_touched = true;
            }
            "--intent" => {
                intent = match value("--intent")?.as_str() {
                    "perceptual" => Intent::Perceptual,
                    "relative" => Intent::Relative,
                    "absolute" => Intent::Absolute,
                    other => bail!("invalid intent {other:?}"),
                };
            }
            "--no-color-management" => no_color_management = true,
            "--profile-gpu" => profile = ProfileMode::OnLoad,
            "--profile-gpu-every" => {
                profile = ProfileMode::Every(parse_duration(&value("--profile-gpu-every")?)?);
            }
            "--dark-hours" => dark_hours = Some(parse_dark_hours(&value("--dark-hours")?)?),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "-v" | "--version" => {
                println!("prism-bg {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}\n{USAGE}"),
        }
    }
    if current_touched || current.output != "*" {
        specs.push(current);
    }
    if specs.is_empty() {
        bail!("no outputs configured\n{USAGE}");
    }
    for spec in &specs {
        if spec.image.is_some() && spec.image_list.is_some() {
            bail!(
                "output {:?}: --image and --list are mutually exclusive",
                spec.output
            );
        }
        if spec.shader.is_some() && (spec.image.is_some() || spec.image_list.is_some()) {
            bail!(
                "output {:?}: --shader is mutually exclusive with --image/--list",
                spec.output
            );
        }
        if (spec.rotate_every.is_some() || spec.randomize || spec.fade.is_some())
            && spec.image_list.is_none()
        {
            bail!(
                "output {:?}: --rotate-every/--randomize/--fade require --list",
                spec.output
            );
        }
        // --fps caps an animated shader — a direct --shader, or a shader entry
        // within a --list playlist (its fps flows through at display time).
        if spec.fps.is_some() && spec.shader.is_none() && spec.image_list.is_none() {
            bail!(
                "output {:?}: --fps requires --shader or --list",
                spec.output
            );
        }
        // A shader fills the whole surface, so geometry modes don't apply.
        if spec.shader.is_none()
            && spec.effective_mode() != Mode::SolidColor
            && spec.image.is_none()
            && spec.image_list.is_none()
        {
            bail!(
                "output {:?}: mode {:?} requires an image",
                spec.output,
                spec.effective_mode()
            );
        }
    }
    Ok(Args {
        specs,
        intent,
        no_color_management,
        profile,
        dark_hours,
    })
}

/// Parse `HH:MM-HH:MM` into a [`DarkHours`] window (24-hour local clock).
fn parse_dark_hours(s: &str) -> Result<DarkHours> {
    let (start, end) = s
        .split_once('-')
        .with_context(|| format!("--dark-hours expects START-END, got {s:?}"))?;
    let minute = |hm: &str| -> Result<u32> {
        let (h, m) = hm
            .split_once(':')
            .with_context(|| format!("--dark-hours time {hm:?} is not HH:MM"))?;
        let h: u32 = h.parse().context("hour")?;
        let m: u32 = m.parse().context("minute")?;
        if h > 23 || m > 59 {
            bail!("--dark-hours time {hm:?} is out of range");
        }
        Ok(h * 60 + m)
    };
    let (start_min, end_min) = (minute(start)?, minute(end)?);
    if start_min == end_min {
        bail!("--dark-hours START and END must differ");
    }
    Ok(DarkHours { start_min, end_min })
}

/// `300ms`, `90s`, `15m`, `1.5h`; a bare number is seconds.
fn parse_seconds(s: &str) -> Result<f64> {
    let (num, mult) = if let Some(n) = s.strip_suffix("ms") {
        (n, 0.001)
    } else if let Some(n) = s.strip_suffix('s') {
        (n, 1.0)
    } else if let Some(n) = s.strip_suffix('m') {
        (n, 60.0)
    } else if let Some(n) = s.strip_suffix('h') {
        (n, 3600.0)
    } else {
        (s, 1.0)
    };
    let n: f64 = num
        .parse()
        .with_context(|| format!("invalid duration {s:?} (expected e.g. 300ms, 90s, 15m, 1h)"))?;
    Ok(n * mult)
}

fn parse_duration(s: &str) -> Result<Duration> {
    let secs = parse_seconds(s)?;
    if !secs.is_finite() || !(1.0..=86_400.0 * 365.0).contains(&secs) {
        bail!("duration {s:?} out of range (1s ..= 365 days)");
    }
    Ok(Duration::from_secs_f64(secs))
}

/// Fades are frame-paced; sub-second values are the common case.
fn parse_fade(s: &str) -> Result<Duration> {
    let secs = parse_seconds(s)?;
    if !secs.is_finite() || !(0.01..=60.0).contains(&secs) {
        bail!("fade duration {s:?} out of range (10ms ..= 60s)");
    }
    Ok(Duration::from_secs_f64(secs))
}

fn parse_nits(s: &str) -> Result<f64> {
    let n: f64 = s
        .parse()
        .with_context(|| format!("invalid luminance {s:?} (expected nits)"))?;
    if !n.is_finite() || n <= 0.0 || n > 10_000.0 {
        bail!("luminance {n} out of range (0, 10000] cd/m²");
    }
    Ok(n)
}

/// Pick the spec that applies to output `name`: an exact match wins over
/// the `*` fallback. swaybg semantics: later specs override earlier ones
/// for the same output.
pub fn spec_for_output<'a>(specs: &'a [OutputSpec], name: &str) -> Option<&'a OutputSpec> {
    specs
        .iter()
        .rev()
        .find(|s| s.output == name)
        .or_else(|| specs.iter().rev().find(|s| s.output == "*"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(args: &[&str]) -> Args {
        parse(args.iter().map(|s| s.to_string())).unwrap()
    }

    #[test]
    fn swaybg_style_groups() {
        let args = parse_ok(&[
            "-i",
            "default.png",
            "-m",
            "fill",
            "-o",
            "DP-1",
            "-i",
            "left.png",
            "-m",
            "tile",
            "-o",
            "DP-2",
            "-c",
            "#336699",
        ]);
        assert_eq!(args.specs.len(), 3);
        assert_eq!(args.specs[0].output, "*");
        assert_eq!(args.specs[0].effective_mode(), Mode::Fill);
        assert_eq!(args.specs[1].output, "DP-1");
        assert_eq!(args.specs[1].effective_mode(), Mode::Tile);
        assert_eq!(args.specs[2].output, "DP-2");
        assert!(args.specs[2].image.is_none());
        assert_eq!(args.specs[2].effective_mode(), Mode::SolidColor);
    }

    #[test]
    fn output_matching_prefers_exact_over_wildcard() {
        let args = parse_ok(&["-i", "a.png", "-o", "DP-1", "-c", "112233"]);
        assert_eq!(spec_for_output(&args.specs, "DP-1").unwrap().output, "DP-1");
        assert_eq!(
            spec_for_output(&args.specs, "HDMI-A-1").unwrap().output,
            "*"
        );
    }

    #[test]
    fn profile_flags() {
        assert_eq!(parse_ok(&["-c", "112233"]).profile, ProfileMode::Off);
        assert_eq!(
            parse_ok(&["-c", "112233", "--profile-gpu"]).profile,
            ProfileMode::OnLoad
        );
        assert_eq!(
            parse_ok(&["-c", "112233", "--profile-gpu-every", "30s"]).profile,
            ProfileMode::Every(Duration::from_secs(30))
        );
        // The interval form requires a value.
        assert!(parse(["--profile-gpu-every".to_string()].into_iter()).is_err());
    }

    #[test]
    fn dark_hours_parsing_and_window() {
        assert!(parse_ok(&["-c", "112233"]).dark_hours.is_none());
        // Wrapping window (evening through morning).
        let dh = parse_ok(&["-c", "112233", "--dark-hours", "19:00-07:00"])
            .dark_hours
            .unwrap();
        assert!(dh.contains(20 * 60)); // 8pm: dark
        assert!(dh.contains(3 * 60)); // 3am: dark
        assert!(!dh.contains(12 * 60)); // noon: bright
        assert_eq!(dh.preference(23 * 60), Luminance::Dark);
        assert_eq!(dh.preference(9 * 60), Luminance::Bright);
        // Non-wrapping window.
        let day = parse_ok(&["-c", "112233", "--dark-hours", "00:00-06:30"])
            .dark_hours
            .unwrap();
        assert!(day.contains(6 * 60 + 29));
        assert!(!day.contains(6 * 60 + 30));
        // Malformed inputs.
        assert!(parse(["--dark-hours".to_string()].into_iter()).is_err());
        for bad in ["19:00", "25:00-07:00", "19:60-07:00", "08:00-08:00"] {
            assert!(
                parse(
                    ["-c", "112233", "--dark-hours", bad]
                        .iter()
                        .map(|s| s.to_string())
                )
                .is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn image_defaults_to_stretch() {
        let args = parse_ok(&["-i", "a.png"]);
        assert_eq!(args.specs[0].effective_mode(), Mode::Stretch);
    }

    #[test]
    fn color_parses_with_and_without_hash() {
        let c = Color::parse("#ff8000").unwrap();
        assert!((c.r - 1.0).abs() < 1e-6);
        assert!((c.g - 128.0 / 255.0).abs() < 1e-6);
        assert!((c.b - 0.0).abs() < 1e-6);
        assert!(Color::parse("zzz").is_err());
    }

    #[test]
    fn mode_without_image_is_rejected() {
        assert!(parse(["-m", "fill"].iter().map(|s| s.to_string())).is_err());
    }

    #[test]
    fn fps_requires_shader_or_list_and_validates_range() {
        // Attaches to the shader's group.
        let args = parse_ok(&["--shader", "s.frag", "--fps", "30"]);
        assert_eq!(args.specs[0].fps, Some(30));
        // A playlist may contain shaders, so --fps is allowed with --list too.
        let args = parse_ok(&["--list", "walls.txt", "--fps", "30"]);
        assert_eq!(args.specs[0].fps, Some(30));
        // Neither shader nor list → rejected.
        assert!(parse(["--fps", "30"].iter().map(|s| s.to_string())).is_err());
        assert!(parse(["-i", "a.png", "--fps", "30"].iter().map(|s| s.to_string())).is_err());
        // Out of range / non-numeric → rejected.
        let bad = |v: &str| {
            parse(
                ["--shader", "s.frag", "--fps", v]
                    .iter()
                    .map(|s| s.to_string()),
            )
        };
        assert!(bad("0").is_err());
        assert!(bad("1001").is_err());
        assert!(bad("x").is_err());
    }

    #[test]
    fn image_list_flags() {
        let args = parse_ok(&[
            "--list",
            "walls.txt",
            "--rotate-every",
            "90s",
            "--randomize",
            "-m",
            "fill",
        ]);
        let spec = &args.specs[0];
        assert_eq!(
            spec.image_list.as_deref(),
            Some(std::path::Path::new("walls.txt"))
        );
        assert_eq!(spec.rotate_every, Some(Duration::from_secs(90)));
        assert!(spec.randomize);
        assert_eq!(spec.effective_mode(), Mode::Fill);
    }

    #[test]
    fn image_list_defaults_to_stretch() {
        let args = parse_ok(&["--list", "walls.txt"]);
        assert_eq!(args.specs[0].effective_mode(), Mode::Stretch);
        assert_eq!(args.specs[0].rotate_every, None); // default applied in main
    }

    #[test]
    fn duration_suffixes() {
        let parse_dur = |d: &str| {
            parse(
                ["--list", "w.txt", "--rotate-every", d]
                    .iter()
                    .map(|s| s.to_string()),
            )
            .map(|a| a.specs[0].rotate_every.unwrap())
        };
        assert_eq!(parse_dur("300").unwrap(), Duration::from_secs(300));
        assert_eq!(parse_dur("15m").unwrap(), Duration::from_secs(900));
        assert_eq!(parse_dur("1.5h").unwrap(), Duration::from_secs(5400));
        assert!(parse_dur("0").is_err());
        assert!(parse_dur("nope").is_err());
    }

    #[test]
    fn fade_durations() {
        let parse_fade = |d: &str| {
            parse(
                ["--list", "w.txt", "--fade", d]
                    .iter()
                    .map(|s| s.to_string()),
            )
            .map(|a| a.specs[0].fade.unwrap())
        };
        assert_eq!(parse_fade("500ms").unwrap(), Duration::from_millis(500));
        assert_eq!(parse_fade("2s").unwrap(), Duration::from_secs(2));
        assert_eq!(parse_fade("2").unwrap(), Duration::from_secs(2));
        assert!(parse_fade("5ms").is_err()); // below 10ms
        assert!(parse_fade("2m").is_err()); // above 60s
        assert!(parse_fade("nope").is_err());
    }

    #[test]
    fn fade_requires_image_list() {
        assert!(parse(
            ["-i", "a.png", "--fade", "500ms"]
                .iter()
                .map(|s| s.to_string())
        )
        .is_err());
    }

    #[test]
    fn image_and_image_list_are_mutually_exclusive() {
        assert!(parse(
            ["-i", "a.png", "--list", "w.txt"]
                .iter()
                .map(|s| s.to_string())
        )
        .is_err());
    }

    #[test]
    fn rotate_flags_require_image_list() {
        assert!(parse(["-i", "a.png", "--randomize"].iter().map(|s| s.to_string())).is_err());
        assert!(parse(
            ["-i", "a.png", "--rotate-every", "5m"]
                .iter()
                .map(|s| s.to_string())
        )
        .is_err());
    }
}

#[cfg(test)]
mod luminance_cli_tests {
    use super::*;

    #[test]
    fn luminance_flags_follow_output_groups() {
        let args = parse(
            [
                "-i",
                "a.jxr",
                "--cap-luminance",
                "600",
                "-o",
                "DP-2",
                "-i",
                "a.jxr",
                "--scale-luminance",
                "1000",
            ]
            .iter()
            .map(|s| s.to_string()),
        )
        .unwrap();
        assert_eq!(
            args.specs[0].luminance,
            Some(LuminanceControl {
                cap: Some(600.0),
                scale_max: None,
                tone_map: None
            })
        );
        assert_eq!(
            args.specs[1].luminance,
            Some(LuminanceControl {
                scale_max: Some(1000.0),
                cap: None,
                tone_map: None
            })
        );
    }

    #[test]
    fn nits_bounds_are_enforced() {
        assert!(parse(
            ["-i", "a.png", "--cap-luminance", "0"]
                .iter()
                .map(|s| s.to_string())
        )
        .is_err());
        assert!(parse(
            ["-i", "a.png", "--cap-luminance", "20000"]
                .iter()
                .map(|s| s.to_string())
        )
        .is_err());
        assert!(parse(
            ["-i", "a.png", "--scale-luminance", "abc"]
                .iter()
                .map(|s| s.to_string())
        )
        .is_err());
    }
}
