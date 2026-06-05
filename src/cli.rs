//! swaybg-compatible command line.
//!
//! The flags are order-sensitive the same way swaybg's are: `-i`/`-m`/`-c`
//! apply to the most recent `-o`; flags before any `-o` configure the
//! default (`*`) spec, which applies to outputs no explicit spec matches.
//! Hand-parsed — clap can't express the positional grouping.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::color::LuminanceControl;

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
            _ => bail!(
                "invalid mode {s:?} (expected stretch|fit|fill|center|tile|solid_color)"
            ),
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
        let chan = |i: usize| -> f64 {
            u8::from_str_radix(&hex[i..i + 2], 16).unwrap() as f64 / 255.0
        };
        Ok(Color {
            r: chan(0),
            g: chan(2),
            b: chan(4),
        })
    }
}

/// Per-output wallpaper spec. `output == "*"` is the fallback.
#[derive(Debug, Clone)]
pub struct OutputSpec {
    pub output: String,
    pub image: Option<PathBuf>,
    pub mode: Option<Mode>,
    pub color: Option<Color>,
    /// HDR luminance shaping (`--cap-luminance` / `--scale-luminance`).
    pub luminance: Option<LuminanceControl>,
}

impl OutputSpec {
    fn new(output: String) -> Self {
        OutputSpec {
            output,
            image: None,
            mode: None,
            color: None,
            luminance: None,
        }
    }

    /// Effective mode: explicit, else stretch with an image (swaybg's
    /// default), else solid color.
    pub fn effective_mode(&self) -> Mode {
        self.mode.unwrap_or(if self.image.is_some() {
            Mode::Stretch
        } else {
            Mode::SolidColor
        })
    }
}

#[derive(Debug)]
pub struct Args {
    pub specs: Vec<OutputSpec>,
    pub intent: Intent,
    /// Escape hatch: skip wp_color_management_v1 entirely (untagged
    /// surfaces, compositor assumes sRGB). For debugging color issues.
    pub no_color_management: bool,
}

const USAGE: &str = "\
Usage: prism-bg <options...>

  -c, --color RRGGBB     Set the background color.
  -i, --image <path>     Set the image to display.
  -m, --mode <mode>      Set the mode to use for the image
                         (stretch|fit|fill|center|tile|solid_color).
  -o, --output <name>    Set the output to operate on or * for all,
                         starting a new per-output group.
      --cap-luminance <nits>
                         Hard-clip HDR content above this luminance.
      --scale-luminance <nits>
                         Scale HDR content linearly so its peak luminance
                         is at most this (preserves highlight structure).
      --intent <intent>  Rendering intent (perceptual|relative|absolute).
                         Default: perceptual.
      --no-color-management
                         Do not tag surfaces with color descriptions.
  -h, --help             Show help message and quit.
  -v, --version          Show the version number and quit.

Like swaybg, -i/-m/-c apply to the most recent -o (or to all outputs if
given before any -o). Color management is automatic: the image's cICP/ICC
metadata is honored and passed to the compositor parametrically.";

pub fn parse<I: Iterator<Item = String>>(mut argv: I) -> Result<Args> {
    let mut specs: Vec<OutputSpec> = Vec::new();
    let mut intent = Intent::Perceptual;
    let mut no_color_management = false;

    // The implicit "*" spec; only kept if any flag touched it.
    let mut current = OutputSpec::new("*".to_string());
    let mut current_touched = false;

    while let Some(arg) = argv.next() {
        let mut value = |flag: &str| -> Result<String> {
            argv.next().with_context(|| format!("{flag} requires a value"))
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
            "-m" | "--mode" => {
                current.mode = Some(Mode::parse(&value("--mode")?)?);
                current_touched = true;
            }
            "-c" | "--color" => {
                current.color = Some(Color::parse(&value("--color")?)?);
                current_touched = true;
            }
            "--cap-luminance" => {
                current.luminance =
                    Some(LuminanceControl::Cap(parse_nits(&value("--cap-luminance")?)?));
                current_touched = true;
            }
            "--scale-luminance" => {
                current.luminance = Some(LuminanceControl::ScaleMax(parse_nits(
                    &value("--scale-luminance")?,
                )?));
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
        if spec.effective_mode() != Mode::SolidColor && spec.image.is_none() {
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
    })
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
            "-i", "default.png", "-m", "fill", "-o", "DP-1", "-i", "left.png", "-m", "tile",
            "-o", "DP-2", "-c", "#336699",
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
        assert_eq!(spec_for_output(&args.specs, "HDMI-A-1").unwrap().output, "*");
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
}

#[cfg(test)]
mod luminance_cli_tests {
    use super::*;

    #[test]
    fn luminance_flags_follow_output_groups() {
        let args = parse(
            ["-i", "a.jxr", "--cap-luminance", "600", "-o", "DP-2", "-i", "a.jxr",
             "--scale-luminance", "1000"]
                .iter()
                .map(|s| s.to_string()),
        )
        .unwrap();
        assert_eq!(args.specs[0].luminance, Some(LuminanceControl::Cap(600.0)));
        assert_eq!(
            args.specs[1].luminance,
            Some(LuminanceControl::ScaleMax(1000.0))
        );
    }

    #[test]
    fn nits_bounds_are_enforced() {
        assert!(parse(["-i", "a.png", "--cap-luminance", "0"].iter().map(|s| s.to_string())).is_err());
        assert!(parse(["-i", "a.png", "--cap-luminance", "20000"].iter().map(|s| s.to_string())).is_err());
        assert!(parse(["-i", "a.png", "--scale-luminance", "abc"].iter().map(|s| s.to_string())).is_err());
    }
}
