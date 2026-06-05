# prism-bg

swaybg, in Rust, color-managed. The wallpaper client for
[prism](https://github.com/computer-whisperer/prism) — works on any
compositor with `wlr-layer-shell` + `wp_viewporter`, but the point is
`wp_color_management_v1`: the image's *actual* color encoding is decoded
and handed to the compositor parametrically, so prism's calibrated
pipeline renders it correctly on every output, SDR or HDR.

## What "color managed" means here

- **Metadata is honored, not assumed.** PNG `cICP`/`iCCP`/`sRGB`/`gAMA`+`cHRM`,
  JPEG XL color encodings (PQ HDR included; HLG is re-rendered to PQ),
  AVIF `colr` boxes (parsed from the container — decoders don't expose
  them), JPEG/WebP ICC profiles, EXR/Radiance scene-linear conventions.
- **ICC resolves client-side** (prism takes parametric descriptions only):
  a v4.4 `cicp` tag maps directly; matrix-shaper profiles whose TRC matches
  a named TF within half an 8-bit step are tagged with pixels untouched
  (custom primaries ride `set_primaries`, so Adobe RGB and friends keep
  their gamut); odd TRCs are re-encoded through the sRGB curve keeping
  their gamut; LUT profiles convert to BT.2020. Conversions stay
  display-referred so SDR content remains anchored to the output's
  reference white.
- **Pixels ship at full fidelity.** 8-bit sRGB-ish sources stay 8-bit;
  anything wider (16-bit PNG, 10-bit AVIF, JXL, EXR) rides fp16 shm
  buffers (`Abgr16161616f`), premultiplied electrical. Compositors
  without fp16 shm get a quantized 8-bit fallback.
- **No client-side resampling.** Every mode except `tile` submits the
  image at native resolution and lets `wp_viewport` describe the crop and
  size; prism scales in linear-light fp16, which is gamma-correct —
  unlike the cairo path in swaybg. `tile` assembles at device pixels
  (pure copies).

## Usage

Drop-in for swaybg:

```
prism-bg -i wallpaper.jxl -m fill
prism-bg -o DP-1 -i left.avif -m fill -o DP-2 -c 002030
```

`-i`/`-m`/`-c` apply to the most recent `-o` (or to all outputs before any
`-o`). Modes: `stretch` (default), `fit`, `fill`, `center`, `tile`,
`solid_color`. Extras beyond swaybg: `--intent
perceptual|relative|absolute` (default perceptual) and
`--no-color-management` (debug escape hatch).

Formats: PNG, JPEG, WebP, JPEG XL, AVIF (needs libdav1d), OpenEXR,
Radiance HDR.

## Known gaps

- EXR `chromaticities` attribute isn't read (the `image` crate doesn't
  expose it); EXR is assumed Rec.709 scene-linear, 1.0 = 203 cd/m².
- AVIF item↔property associations aren't chased; the first `colr` box
  wins (right for single-image files).
- Fractional scale: `center`/`tile` use the integer output scale.
- Animated outputs of any kind (this is a wallpaper).
