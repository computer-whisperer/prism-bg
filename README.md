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
  buffers (`Abgr16161616f`), premultiplied electrical. Without fp16 shm
  the buffer degrades along a ladder: 16-bit unorm (`Abgr16161616`) —
  PQ-encoding linear HDR content so the luminance range survives intact
  (the KWin path: real HDR, no fp16) — then 8-bit. The compositor's
  named-TF vocabulary is negotiated the same way (KWin dropped the
  deprecated `srgb` TF; pixels re-encode to gamma 2.2 there).
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
`solid_color`.

Extras beyond swaybg (also per-output-group):

- `--cap-luminance <nits>` — hard-clip HDR content above this luminance.
- `--scale-luminance <nits>` — scale HDR content linearly so its measured
  peak is at most this (preserves highlight structure; no-op if already
  below). The two compose, scale first: a generous scale plus a tight cap
  tames "color is sane, white peaks are crazy" masters. Both operate in
  absolute nits on linear and PQ sources, update the declared luminance
  maximum to the honest post-treatment ceiling (the measured peak, even
  when nothing changed), and warn+pass-through on SDR images.
- `--tone-map <nits|auto>` — remaster HDR content to a display peak with
  the BT.2390 EETF (knee + Hermite roll-off in PQ space, applied max-RGB
  so hue survives). `auto` resolves the target per output from the
  compositor's preferred image description (`target_max_cll`, falling
  back to the target luminance) — the same numbers prism derives from
  its HDR config and EDID. Runs between scale and cap.
- `--intent perceptual|relative|absolute` (default perceptual).
- `--no-color-management` (debug escape hatch).

Formats: PNG, JPEG, WebP, JPEG XL, AVIF (needs libdav1d), JPEG XR
(Windows HDR wallpapers/screenshots — scRGB), OpenEXR, Radiance HDR.

## Known gaps

- EXR `chromaticities` attribute isn't read (the `image` crate doesn't
  expose it); EXR is assumed Rec.709 scene-linear, 1.0 = 203 cd/m².
- AVIF item↔property associations aren't chased; the first `colr` box
  wins (right for single-image files).
- Fractional scale: `center`/`tile` use the integer output scale.
- Animated outputs of any kind (this is a wallpaper).
