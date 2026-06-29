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
- `--list <file>` — rotate through the images listed in `<file>`,
  one path per line (relative paths resolve against the file's directory;
  blank lines and `#` comments are ignored; `~/` expands). Mutually
  exclusive with `-i`. All outputs matched by the group rotate in
  lockstep, decoding each image once. Unloadable entries are skipped with
  a warning, at startup and at rotation time.
- `--rotate-every <duration>` — rotation period, e.g. `90s`, `15m`, `1h`
  (bare number = seconds). Default 15m.
- `--randomize` — shuffle the playlist; reshuffles each pass, never
  repeating the same image back-to-back.
- `--fade <duration>` — crossfade on rotation instead of a hard cut,
  e.g. `500ms`, `2s`. The incoming image rides a second subsurface whose
  `wp_alpha_modifier_v1` multiplier ramps up while the outgoing one stays
  mapped beneath it — the compositor does the blend (prism in linear
  fp16, so the dissolve is gamma-correct), and each image keeps its own
  color description throughout. Without compositor support the flag
  degrades to a hard cut with a warning.
- `--intent perceptual|relative|absolute` (default perceptual).
- `--no-color-management` (debug escape hatch).
- `--shader <file>` — render a GLSL fragment shader on the GPU as a live
  wallpaper (extended-linear output, so it goes through the same HDR-aware
  pipeline as an image). The shader provides `main()` and reads a
  push-constant block; a shader that uses `iTime` animates (vsync-paced,
  paused when occluded), one that doesn't renders once. See
  `examples/shaders/` for the format. The uniforms:

  ```glsl
  layout(push_constant) uniform Push {
      vec2 iResolution;        // this output, device px (fragCoord is y-up, 0..iResolution)
      float iTime;             // seconds since start (wall-clock)
      float _pad;
      vec2 iOutputOffset;      // this output's bottom-left in the multi-monitor
      vec2 iOutputSize;        //   cluster, and its size — both logical px, y-up
      vec2 iGlobalResolution;  // the whole cluster, logical px
      float iRefWhite;         // cd/m²: output value 1.0 = diffuse white
      float iMaxLum;           // cd/m²: peak luminance to master against
      vec4 iMouse;             // xy: cursor (while held); zw: click pos (sign = state)
      vec4 iDate;              // local (year, month 0-11, day, seconds-since-midnight)
      float iTimeDelta;        // seconds since the previous frame (0 on the first)
      int iFrame;              // frames rendered since start (0 on the first)
  } pc;
  ```

  A shader may declare only the prefix it uses — the block is read positionally,
  so older shaders that stop at `iGlobalResolution` keep working.

  The cluster fields let a shader tile *continuously across the whole
  workspace* instead of restarting per monitor:
  `vec2 g = iOutputOffset + (fragCoord / iResolution) * iOutputSize;` is this
  fragment's position in shared cluster space (logical px, same on every
  monitor regardless of resolution/DPI); divide by `iGlobalResolution` for a
  `0..1` coordinate spanning the desktop. A shader that only reads
  `iResolution`/`iTime` still works and behaves per-output. See
  `examples/shaders/hexgrid.frag` for a worked example.

  **Luminance (`iRefWhite` / `iMaxLum`).** The shader output is tagged
  extended-linear with the perceptual (anchored) intent, so the compositor maps
  value `1.0` to the output's diffuse white — i.e. **`1.0` is white**, and the
  buffer is HDR-capable above that. `iRefWhite` is that white in cd/m² and
  `iMaxLum` is the peak the compositor advertises to master against (its
  configured mastering peak, *not* the panel's marketing/HDR-metadata number).
  Highlight headroom above white is `iMaxLum / iRefWhite` (≥ 1.0; exactly 1.0 on
  an SDR output). **A shader is responsible for keeping its own output within
  `[0, iMaxLum/iRefWhite]`** — prism does not tone-map the buffer for you; a
  shader that drives past the peak gets rolled off by the output's display LUT
  and reads as blown-out. Master HDR highlights into the headroom and clamp.
  Before an output's capabilities resolve (and with no color management) the
  values default to an SDR-safe `203 / 203` (headroom 1.0), so a shader never
  overblows at startup. See `examples/shaders/bloom.frag` (HDR bloom into the
  headroom) and `examples/shaders/feedback.frag`.

  **Mouse (`iMouse`).** Reference `iMouse` and prism-bg makes the surface
  pointer-interactive: it binds a seat pointer, gives the wallpaper a normal
  cursor, and **redraws it on each pointer event** ("repaint on motion"), so a
  mouse-reactive shader needs no `iTime` — it renders once and then only while
  the cursor moves or clicks, costing no GPU when idle. The convention matches
  Shadertoy, in device pixels with a y-up origin (like `fragCoord`): `iMouse.xy`
  is the cursor position while a button is held (holding the last drag position
  once released), `iMouse.zw` is the position of the press with `sign(iMouse.z)`
  encoding whether the button is currently down and `sign(iMouse.w)` whether the
  press happened this frame. All four are zero until the first click. Only
  shaders that reference `iMouse` receive input; every other wallpaper stays
  click-through. See `examples/shaders/feature_demos/pointer.frag`.

  **Timing and date (`iTimeDelta` / `iFrame` / `iDate`).** `iTimeDelta` is the
  wall-clock seconds since the previous rendered frame (for frame-rate-independent
  integration), `iFrame` the frame counter since start, and `iDate` the local
  wall clock as `(year, month 0-11, day, seconds-since-midnight)` with a
  fractional seconds component — enough for a clock wallpaper. See
  `examples/shaders/feature_demos/clock.frag`.

  A shader can also be **audio-reactive**: reference any of the audio uniforms
  and prism-bg captures the default sink's output over PipeWire, runs an FFT,
  and feeds a live spectrum each frame (the capture only starts when a shader
  uses it, and follows the default device). Audio shaders redraw continuously
  to track the sound, so they count as animated (cap them with `--fps`).

  ```glsl
  layout(set = 0, binding = 0, std140) uniform Audio {
      vec4 iAudioBins[8];   // 32 magnitude bins 0..1, low→high, packed 4/vec4
      float iAudioLevel;    // overall loudness 0..1
      float iAudioBass;     // low/mid/high band energy 0..1
      float iAudioMid;
      float iAudioTreble;
  } au;
  // read bin i (0..31): au.iAudioBins[i >> 2][i & 3]
  ```

  See `examples/shaders/feature_demos/spectrum.frag` for a worked example. With no PipeWire
  (or no audio playing) the values are zero and the shader just renders silence.

  A shader can also use **feedback** — sampling its own previous frame — for
  trails, decay, reaction-diffusion, fluid, and other evolving effects.
  Reference `iPrevFrame` and prism-bg renders the shader into a ping-pong
  buffer, feeding last frame back in (the buffer is `RGBA16_SFLOAT` linear, so
  trails keep HDR range — unlike Shadertoy's LDR buffers):

  ```glsl
  layout(set = 1, binding = 0) uniform sampler2D iPrevFrame;
  // fragCoord is y-up but textures sample y-down, so flip y when reading:
  vec3 prev(vec2 uv) { return texture(iPrevFrame, vec2(uv.x, 1.0 - uv.y)).rgb; }
  ```

  See `examples/shaders/feedback.frag`. Feedback shaders redraw every frame to
  evolve (cap with `--fps`); on the first frame `iPrevFrame` is black.

  For effects that need more than one buffer — separable blur, bloom, fluid,
  multi-stage simulation — a shader can declare a full **multi-pass render
  graph** with a `/*!prism …*/` JSON metadata block. It lists named offscreen
  buffer passes (each an `RGBA16_SFLOAT` ping-pong target) and wires each pass's
  `iChannel0..3` inputs; the pass bodies follow in `//!pass <name>` sections,
  with optional shared code in a `//!common` section (spliced into every pass
  after its `#version`):

  ```glsl
  /*!prism
  { "buffers": ["scene", "bloom"],
    "channels": {
      "bloom": {"0": "scene"},               // reads the scene buffer
      "image": {"0": "scene", "1": "bloom"}  // displayed pass reads both
    } }
  */
  //!common
  /* uniforms + helpers shared by all passes */
  //!pass scene
  #version 450
  /* … renders into the "scene" buffer … */
  //!pass bloom
  #version 450
  layout(set = 1, binding = 0) uniform sampler2D iChannel0; // = scene
  /* … */
  //!pass image
  #version 450
  layout(set = 1, binding = 0) uniform sampler2D iChannel0; // = scene
  layout(set = 1, binding = 1) uniform sampler2D iChannel1; // = bloom
  /* … the displayed result … */
  ```

  Buffers render in declared order, then the `image` pass to the screen. A
  channel reads the referenced buffer's *current* frame if that buffer rendered
  earlier this frame, else its *previous* frame; `"self"` reads a buffer's own
  previous frame (feedback, like `iPrevFrame`). Channels are `set = 1`,
  `binding = <N>` (the `iChannelN` index) and are sampled y-flipped, same as
  `iPrevFrame`. Up to 4 channels per pass. Multi-pass shaders redraw every
  frame. See `examples/shaders/bloom.frag` for a worked separable-blur bloom.

  **Static image channels (`textures`).** A `/*!prism …*/` block can also declare
  a `textures` map (name → path, resolved relative to the `.frag` file); a channel
  routes to a texture by naming it, exactly like a buffer. The image is decoded
  and uploaded once per GPU (repeat wrap + linear filter). A shader that only
  wants a texture needs no `//!pass` sections (a plain body plus the metadata
  block is enough):

  ```glsl
  /*!prism
  { "textures": {
      "noise": "../textures/rgba-noise.png",          // raw (default)
      "photo": { "path": "wall.jpg", "srgb": true }   // color: linearize on read
    },
    "channels": { "image": { "0": "noise", "1": "photo" } } }
  */
  #version 450
  layout(set = 1, binding = 0) uniform sampler2D iChannel0; // = noise
  layout(set = 1, binding = 1) uniform sampler2D iChannel1; // = photo
  /* … sample texture(iChannelN, uv) … */
  ```

  By default a texture is sampled **raw** (Shadertoy-style, no linearization) —
  correct for noise/data textures, where the stored value *is* the data. Set
  `"srgb": true` (the object form) for a **color** image, so the sampler
  linearizes it on read into the linear working space. Texture, buffer, and
  `"self"` channels mix freely in one shader. Sources are decoded to 8-bit sRGB
  (HDR/wide-gamut textures are flattened — author HDR in-shader for now), and
  `iChannelResolution` is not yet provided. See `examples/shaders/clouds.frag`
  (fbm clouds from a raw noise texture).
- `--fps <n>` — cap an animated shader's render rate (1..=1000); default is
  the compositor's vsync cadence. `iTime` stays real-time, so animation speed
  is unchanged — it's just sampled less often. Requires `--shader`.

Formats: PNG, JPEG, WebP, JPEG XL, AVIF (needs libdav1d), JPEG XR
(Windows HDR wallpapers/screenshots — scRGB), OpenEXR, Radiance HDR.

## Known gaps

- EXR `chromaticities` attribute isn't read (the `image` crate doesn't
  expose it); EXR is assumed Rec.709 scene-linear, 1.0 = 203 cd/m².
- AVIF item↔property associations aren't chased; the first `colr` box
  wins (right for single-image files).
- Fractional scale: `center`/`tile` use the integer output scale.
- Animated outputs of any kind (this is a wallpaper).

## Arch package

An AUR-oriented `PKGBUILD` is provided for tagged releases. It installs the
`prism-bg` binary; runtime dependencies are just `dav1d` (AVIF decode),
`gcc-libs`, and `glibc` — the Wayland stack is pure Rust.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
