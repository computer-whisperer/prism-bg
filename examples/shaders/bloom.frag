// Bloom via a separable blur — the canonical multi-pass effect. Animated
// glowing orbs drift across the screen; their bright cores bleed a soft halo
// produced by a two-axis Gaussian blur split across buffers (the only
// affordable way to do a wide blur: a single 2D kernel is O(r²) taps, two 1D
// passes are O(r)).
//
//   prism-bg --shader examples/shaders/bloom.frag
//
// MULTI-PASS: the prism metadata block below declares offscreen buffer
// passes and how each pass's iChannelN inputs are wired. Each named buffer is
// an fp16 ping-pong target rendered in order; a pass reads another buffer's
// CURRENT frame if it renders earlier this frame, else its PREVIOUS frame.
// "self" means a buffer's own previous frame (feedback). The //!common section
// is shared GLSL spliced into every pass after its #version line; //!pass
// <name> sections hold each pass body. As with iPrevFrame, buffers are sampled
// y-flipped (fragCoord is y-up, textures are y-down) — see tap() below.
//
//   scene : the sharp source (orbs).
//   bloom : horizontal blur of scene's bright pass.
//   image : scene + vertical blur of bloom  →  the composited result.
/*!prism
{
  "buffers": ["scene", "bloom"],
  "channels": {
    "bloom": {"0": "scene"},
    "image": {"0": "scene", "1": "bloom"}
  }
}
*/

//!common
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push {
    vec2 iResolution;
    float iTime;
    float _pad;
    vec2 iOutputOffset;
    vec2 iOutputSize;
    vec2 iGlobalResolution;
} pc;

// 9-tap Gaussian weights (sigma ≈ 2), normalized.
const float W[5] = float[](0.227027, 0.194595, 0.121622, 0.054054, 0.016216);

// Sample a buffer y-flipped (see header).
vec3 tap(sampler2D t, vec2 uv) { return texture(t, vec2(uv.x, 1.0 - uv.y)).rgb; }

// Separable Gaussian along `dir` (in texels), scaled by `spread`.
vec3 blur(sampler2D t, vec2 uv, vec2 dir, float spread) {
    vec2 px = dir / pc.iResolution * spread;
    vec3 acc = tap(t, uv) * W[0];
    for (int i = 1; i < 5; i++) {
        acc += tap(t, uv + px * float(i)) * W[i];
        acc += tap(t, uv - px * float(i)) * W[i];
    }
    return acc;
}

//!pass scene
#version 450
// Three drifting orbs with bright HDR cores.
vec3 palette(float t) {
    return 0.5 + 0.5 * cos(6.28318 * (t + vec3(0.0, 0.33, 0.67)));
}
void main() {
    vec2 uv = fragCoord / pc.iResolution;
    float aspect = pc.iResolution.x / pc.iResolution.y;
    vec2 p = vec2(uv.x * aspect, uv.y);
    vec3 col = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        vec2 c = vec2(
            0.5 * aspect + 0.32 * aspect * cos(pc.iTime * (0.5 + 0.2 * fi) + fi * 2.1),
            0.5 + 0.30 * sin(pc.iTime * (0.6 + 0.15 * fi) + fi * 1.7)
        );
        float d = distance(p, c);
        // Sharp core (overdriven past 1.0 so the bloom has HDR energy to spread)
        // plus a faint near-field glow.
        col += palette(0.1 * fi + pc.iTime * 0.05) * (4.0 * smoothstep(0.05, 0.0, d));
    }
    outColor = vec4(col, 1.0);
}

//!pass bloom
#version 450
// Bright-pass + horizontal Gaussian blur of the scene: keep only energy above
// 1.0 (the orb cores), then blur it sideways. The vertical half happens in the
// image pass, completing the separable 2D blur.
layout(set = 1, binding = 0) uniform sampler2D iChannel0; // scene
vec3 bright(vec2 uv) { return max(tap(iChannel0, uv) - 1.0, vec3(0.0)); }
void main() {
    vec2 uv = fragCoord / pc.iResolution;
    vec2 px = vec2(1.0, 0.0) / pc.iResolution * 2.0;
    vec3 acc = bright(uv) * W[0];
    for (int i = 1; i < 5; i++) {
        acc += bright(uv + px * float(i)) * W[i];
        acc += bright(uv - px * float(i)) * W[i];
    }
    outColor = vec4(acc, 1.0);
}

//!pass image
#version 450
// Composite: sharp scene + vertical blur of the horizontally-blurred bloom.
layout(set = 1, binding = 0) uniform sampler2D iChannel0; // scene
layout(set = 1, binding = 1) uniform sampler2D iChannel1; // bloom (H-blurred)
void main() {
    vec2 uv = fragCoord / pc.iResolution;
    vec3 scene = tap(iChannel0, uv);
    vec3 glow = blur(iChannel1, uv, vec2(0.0, 1.0), 2.0);
    outColor = vec4(scene + glow * 1.3, 1.0);
}
