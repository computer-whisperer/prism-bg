//!luminance dark
// Bloom: will-o'-the-wisps — drifting lights whose HDR cores blossom into soft
// halos through a real separable-blur bloom. The worked example for the
// multi-pass render graph (see README, "multi-pass render graph"): a `scene`
// pass draws hot cores far above 1.0, `blurx` extracts everything over white
// and blurs it horizontally, `bloomy` blurs that vertically, and the `image`
// pass composites the glow back over the scene, mastered into the headroom.
// Uses iTime -> animated (multi-pass shaders redraw every frame; cap with
// --fps to taste).
//
//   prism-bg --shader examples/shaders/bloom.frag
//
// One field of lights across the whole cluster. Cheap: a fixed loop of ten
// gaussians in the scene pass and two 13-tap blurs at output resolution.
/*!prism
{ "buffers": ["scene", "blurx", "bloomy"],
  "channels": {
    "blurx":  {"0": "scene"},
    "bloomy": {"0": "blurx"},
    "image":  {"0": "scene", "1": "bloomy"} } }
*/
//!common
layout(push_constant) uniform Push {
    vec2 iResolution;
    float iTime;
    float _pad;
    vec2 iOutputOffset;
    vec2 iOutputSize;
    vec2 iGlobalResolution;
    float iRefWhite;  // cd/m²: output value 1.0 = diffuse white
    float iMaxLum;    // cd/m²: peak to master against
} pc;

float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

// Buffers sample y-down; fragCoord is y-up.
vec2 flip(vec2 uv) { return vec2(uv.x, 1.0 - uv.y); }

const float U32 = 2.3283064e-10;  // 1 / 2^32
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (s >> 22u) ^ s;
}
vec4 wispRand(uint i) {
    uint h = pcg(i * 0x9e3779b9u + 0x85ebca6bu);
    return vec4(pcg(h), pcg(h ^ 0x68bc21ebu), pcg(h ^ 0x02e5be93u), pcg(h ^ 0x967a889bu)) * U32;
}

// 13-tap gaussian with 6px tap spacing (sigma ~24px — wide enough that the
// halo clearly outreaches the core; the dilated taps don't band because the
// extracted signal is already smooth); `dir` selects the axis. `thr` is
// subtracted per tap BEFORE accumulating, so the first pass extracts only
// what exceeds diffuse white — thresholding after the blur would average the
// overshoot away and leave almost nothing to bloom.
vec3 blur13(sampler2D src, vec2 uv, vec2 dir, float thr) {
    vec2 px = dir * 6.0 / pc.iResolution;
    vec3 s = max(texture(src, uv).rgb - thr, 0.0) * 0.1064;
    float w[6] = float[](0.1027, 0.0926, 0.0779, 0.0612, 0.0448, 0.0306);
    for (int i = 1; i <= 6; i++) {
        s += max(texture(src, uv + px * float(i)).rgb - thr, 0.0) * w[i - 1];
        s += max(texture(src, uv - px * float(i)).rgb - thr, 0.0) * w[i - 1];
    }
    return s;
}

//!pass scene
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
void main() {
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = g / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 p = (gv - 0.5) * vec2(aspect, 1.0);

    // A dark marsh-at-night gradient with low mist pooling at the bottom.
    vec3 col = mix(vec3(0.016, 0.026, 0.032), vec3(0.003, 0.005, 0.010),
                   smoothstep(0.0, 1.0, gv.y));
    col += vec3(0.012, 0.020, 0.017) * smoothstep(0.45, 0.0, gv.y);

    // Twelve wisps wandering on slow sine paths, one continuous field across
    // the cluster. Cores run far above 1.0 — that overshoot is what blooms.
    for (uint i = 0u; i < 12u; i++) {
        vec4 rnd = wispRand(i);
        float t = pc.iTime * (0.05 + 0.05 * rnd.z) + rnd.w * 40.0;
        vec2 c = vec2(
            (rnd.x - 0.5) * aspect + 0.30 * aspect * sin(t + rnd.y * 6.28),
            (rnd.y - 0.5) + 0.22 * sin(t * 1.7 + rnd.x * 6.28));
        float breathe = 0.75 + 0.25 * sin(pc.iTime * (0.3 + rnd.y) + rnd.z * 6.28);
        vec3 tint = mix(vec3(0.55, 1.0, 0.75), vec3(0.55, 0.75, 1.0), rnd.z);
        tint = mix(tint, vec3(1.0, 0.75, 0.45), step(0.75, rnd.w));
        float d = length(p - c);
        col += tint * exp(-d * d * 7000.0) * 5.0 * breathe;  // hot core, >> 1.0
        col += tint * exp(-d * d * 220.0) * 0.06;            // faint local haze
    }
    outColor = vec4(col, 1.0);
}

//!pass blurx
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(set = 1, binding = 0) uniform sampler2D iChannel0;  // = scene
void main() {
    vec2 uv = flip(fragCoord / pc.iResolution);
    // Extract what exceeds diffuse white and blur it horizontally.
    outColor = vec4(blur13(iChannel0, uv, vec2(1.0, 0.0), 1.0), 1.0);
}

//!pass bloomy
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(set = 1, binding = 0) uniform sampler2D iChannel0;  // = blurx
void main() {
    vec2 uv = flip(fragCoord / pc.iResolution);
    outColor = vec4(blur13(iChannel0, uv, vec2(0.0, 1.0), 0.0), 1.0);
}

//!pass image
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(set = 1, binding = 0) uniform sampler2D iChannel0;  // = scene
layout(set = 1, binding = 1) uniform sampler2D iChannel1;  // = bloomy
void main() {
    vec2 uv = flip(fragCoord / pc.iResolution);
    vec3 scene = texture(iChannel0, uv).rgb;
    vec3 glow = texture(iChannel1, uv).rgb;
    // The blurred overshoot becomes the halo; the sum is mastered into the
    // available headroom (on SDR the halo simply saturates toward white).
    vec3 col = scene + glow * 1.3;
    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
