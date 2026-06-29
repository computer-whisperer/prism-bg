//!luminance dark
// Dreamy floating bokeh — soft out-of-focus light discs drifting upward over
// a dark gradient, like backlit dust. Uses iTime → animated. Cheap: a small
// loop of cheap disc evaluations, no noise.
//
//   prism-bg --shader examples/shaders/bokeh.frag
//
// Built in shared cluster space: the discs tile continuously across a
// multi-monitor desktop at constant per-monitor density, rather than cloning
// the same field onto every output.
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
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

// Highlight headroom above diffuse white (>= 1.0; 1.0 on SDR).
float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

// PCG hashes (full period) so the tiled discs don't repeat across the workspace
// as the integer column index marches. pcg4d yields four decorrelated randoms
// per disc in one mixing pass; a scalar pcg supplies an independent tint pick.
const float U32 = 2.3283064e-10;  // 1 / 2^32
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (s >> 22u) ^ s;
}
uvec4 pcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}
// Four randoms in [0,1) for the disc in column `cx`, sub-index `k`.
vec4 discRand(int cx, int k) {
    uvec4 s = uvec4(uint(cx), uint(k), uint(cx) ^ 0x9e3779b9u, uint(k) ^ 0x85ebca6bu);
    return vec4(pcg4d(s)) * U32;
}

void main() {
    // Global cluster space: discs tile across a multi-monitor desktop at
    // constant per-monitor density instead of cloning the same field per output.
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = g / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 P = (gv - 0.5) * vec2(aspect, 1.0);   // centred, aspect-correct, y in [-0.5, 0.5]
    float t = pc.iTime * 0.05;

    // Backdrop: cool gradient, skewed darker than the single-screen bokeh, and
    // continuous across outputs via global gv. A hint warmer low and to the left.
    vec3 col = mix(vec3(0.010, 0.015, 0.030), vec3(0.025, 0.020, 0.050), gv.y);
    col += vec3(0.020, 0.010, 0.016) * (1.0 - gv.x) * (1.0 - gv.y);

    // Discs tiled in vertical lanes: one column every CELL units of global x,
    // K discs per column rising and wrapping just off the top/bottom edges (so
    // the wrap is hidden behind the view boundary). Scan neighbouring columns
    // since a soft disc spills past its lane. A touch sparser than the old 18.
    const float CELL = 0.28;
    const int K = 2;
    const float SPAN = 1.5;   // vertical wrap span; margin > max disc radius hides the wrap
    int base = int(floor(P.x / CELL));
    for (int cx = base - 2; cx <= base + 2; cx++) {
        for (int k = 0; k < K; k++) {
            vec4 r = discRand(cx, k);
            float speed = 0.04 + 0.10 * r.y;
            float size = 0.05 + 0.14 * r.z;

            float y = fract(r.x + t * speed * 6.0) * SPAN - 0.5 * SPAN;
            float x = (float(cx) + r.w) * CELL + 0.10 * sin(t * 6.0 + r.x * 6.2831);

            float d = length(P - vec2(x, y)) / size;

            // Soft disc with a slightly brighter rim — classic bokeh look.
            float disc = smoothstep(1.0, 0.6, d);
            float rim = smoothstep(1.0, 0.9, d) * (1.0 - smoothstep(0.9, 0.6, d));
            float a = disc * 0.12 + rim * 0.10;

            // Tint: cool blue <-> warm amber, an independent pick per disc.
            float tk = float(pcg(uint(cx) * 2654435761u + uint(k) * 40503u + 12345u)) * U32;
            vec3 tint = mix(vec3(0.5, 0.7, 1.0), vec3(1.0, 0.7, 0.5), tk);
            col += tint * a * headroom();
        }
    }

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
