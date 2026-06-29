//!luminance dark
// Neon rain on glass: falling droplets refract a soft city glow behind them.
// Uses iTime -> animated wallpaper. Procedural only; no texture channel.
//
//   prism-bg --shader examples/shaders/neon-rain.frag
//
// Built in shared cluster space: the rain columns and the city glow run
// continuously across a multi-monitor desktop instead of restarting on each
// output, at constant per-monitor density.
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

float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

float hash21(vec2 p) {
    p = fract(p * vec2(234.13, 87.37));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

// PCG bit-hash (full period) for the city-glow grid: one sign per cell along a
// global horizontal axis, so its placement must not repeat as the integer cell
// index marches across the workspace.
const float U32 = 2.3283064e-10;  // 1 / 2^32
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (s >> 22u) ^ s;
}
// Three decorrelated randoms in [0,1) keyed on a 1-D cell index.
vec3 cellRand3(uint seed) {
    uint h = pcg(seed);
    return vec3(pcg(h), pcg(h ^ 0x9e3779b9u), pcg(h ^ 0x85ebca6bu)) * U32;
}

float dropLayer(vec2 uv, float cells, float speed, float t) {
    vec2 p = uv * vec2(cells, cells * 0.55);
    p.y += t * speed;
    vec2 id = floor(p);
    vec2 f = fract(p);
    float h = hash21(id);
    float lane = h * 0.72 + 0.14;
    float y = fract(h * 13.7 + t * speed * (0.25 + 0.5 * h));
    vec2 d = vec2((f.x - lane) * 2.4, f.y - y);
    float head = smoothstep(0.11, 0.0, length(d * vec2(1.0, 2.4)));
    float trail = smoothstep(0.055, 0.0, abs(f.x - lane));
    float behind = f.y - y;
    trail *= smoothstep(-0.55, -0.06, behind) * (1.0 - smoothstep(-0.02, 0.08, behind));
    return (head + trail * 0.45) * smoothstep(0.35, 1.0, h);
}

// City glow behind the glass: soft neon signs on a repeating horizontal grid,
// so the field stays continuous and keeps constant density across any number of
// outputs. `X` is an aspect-correct global horizontal coordinate (continuous
// across the seam between monitors); `v` is vertical in [0, 1].
vec3 neonField(float X, float v) {
    vec3 col = mix(vec3(0.015, 0.020, 0.040), vec3(0.030, 0.015, 0.035), v);
    float freq = 3.3;                   // signs per unit X (~6 across a 16:9 output)
    float cell = floor(X * freq);
    for (int k = -1; k <= 1; k++) {     // neighbours, so a sign's glow crosses cell edges
        float n = cell + float(k);
        vec3 r = cellRand3(uint(int(n) + 9973));
        float cx = (n + r.x) / freq;    // jittered horizontal centre
        float cy = 0.14 + 0.10 * r.y;   // low on the wall
        float d = length(vec2((X - cx) * 3.6, (v - cy) * 4.0));
        vec3 hue = 0.5 + 0.5 * cos(vec3(0.0, 2.1, 4.2) + n * 1.3);
        col += hue * exp(-d * 2.6) * 0.20;
    }
    return col;
}

void main() {
    // Global cluster space: the rain columns and city glow run continuously
    // across outputs instead of restarting per monitor. gx is aspect-correct
    // (so drops stay vertical and signs stay round) and continuous across seams.
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = g / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    float gx = gv.x * aspect;

    // cells per unit-x is fixed, so feeding a wider global x scales the cell
    // count with the workspace and keeps per-monitor drop density unchanged.
    float d1 = dropLayer(vec2(gx, gv.y), 18.0, 0.45, pc.iTime);
    float d2 = dropLayer(vec2(gx + 7.3, gv.y + 2.1), 31.0, 0.85, pc.iTime);
    float drops = clamp(d1 + d2 * 0.7, 0.0, 1.0);

    vec2 refr = vec2(dFdx(drops), dFdy(drops)) * 0.075;
    vec3 bg = neonField(gx + refr.x, gv.y + refr.y);

    // Window streaks and wet specular edges.
    float streak = smoothstep(0.0, 0.9, drops);
    vec3 glass = vec3(0.38, 0.58, 0.75) * streak * 0.18;
    vec3 spec = vec3(0.9, 0.95, 1.0) * pow(streak, 5.0) * 0.45 * headroom();

    // Full-workspace vignette: each axis is normalized to the desktop rectangle
    // so the falloff tracks the whole span (no per-monitor seam) and stays the
    // same shape at any monitor count. A high floor keeps the outer monitors of
    // a wide layout lit; the gentle centre dip also tames the overall level.
    vec2 vc = vec2((gx - 0.5 * aspect) / max(0.5 * aspect, 1e-3),
                   (gv.y - 0.5) / 0.5);
    float vig = smoothstep(1.6, 0.3, length(vc));
    vec3 col = bg * (0.55 + 0.40 * vig) + glass + spec;

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
