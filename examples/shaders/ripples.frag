//!luminance dark
// Rain on dark water — raindrops scattered across a night-black pool, each
// sending an expanding ring of capillary waves that fades as it spreads; where
// rings overlap they interfere. Uses iTime -> animated. Calm and nocturnal.
//
//   prism-bg --shader examples/shaders/ripples.frag
//
// Shaded in shared cluster space: one continuous water surface across a
// multi-monitor desktop, at constant per-monitor drop density. Drops live on a
// PCG-hashed grid (one per cell, jittered), so neighbouring outputs don't share
// a seam or repeat. Output is extended-linear with sRGB primaries (1.0 =
// reference white); the brightest crests glint into HDR.
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

// PCG: four decorrelated randoms per drop cell (position jitter, rate, phase).
const float U32 = 2.3283064e-10;  // 1 / 2^32
uvec4 pcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}
vec4 cellRand4(vec2 cell) {
    uvec2 q = uvec2(ivec2(cell));
    return vec4(pcg4d(uvec4(q, q.x ^ 0x9e3779b9u, q.y ^ 0x85ebca6bu))) * U32;
}

const float FREQ = 16.0;   // ripple wavenumber (rings per unit)
const float WAVESPEED = 0.6;  // how fast a ring front expands (units / sec)

void main() {
    // Global cluster space: one continuous water surface across outputs.
    vec2 local = fragCoord / pc.iResolution;
    vec2 gv = (pc.iOutputOffset + local * pc.iOutputSize) / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 P = gv * vec2(aspect, 1.0) * 3.0;   // ~3 drop cells per unit
    float t = pc.iTime;

    // Sum the wave height from drops in the neighbouring cells. One drop per
    // cell, re-falling on its own rate; the ring exists only inside the radius
    // it has had time to reach, and decays with age and distance. The scan is
    // +/-2 cells and each drop's contribution is windowed to zero by d = 2.0 —
    // inside the guaranteed-covered radius — so no ring is ever cut abruptly at
    // a cell boundary (which would draw a visible grid seam).
    vec2 baseCell = floor(P);
    float h = 0.0;
    for (int j = -2; j <= 2; j++)
    for (int i = -2; i <= 2; i++) {
        vec2 cid = baseCell + vec2(i, j);
        vec4 r = cellRand4(cid);
        vec2 center = cid + 0.5 + (r.xy - 0.5) * 0.7;
        float rate = 0.10 + 0.10 * r.z;             // drops per second (slow, calm rain)
        float age = fract(t * rate + r.w) / rate;   // seconds since this drop fell

        float d = length(P - center);
        float front = WAVESPEED * age;              // current ring radius
        // Both windows below are exactly 0 past the ring front / the d = 2.0
        // seam guard, so skip the sin/exp for cells that can't contribute
        // (always the scan's corners, and most cells while a ring is young).
        if (d >= front || d >= 2.0) continue;
        float wave = sin(d * FREQ - age * (FREQ * WAVESPEED));
        float inside = smoothstep(front, front - 0.30, d);   // only where the ring has reached
        float decay = exp(-(age * 1.4 + d * 1.1));           // ripple dies with age + spread
        float win = smoothstep(2.0, 1.4, d);                 // -> 0 before the scan edge: no grid seam
        h += wave * inside * decay * win;
    }

    // Night water: deep blue-black, ripple crests glow cool, troughs sink darker.
    vec3 col = vec3(0.010, 0.022, 0.045);
    col += vec3(0.15, 0.45, 0.75) * max(h, 0.0);
    col += vec3(0.70, 0.90, 1.0) * smoothstep(0.45, 0.95, h) * 0.6 * headroom();
    col *= 1.0 + 0.6 * min(h, 0.0);   // troughs darken the surface

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
