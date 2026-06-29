//!luminance dark
// Northern lights over a dark, star-flecked sky. Uses iTime → animated.
// Curtains of green/teal/blue drift and shimmer beneath a faint cyan crown.
// Moderately cheap (a couple of fbm evaluations per pixel).
//
//   prism-bg --shader examples/shaders/aurora.frag
//
// Shaded in shared cluster space: the sky, stars, and curtains are continuous
// across a multi-monitor desktop, not a copy per output. Output is extended-
// linear with sRGB primaries (1.0 = reference white); prism-bg tags the surface
// and the compositor converts.
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

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

// PCG bit-hash (full period) for the stars. hash21 above is fine for the
// interpolated curtain fbm, but the stars are sparse points gated on a marching
// integer cell id, where fract(p*k)'s short period made them cluster and repeat
// (the original aurora's tell). pcg scatters them properly.
const float U32 = 2.3283064e-10;  // 1 / 2^32
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (s >> 22u) ^ s;
}
float starHash(vec2 cell) {
    uvec2 q = uvec2(ivec2(cell));
    return float(pcg(q.x ^ pcg(q.y))) * U32;
}

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) {
        s += a * vnoise(p);
        p = p * 2.02 + 7.0;
        a *= 0.5;
    }
    return s;
}

void main() {
    // Global cluster space: continuous sky, stars, and curtains across outputs.
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = g / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    float gx = gv.x * aspect;                       // aspect-correct horizontal
    float t = pc.iTime * 0.08;

    // Night-sky vertical gradient: indigo at the horizon to near-black up top.
    vec3 col = mix(vec3(0.015, 0.03, 0.07), vec3(0.0, 0.005, 0.02), gv.y);

    // Sparse stars (full-period PCG hash → no repeating clusters), only in the
    // upper sky so the aurora reads clearly below.
    vec2 id = floor(g * (320.0 / max(pc.iGlobalResolution.y, 1.0)));
    float star = pow(starHash(id), 220.0);
    star *= smoothstep(0.35, 0.95, gv.y);
    col += vec3(0.8, 0.85, 1.0) * star * headroom();

    // Two drifting curtains. Each is a horizontal noise band whose height is
    // modulated by fbm; brightness falls off above and below the band.
    for (int k = 0; k < 2; k++) {
        float fk = float(k);
        float drift = t * (1.0 + fk * 0.6);
        float band = fbm(vec2(gx * 2.0 + drift, fk * 11.0));
        float height = 0.32 + 0.22 * fk + 0.16 * band;
        float thick = 0.12 + 0.05 * fk;
        float curtain = smoothstep(thick, 0.0, abs(gv.y - height));

        // Vertical filaments shimmering within the curtain.
        float fil = fbm(vec2(gx * 9.0, gv.y * 5.0 - drift * 6.0));
        curtain *= 0.55 + 0.65 * fil;

        vec3 tint = mix(vec3(0.05, 0.7, 0.35), vec3(0.1, 0.45, 0.8), fk);
        col += tint * curtain * 0.7 * headroom();
    }

    // Faint cyan crown skimming the top of the higher curtain — in-family with
    // the green/teal/blue curtains (replaces the old magenta fringe).
    float crown = smoothstep(0.6, 0.78, gv.y) * (1.0 - smoothstep(0.78, 0.95, gv.y));
    col += vec3(0.12, 0.55, 0.5) * crown * 0.14 * headroom();

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
