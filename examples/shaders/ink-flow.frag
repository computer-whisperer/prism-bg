//!luminance dark
// Ink-in-water — slow monochrome filaments drifting along a divergence-free
// (curl-noise) flow. Velocity is the curl of a smooth scalar potential, so the
// field only swirls and shears and never piles ink into sources or sinks; a dye
// field is smeared along the streamlines (a light line-integral convolution)
// into flow-aligned strands that evolve forever without repeating. Uses iTime
// -> animated. Moderately heavy: the flow is integrated per pixel, so profile
// with --profile-gpu and trim N / potential octaves if needed.
//
//   prism-bg --shader examples/shaders/ink-flow.frag
//
// Shaded in shared cluster space: one continuous body of water across a
// multi-monitor desktop, not a copy per output. Output is extended-linear with
// sRGB primaries (1.0 = reference white); the densest filaments glint into HDR.
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

// PCG value noise (full period). The flow is sampled over the whole desktop, so
// cell ids range large; pcg stays well-distributed where a fract(sin) hash would
// band.
const float U32 = 2.3283064e-10;  // 1 / 2^32
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (s >> 22u) ^ s;
}
float hash(vec2 cell) {
    uvec2 q = uvec2(ivec2(cell));
    return float(pcg(q.x ^ pcg(q.y))) * U32;
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i), b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0)), d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
// Value noise with analytic gradient: returns (value, d/dx, d/dy) in one pass,
// so the flow's curl comes from a single evaluation instead of four
// finite-difference samples — the streamline loop's dominant cost.
vec3 vnoiseD(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    vec2 du = 6.0 * f * (1.0 - f);
    float a = hash(i), b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0)), d = hash(i + vec2(1.0, 1.0));
    float k1 = b - a, k2 = c - a, k3 = a - b - c + d;
    float val = a + k1 * u.x + k2 * u.y + k3 * u.x * u.y;
    vec2 grad = du * vec2(k1 + k3 * u.y, k2 + k3 * u.x);
    return vec3(val, grad);
}
// Two-octave stream-function potential with its gradient (the chain rule carries
// each octave's frequency). Kept low: the flow only needs to be smooth.
vec3 fbm2D(vec2 p) {
    vec3 n0 = vnoiseD(p);
    vec3 n1 = vnoiseD(p * 2.03 + 11.0);
    return vec3(0.65 * n0.x + 0.35 * n1.x,
                0.65 * n0.yz + 0.35 * 2.03 * n1.yz);
}
// Divergence-free velocity from the curl of the (slowly drifting) potential:
// v = (dψ/dy, -dψ/dx).
vec2 flow(vec2 p, float drift) {
    vec3 P = fbm2D(p + vec2(drift, drift * 0.6));
    return vec2(P.z, -P.y);
}

void main() {
    // Global cluster space: continuous, aspect-correct flow across all outputs.
    vec2 local = fragCoord / pc.iResolution;
    vec2 gpix = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = gpix / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 p = (gv - 0.5) * vec2(aspect, 1.0) * 3.0;

    float t = pc.iTime;
    float drift = t * 0.02;   // the flow field evolves slowly
    float dyeT = t * 0.05;    // the dye crawls along the streamlines

    // Smear a dye field backward along the streamline through this pixel. The
    // velocity is normalized so each step is equal arc length (a clean LIC).
    const int N = 10;
    const float STEPLEN = 0.045;
    vec2 pos = p;
    float acc = 0.0, wsum = 0.0;
    for (int i = 0; i < N; i++) {
        vec2 v = flow(pos, drift);
        v /= max(length(v), 1e-4);
        pos -= v * STEPLEN;
        float dye = vnoise(pos * 2.3 + vec2(dyeT, -dyeT * 0.4));
        float wgt = 1.0 - float(i) / float(N);   // taper toward the tail
        acc += dye * wgt;
        wsum += wgt;
    }
    float ink = smoothstep(0.40, 0.72, acc / wsum);   // carve the strands out

    // Cool monochrome ink on near-black water; the densest strands glint to HDR.
    vec3 bg = vec3(0.012, 0.020, 0.040);
    vec3 inkLo = vec3(0.03, 0.09, 0.17);
    vec3 inkHi = vec3(0.45, 0.78, 1.00);
    vec3 col = bg + mix(inkLo, inkHi, ink) * ink;
    col += vec3(0.5, 0.8, 1.0) * smoothstep(0.80, 1.0, ink) * 0.25 * headroom();

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
