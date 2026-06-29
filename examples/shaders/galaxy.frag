//!luminance dark
// Spiral galaxy spanning the whole desktop — ONE continuous scene across every
// monitor, not a copy per output. Slowly winding log-spiral arms, a blown-out
// HDR core, dust lanes and a continuous star field; the galaxy's centre sits at
// the middle of your cluster and each output renders its own slice. Uses iTime
// → animated.
//
//   prism-bg --shader examples/shaders/galaxy.frag
//
// MULTI-MONITOR: the trick is to shade in CLUSTER space, not per-output space.
// iOutputOffset / iOutputSize place this output inside the cluster bounding box
// iGlobalResolution (all in y-up logical pixels). The global coordinate of a
// pixel is  g = iOutputOffset + (fragCoord/iResolution) * iOutputSize, and
// g / iGlobalResolution is 0..1 across the whole workspace. Everything below is
// a function of that, so the picture is seamless across monitor seams. On a lone
// output the fields collapse to offset 0, size == global and it just fills the
// screen.
//
// Output is extended-linear with sRGB primaries (1.0 = reference white); the
// core is pushed into HDR headroom so it reads as a genuine bright source.
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

const float PI = 3.14159265;
const float TAU = 6.28318531;

// Highlight headroom above diffuse white (>= 1.0; 1.0 on SDR).
float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

// PCG bit-hash (full period) for the star field. hash21 above is fine for the
// interpolated nebula/dust noise, but the stars are sparse points gated on a
// marching integer cell id, where the short period of fract(p*k) makes the
// constellation repeat across the desktop (the seam-free cluster coords don't
// help — it's the hash that repeats). pcg4d gives four decorrelated randoms
// per cell in one pass: brightness, twinkle rate, twinkle phase, tint.
const float U32 = 2.3283064e-10;  // 1 / 2^32
uvec4 pcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}
vec4 cellRand4(vec2 cell, uint layer) {
    uvec2 q = uvec2(ivec2(cell));
    uvec4 s = uvec4(q.x, q.y, q.x ^ (layer * 0x9e3779b9u), q.y ^ (layer + 0x85ebca6bu));
    return vec4(pcg4d(s)) * U32;
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
        p = p * 2.03 + 11.0;
        a *= 0.5;
    }
    return s;
}

// IQ cosine palette, warm-to-cool for the disk tint.
vec3 palette(float t) {
    return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.12, 0.30)));
}

void main() {
    // Cluster-space coordinates: continuous across every output (see header).
    vec2 g  = pc.iOutputOffset + (fragCoord / pc.iResolution) * pc.iOutputSize;
    vec2 gn = g / pc.iGlobalResolution;                       // 0..1 over the desktop
    float aspect = pc.iGlobalResolution.x / pc.iGlobalResolution.y;
    vec2 p = (gn - 0.5) * vec2(aspect, 1.0);                  // centred, aspect-correct

    float t = pc.iTime;

    // --- Deep-space backdrop: a faint, slowly drifting nebula wash. ---
    float neb = fbm(p * 2.2 + vec2(0.0, t * 0.01));
    vec3 col = mix(vec3(0.004, 0.006, 0.013), vec3(0.02, 0.012, 0.03), neb);

    // --- Continuous star field (shaded in cluster pixels so it never repeats
    //     or jumps at a monitor seam). Two layers for a little depth. ---
    for (int L = 0; L < 2; L++) {
        float cell = (L == 0) ? 7.0 : 13.0;
        vec2 id = floor(g / cell);
        vec4 r = cellRand4(id, uint(L));
        float star = pow(r.x, 240.0);                        // very sparse
        // Twinkle, each star on its own rate and phase.
        float tw = 0.6 + 0.4 * sin(t * (1.0 + 3.0 * r.y) + r.z * TAU);
        vec3 sc3 = mix(vec3(0.7, 0.8, 1.0), vec3(1.0, 0.9, 0.8), r.w);
        col += sc3 * star * tw * (L == 0 ? 0.9 : 0.5);
    }

    // --- Galaxy geometry in polar coordinates. ---
    float r = length(p);
    float a = atan(p.y, p.x);
    float spin = t * 0.035;                                  // slow rigid drift
    float lr = log(r + 0.05);

    // Log-spiral arms: an integer arm count keeps cos() continuous across the
    // atan branch cut, so there's no seam at the −x axis. The log term sets the
    // winding tightness; a touch of fbm breaks the arms into clumps.
    const float ARMS = 2.0;
    float swirl = 5.5;
    float arm = 0.5 + 0.5 * cos(ARMS * a - swirl * lr + spin * 6.0);
    arm = pow(arm, 2.2);
    float clump = 0.55 + 0.9 * fbm(vec2(a * 1.6, lr * 3.0) + spin);

    // Disk brightness: exponential radial falloff, modulated by the arms.
    float disk = exp(-r * 3.0);
    float arms = disk * mix(0.10, 1.0, arm) * clump;

    // Dust lanes: dark fbm filaments riding along the arms, subtracted.
    float dust = smoothstep(0.45, 0.85, fbm(vec2(a * 2.0, lr * 4.0) - spin));
    arms *= 1.0 - 0.6 * dust;

    // Disk colour: cool blue-white in the outskirts warming toward the bulge.
    vec3 diskCol = palette(0.62 - r * 0.5);
    col += diskCol * arms * 1.6;

    // HII regions: pink star-forming knots scattered along the bright arms.
    float knot = pow(clump * arm, 3.0) * disk;
    col += vec3(1.0, 0.35, 0.5) * knot * 0.8;

    // --- Central bulge + blown-out HDR core. ---
    float bulge = exp(-r * 9.0);
    col += vec3(1.0, 0.82, 0.55) * bulge * 1.2;
    float core = exp(-r * r * 1400.0);
    col += vec3(1.0, 0.95, 0.85) * core * headroom();

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
