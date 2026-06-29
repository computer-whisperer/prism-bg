//!luminance bright
// Molten cellular glass with glowing edges — animated Voronoi. Warm ember /
// copper / gold cells breathe as their feature points drift on sine orbits,
// with hot gold borders. Uses iTime → animated. Moderately heavy: two
// neighborhood passes per pixel (nearest cell, then the distance to its
// borders), the classic two-pass Voronoi-edge approach.
//
//   prism-bg --shader examples/shaders/voronoi.frag
//
// Original implementation of a well-known technique (Inigo Quilez's Voronoi
// distance), shaded in shared cluster space so the cells tile continuously
// across a multi-monitor desktop instead of cloning per output. Output is
// extended-linear with sRGB primaries (1.0 = white).
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

// PCG bit-hash (full period). The cells are shaded in global cluster space, so
// cell ids range over the whole desktop; the usual fract(sin(p)) hash bands at
// those magnitudes, while pcg stays well-distributed.
const float U32 = 2.3283064e-10;  // 1 / 2^32
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (s >> 22u) ^ s;
}
vec2 hash22(vec2 p) {
    uvec2 q = uvec2(ivec2(p));
    uint h = pcg(q.x ^ pcg(q.y));
    return vec2(pcg(h), pcg(h ^ 0x9e3779b9u)) * U32;
}

// Molten ramp: ember -> copper -> gold, confined to warm hues (no full-spectrum
// hue walk). s in [0,1].
vec3 warmRamp(float s) {
    s = clamp(s, 0.0, 1.0);
    vec3 ember  = vec3(0.16, 0.055, 0.025);
    vec3 copper = vec3(0.43, 0.180, 0.070);
    vec3 gold   = vec3(0.85, 0.520, 0.180);
    return s < 0.5 ? mix(ember, copper, s * 2.0)
                   : mix(copper, gold, (s - 0.5) * 2.0);
}

void main() {
    // Global cluster space: the cells tile continuously across a multi-monitor
    // desktop at constant per-monitor density instead of cloning per output.
    vec2 local = fragCoord / pc.iResolution;
    vec2 g0 = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = g0 / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 uv = (gv - 0.5) * vec2(aspect, 1.0);   // centred, aspect-correct, continuous
    vec2 p = uv * 5.0;
    float t = pc.iTime * 0.18;

    vec2 ip = floor(p), fp = fract(p);

    // Pass 1: nearest feature point and which cell it lives in.
    float md = 8.0;
    vec2 mr, mg;
    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
        vec2 g = vec2(i, j);
        vec2 o = hash22(ip + g);
        o = 0.5 + 0.5 * sin(t + 6.28318 * o);
        vec2 r = g + o - fp;
        float d = dot(r, r);
        if (d < md) { md = d; mr = r; mg = g; }
    }

    // Pass 2: distance to the borders with neighboring cells (perpendicular
    // bisector between the two nearest feature points).
    float me = 8.0;
    for (int j = -2; j <= 2; j++)
    for (int i = -2; i <= 2; i++) {
        vec2 g = mg + vec2(i, j);
        vec2 o = hash22(ip + g);
        o = 0.5 + 0.5 * sin(t + 6.28318 * o);
        vec2 r = g + o - fp;
        if (dot(mr - r, mr - r) > 0.00001)
            me = min(me, dot(0.5 * (mr + r), normalize(r - mr)));
    }

    // Cell fill: a warm molten tint, each cell on its own slow phase through the
    // ember->copper->gold ramp (a smooth oscillation, so no hue-wrap pop).
    vec2 cellId = ip + mg;
    float ph = hash22(cellId).x;
    float cellVar = 0.5 + 0.5 * sin(t * 0.8 + 6.28318 * ph);
    vec3 col = warmRamp(0.12 + 0.65 * cellVar);
    // Brighten toward each cell's feature point, dim toward its borders.
    col *= 0.30 + 0.45 * smoothstep(0.0, 0.6, sqrt(md));

    // Glowing molten borders, pushed into HDR headroom so they read as hot.
    float edge = smoothstep(0.04, 0.0, me);
    col += vec3(1.0, 0.62, 0.22) * edge * 0.7 * headroom();

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
