//!luminance dark
// Constellation network — parallax layers of drifting nodes wired together with
// glowing lines, falling endlessly toward the centre of the desktop. Each layer
// zooms and crossfades into the next for infinite depth; the whole web slowly
// rotates. Uses iTime -> animated.
//
//   prism-bg --shader examples/shaders/constellation.frag
//
// Adapted from Martijn Steinrucken's "The Universe Within" (BigWIngs, CC-BY-NC-
// SA) into prism-bg's conventions: shaded in shared CLUSTER space so the zoom
// and rotation are centred on the whole desktop and the network is one
// continuous scene across every output (not a copy per monitor); a coordinated
// icy palette in place of the original hue cycling; a PCG node hash (the global
// coords push cell ids large); HDR node sparkles; no audio and no periodic
// fade-out. Output is extended-linear with sRGB primaries (1.0 = reference
// white); node cores glint into HDR headroom.
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

#define S(a, b, t) smoothstep(a, b, t)
#define NUM_LAYERS 3.

// PCG node hash (full period). The network is shaded in global cluster space, so
// cell ids span the whole desktop; a fract(p*k) hash would band at that range.
// The 128x fixed-point keeps the per-layer fractional id offset (n) distinct.
const float U32 = 2.3283064e-10;  // 1 / 2^32
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (s >> 22u) ^ s;
}
float N21(vec2 p) {
    uvec2 q = uvec2(ivec2(floor(p * 128.0)));
    return float(pcg(q.x ^ pcg(q.y))) * U32;
}

// A node's position: it orbits its cell on its own two frequencies.
vec2 GetPos(vec2 id, vec2 offs, float t) {
    float n = N21(id + offs);
    float n1 = fract(n * 10.0);
    float n2 = fract(n * 100.0);
    float a = t + n;
    return offs + vec2(sin(a * n1), cos(a * n2)) * 0.4;
}

// Distance from point p to segment a-b.
float df_line(vec2 a, vec2 b, vec2 p) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// A glowing wire between two nodes, fading with their separation (and a subtle
// highlight at the "ideal" length) — the original's character, kept intact.
float line(vec2 a, vec2 b, vec2 uv) {
    float r1 = 0.04, r2 = 0.01;
    float d = df_line(a, b, uv);
    float d2 = length(a - b);
    float fade = S(1.5, 0.5, d2);
    fade += S(0.05, 0.02, abs(d2 - 0.75));
    return S(r1, r2, d) * fade;
}

// One depth layer: wire the 3x3 neighbourhood of nodes into a web and glint the
// nodes. Returns (line coverage, sparkle intensity) kept separate so they can be
// coloured independently.
vec2 NetLayer(vec2 st, float n, float t) {
    vec2 id = floor(st) + n;
    st = fract(st) - 0.5;

    vec2 p[9];
    int idx = 0;
    for (float y = -1.0; y <= 1.0; y++)
        for (float x = -1.0; x <= 1.0; x++)
            p[idx++] = GetPos(id, vec2(x, y), t);

    float m = 0.0, sparkle = 0.0;
    for (int i = 0; i < 9; i++) {
        m += line(p[4], p[i], st);

        float d = length(st - p[i]);
        float s = 0.005 / (d * d);
        s *= S(1.0, 0.7, d);
        float pulse = sin((fract(p[i].x) + fract(p[i].y) + t) * 3.0) * 0.4 + 0.6;
        pulse = pow(pulse, 30.0);   // sharper, briefer => less frequent flashes
        sparkle += s * pulse;
    }

    // Close the square so the web reads as a lattice, not just a star burst.
    m += line(p[1], p[3], st);
    m += line(p[1], p[5], st);
    m += line(p[7], p[5], st);
    m += line(p[7], p[3], st);

    return vec2(m, sparkle);
}

void main() {
    // Global cluster space: zoom and rotation centred on the whole desktop, so
    // the network is one continuous scene across every output.
    vec2 local = fragCoord / pc.iResolution;
    vec2 gpix = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = gpix / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 uv = (gv - 0.5) * vec2(aspect, 1.0);

    float t = pc.iTime * 0.06;            // slow zoom + rotation
    float nodeT = pc.iTime * 0.5;         // slow node orbit + twinkle

    // Slow automatic parallax drift (replaces the original's mouse control).
    vec2 M = vec2(sin(pc.iTime * 0.03), cos(pc.iTime * 0.022)) * 0.15;

    float s = sin(t), c = cos(t);
    mat2 rot = mat2(c, -s, s, c);
    vec2 st = uv * rot;
    M *= rot * 2.0;

    float mLine = 0.0, mSpark = 0.0;
    for (float i = 0.0; i < 1.0; i += 1.0 / NUM_LAYERS) {
        float z = fract(t + i);
        float size = mix(10.0, 1.0, z);                 // smaller zoom-out => sparser
        float fade = S(0.0, 0.6, z) * S(1.0, 0.8, z);   // crossfade as it zooms past
        vec2 nl = NetLayer(st * size - M * z, i, nodeT);
        mLine += fade * nl.x;
        mSpark += fade * nl.y;
    }

    // Icy coordinated palette: deep-blue wires, cyan-white nodes glinting to HDR.
    vec3 lineCol = vec3(0.16, 0.35, 0.62);
    vec3 nodeCol = vec3(0.80, 0.95, 1.0);
    vec3 col = lineCol * mLine * 0.8;
    col += nodeCol * mSpark * 0.28 * headroom();

    // Gentle full-desktop vignette so the centre carries the eye.
    col *= 1.0 - 0.55 * dot(uv, uv);

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
