//!luminance bright
// Zen garden: raked sand shaded as a real relief — straight furrows that bend
// into concentric rings around a handful of stones, each stone casting a soft
// shadow. The only motion is the light itself, wheeling imperceptibly slowly
// around the garden, so shadows and furrow shading migrate over minutes.
// Uses iTime -> animated.
//
//   prism-bg --shader examples/shaders/zen-garden.frag
//
// One garden across the whole cluster. Cheap: the furrow height field is a
// handful of cosines, sampled three times for a finite-difference normal.
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

const float TAU = 6.28318530718;
const float U32 = 2.3283064e-10;  // 1 / 2^32
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (s >> 22u) ^ s;
}
vec4 stoneRand(uint i) {
    uint h = pcg(i * 0x9e3779b9u + 0x85ebca6bu);
    return vec4(pcg(h), pcg(h ^ 0x68bc21ebu), pcg(h ^ 0x02e5be93u), pcg(h ^ 0x967a889bu)) * U32;
}
float hash21(vec2 p) {
    p = fract(p * vec2(157.31, 313.97));
    p += dot(p, p + 26.17);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

const uint NSTONES = 5u;
const float FREQ = 78.0;  // furrows per unit of p-space

// Stone i: xy centre in p-space, z radius. Spread across the width so the
// garden composes on any cluster shape.
vec3 stone(uint i, float aspect) {
    vec4 rnd = stoneRand(i);
    float x = ((float(i) + 0.18 + 0.64 * rnd.x) / float(NSTONES) - 0.5) * aspect * 0.94;
    float y = (rnd.y - 0.5) * 0.72;
    return vec3(x, y, 0.045 + 0.045 * rnd.z);
}

// Raked-sand height: straight furrows yielding to rings near each stone,
// blended by proximity weight so the patterns interleave like real raking.
float sandHeight(vec2 p, float aspect) {
    float s = cos(p.y * FREQ);
    float w = 0.55;  // background rake weight
    for (uint i = 0u; i < NSTONES; i++) {
        vec3 st = stone(i, aspect);
        float d = length(p - st.xy);
        float wi = smoothstep(0.42, st.z, d) * smoothstep(st.z * 0.5, st.z * 1.1, d);
        s = mix(s, cos((d - st.z) * FREQ), wi);
        w = max(w, wi);
    }
    return s;
}

void main() {
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = g / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 p = (gv - 0.5) * vec2(aspect, 1.0);

    // The sun swings through a shallow arc, one pass every ~20 minutes.
    float sunAng = 2.4 + 0.9 * sin(pc.iTime * TAU / 1200.0);
    vec3 L = normalize(vec3(cos(sunAng), sin(sunAng), 1.35));

    // Furrow relief -> finite-difference normal -> soft directional shading.
    float e = 0.008;
    float h = sandHeight(p, aspect);
    float hx = sandHeight(p + vec2(e, 0.0), aspect);
    float hy = sandHeight(p + vec2(0.0, e), aspect);
    vec3 n = normalize(vec3((h - hx) * 0.012 / e, (h - hy) * 0.012 / e, 1.0));
    float diff = max(dot(n, L), 0.0);

    vec3 sand = mix(vec3(0.72, 0.66, 0.55), vec3(0.88, 0.83, 0.72), diff);
    sand *= 0.96 + 0.04 * h;                  // valleys sit a touch darker
    sand *= 0.94 + 0.11 * hash21(g);          // fine grain
    sand *= 0.93 + 0.07 * noise(g * 0.012);   // smooth tonal drift

    vec3 col = sand;
    for (uint i = 0u; i < NSTONES; i++) {
        vec3 st = stone(i, aspect);
        vec4 rnd = stoneRand(i);
        // Gently elliptical pebble.
        vec2 q = (p - st.xy) * vec2(1.0, 1.12 + 0.2 * rnd.w);
        float d = length(q);

        // Contact shadow cast away from the light, and a dark contact ring.
        float sh = smoothstep(st.z * 1.75, st.z * 0.4, length(q + L.xy * st.z * 0.55));
        col *= 1.0 - 0.38 * sh;

        // Stone shading: hemispherical normal, matte grey with lichen speckle.
        float in_ = smoothstep(st.z, st.z - 0.004, d);
        if (in_ > 0.0) {
            float z = sqrt(max(1.0 - (d * d) / (st.z * st.z), 0.0));
            vec3 sn = normalize(vec3(q / st.z, z + 0.35));
            float sd = max(dot(sn, L), 0.0);
            vec3 rock = mix(vec3(0.16, 0.16, 0.17), vec3(0.52, 0.50, 0.48), sd);
            rock *= 0.86 + 0.20 * noise(g * 0.45 + rnd.xy * 90.0);  // granite grain
            rock *= 0.92 + 0.16 * noise(g * 0.06 + rnd.zw * 70.0);  // broad mottle
            col = mix(col, rock, in_);
        }
    }

    // Soft daylight falloff toward the garden's edges.
    float vignette = smoothstep(1.55, 0.35, length(p));
    col *= 0.86 + 0.14 * vignette;

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
