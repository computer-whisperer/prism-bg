//!luminance dark
// Snowfall: quiet snow on a winter night. Four parallax layers of flakes fall
// at their own pace and sway on individual phases; nearer layers are larger,
// brighter, and faster. A low moon haze lights the upper sky and the ground
// returns a faint snow-glow from below. Uses iTime -> animated.
//
//   prism-bg --shader examples/shaders/snowfall.frag
//
// Flakes live in shared cluster space, so they cross monitor bezels intact.
// Cheap: four grid lookups per pixel, PCG-hashed cells, no noise octaves.
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
vec4 cellRand4(vec2 cell, uint layer) {
    uvec2 q = uvec2(ivec2(cell) + 0x8000);
    uint h = pcg(q.x ^ pcg(q.y ^ pcg(layer * 0x9e3779b9u)));
    return vec4(pcg(h), pcg(h ^ 0x68bc21ebu), pcg(h ^ 0x02e5be93u), pcg(h ^ 0x967a889bu)) * U32;
}
float hash21(vec2 p) {
    p = fract(p * vec2(167.17, 353.51));
    p += dot(p, p + 19.73);
    return fract(p.x * p.y);
}

// One flake layer: a jittered grid falling through cluster space. `depth` 0
// is the farthest layer, 1 the nearest.
float flakes(vec2 g, uint layer, float depth) {
    float cellPx = mix(90.0, 260.0, depth);           // nearer -> sparser, bigger
    float fall = mix(18.0, 78.0, depth);              // px/s, nearer -> faster
    vec2 uv = vec2(g.x, g.y + pc.iTime * fall) / cellPx;
    vec2 cell = floor(uv);
    vec4 rnd = cellRand4(cell, layer);
    if (rnd.w < 0.42) return 0.0;                      // not every cell snows

    // Flake position inside the cell, swaying on its own phase. Jitter plus
    // sway stays well inside the cell so the halo never clips at its border.
    vec2 c = 0.5 + 0.28 * (rnd.xy - 0.5) * 2.0;
    c.x += 0.10 * sin(pc.iTime * (0.4 + 0.5 * rnd.z) + rnd.y * TAU);
    float d = length((uv - cell - c) * cellPx);

    float radius = mix(1.2, 4.4, depth) * (0.7 + 0.6 * rnd.z);
    float core = smoothstep(radius, radius * 0.25, d);
    float halo = exp(-d * d / (radius * radius * 9.0)) * 0.35;
    return (core + halo) * mix(0.35, 1.0, depth);
}

void main() {
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gres = max(pc.iGlobalResolution, vec2(1.0));
    vec2 gv = g / gres;
    float aspect = gres.x / gres.y;

    // Winter night: blue-black zenith, a cold moon haze high on one side,
    // and the dim glow snow-covered ground gives back to the sky.
    vec3 col = mix(vec3(0.014, 0.022, 0.042), vec3(0.036, 0.052, 0.086),
                   smoothstep(0.9, 0.0, gv.y));
    vec2 moon = (vec2(0.72, 0.86) - gv) * vec2(aspect, 1.0);
    col += vec3(0.16, 0.19, 0.26) * exp(-dot(moon, moon) * 4.5);
    col += vec3(0.05, 0.06, 0.09) * smoothstep(0.30, 0.0, gv.y);

    // Far to near; the nearest layer drifts sideways slightly with the wind.
    vec3 snow = vec3(0.83, 0.88, 0.97);
    float a = 0.0;
    for (uint i = 0u; i < 4u; i++) {
        float depth = float(i) / 3.0;
        vec2 gw = g + vec2(pc.iTime * mix(2.0, 9.0, depth), 0.0);
        a += flakes(gw + vec2(float(i) * 771.0, 0.0), i, depth);
    }
    col += snow * min(a, 1.4) * 0.85;

    // Grain keeps the long night gradient from banding.
    col += (hash21(g) - 0.5) * 0.004;

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
