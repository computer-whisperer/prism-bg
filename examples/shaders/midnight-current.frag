//!luminance dark
// Midnight current: luminous flow lines drifting through deep blue-black water,
// with sparse mica-like flecks catching the current. Uses iTime -> animated.
//
//   prism-bg --shader examples/shaders/midnight-current.frag
//
// Coordinates are in global cluster space so the currents and flecks move as
// one field across the whole desktop.
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
    p = fract(p * vec2(113.17, 415.29));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

const float U32 = 2.3283064e-10;  // 1 / 2^32
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (s >> 22u) ^ s;
}
vec3 cellRand3(vec2 cell) {
    uvec2 q = uvec2(ivec2(cell));
    uint h = pcg(q.x ^ pcg(q.y));
    return vec3(pcg(h), pcg(h ^ 0x9e3779b9u), pcg(h ^ 0x85ebca6bu)) * U32;
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

// Octave count per call site: the domain-warp and depth fields are sampled at
// low frequency where octaves past the third are sub-pixel, so only the
// filament displacement pays for four.
float fbm(vec2 p, int oct) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < oct; i++) {
        s += a * noise(p);
        p = mat2(1.52, 1.08, -1.08, 1.52) * p + vec2(6.0, 17.0);
        a *= 0.5;
    }
    return s;
}

float filament(vec2 p, float phase, float width) {
    float y = p.y + 0.35 * fbm(p * 0.70 + phase, 4) + 0.12 * sin(p.x * 2.0 + phase);
    float f = abs(fract(y) - 0.5);
    return smoothstep(width, 0.0, f);
}

void main() {
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = g / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 p = (gv - 0.5) * vec2(aspect, 1.0);

    float t = pc.iTime * 0.055;
    vec2 q = p * 5.0;
    q += vec2(0.55 * fbm(q * 0.45 + vec2(t, -t), 3), 0.35 * fbm(q * 0.50 - vec2(t * 1.4, t), 3));

    float f1 = filament(q + vec2(t * 1.7, 0.0), t * 4.0, 0.055);
    float f2 = filament(q * 1.35 + vec2(-t * 1.2, 3.7), -t * 3.0, 0.040);
    float f3 = filament(q * 0.72 + vec2(t * 0.5, -2.1), t * 2.0, 0.075);
    float flow = clamp(f1 * 0.65 + f2 * 0.50 + f3 * 0.35, 0.0, 1.0);

    vec3 col = mix(vec3(0.003, 0.007, 0.014), vec3(0.012, 0.035, 0.052), gv.y);
    float depth = fbm(q * 0.22 + vec2(0.0, t), 3);
    col += vec3(0.00, 0.10, 0.15) * depth * 0.35;

    vec3 teal = vec3(0.08, 0.84, 0.76);
    vec3 blue = vec3(0.20, 0.40, 1.0);
    vec3 violet = vec3(0.58, 0.22, 0.86);
    vec3 tint = mix(mix(teal, blue, 0.5 + 0.5 * sin(q.x * 0.4)), violet, 0.22 + 0.20 * depth);
    col += tint * flow * (0.18 + 0.30 * smoothstep(0.55, 1.0, flow)) * headroom();

    // Sparse flecks advect upward through the same global field.
    vec2 grid = vec2(g.x * 0.014 + 0.5 * sin(t), g.y * 0.014 - pc.iTime * 0.08);
    vec2 cell = floor(grid);
    vec2 fp = fract(grid);
    vec3 rnd = cellRand3(cell);
    float fleck = smoothstep(0.040, 0.0, length(fp - rnd.xy)) * step(0.962, rnd.z);
    fleck *= 0.5 + 0.5 * sin(pc.iTime * (1.2 + rnd.x * 2.0) + rnd.y * 6.28318);
    col += vec3(0.75, 0.95, 1.0) * fleck * headroom();

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
