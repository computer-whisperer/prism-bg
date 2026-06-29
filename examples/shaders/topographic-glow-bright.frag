//!luminance bright
// Topographic glow (bright): contour lines slide over a living height field,
// with subtle color shifts and bright survey marks. Uses iTime -> animated.
//
// The bright daytime counterpart to topographic-glow.frag: a near-black field
// crossed by punchy, high-contrast survey lines and marks. topographic-glow.frag
// drives the same map much darker — a quiet near-black contour field for a dark
// room. Pair them in a --list with --dark-hours to swap by time of day.
//
//   prism-bg --shader examples/shaders/topographic-glow-bright.frag
//
// Coordinates are in global cluster space so the map is continuous across
// outputs and keeps the same apparent scale regardless of per-monitor DPI.
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
    p = fract(p * vec2(61.17, 289.93));
    p += dot(p, p + 23.45);
    return fract(p.x * p.y);
}

// PCG bit-hash (full period) for the survey-mark placement. The fract(p*k)
// hash above is fine for the interpolated height field, but sampled with
// incrementing integer cell ids it has a short period — sparse marks gated
// off it form a visibly repeating constellation across the workspace.
const float U32 = 2.3283064e-10;  // 1 / 2^32
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (s >> 22u) ^ s;
}
// Two decorrelated randoms in [0,1) keyed on an integer cell.
vec2 cellRand2(vec2 cell) {
    uvec2 q = uvec2(ivec2(cell));
    uint h = pcg(q.x ^ pcg(q.y));
    return vec2(pcg(h), pcg(h ^ 0x9e3779b9u)) * U32;
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

float fbm(vec2 p) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < 6; i++) {
        s += a * noise(p);
        p = mat2(1.78, 0.84, -0.84, 1.78) * p + vec2(5.0, 11.0);
        a *= 0.5;
    }
    return s;
}

void main() {
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 p = (g - 0.5 * pc.iGlobalResolution) / 260.0;
    float t = pc.iTime * 0.030;

    float h = fbm(p + vec2(t, -t * 0.55));
    h += 0.35 * fbm(p * 2.2 - vec2(t * 1.7, t));

    float contourPhase = h * 11.0 - t * 1.6;
    float contour = abs(fract(contourPhase) - 0.5);
    float major = abs(fract(contourPhase * 0.2) - 0.5);
    float line = smoothstep(0.075, 0.018, contour);
    float majorLine = smoothstep(0.060, 0.012, major);

    vec3 low = vec3(0.020, 0.028, 0.035);
    vec3 high = vec3(0.055, 0.038, 0.045);
    vec3 col = mix(low, high, local.y + h * 0.25);

    vec3 cyan = vec3(0.08, 0.75, 0.72);
    vec3 amber = vec3(0.95, 0.48, 0.14);
    vec3 magenta = vec3(0.75, 0.13, 0.55);
    vec3 tint = mix(cyan, amber, smoothstep(0.2, 1.2, h));
    tint = mix(tint, magenta, 0.25 + 0.25 * sin(p.x * 0.7 + pc.iTime * 0.2));

    col += tint * line * 0.22 * headroom();
    col += vec3(1.0, 0.88, 0.58) * majorLine * 0.24 * headroom();

    // Survey marks at deterministic cell centers, pulsing as contour bands pass.
    vec2 cell = floor(p * 0.7);
    vec2 f = fract(p * 0.7) - 0.5;
    vec2 rnd = cellRand2(cell);
    float gate = step(0.88, rnd.x);
    float mark = smoothstep(0.055, 0.0, abs(length(f) - 0.16));
    mark *= gate * (0.45 + 0.55 * sin(pc.iTime * 0.8 + rnd.y * 6.28318));
    col += vec3(0.7, 0.95, 1.0) * mark * 0.35 * headroom();

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
