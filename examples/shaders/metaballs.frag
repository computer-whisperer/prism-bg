//!luminance dark
// Gooey metaballs — soft red/orange blobs that orbit and merge into a molten
// lava-lamp mass on a near-black field. Uses iTime → animated. Cheap: a fixed
// loop of inverse-square field contributions, no noise.
//
//   prism-bg --shader examples/shaders/metaballs.frag
//
// Each output keeps its own self-contained, centred lamp, but the orbits are
// phase-shifted by the output's place in the cluster so neighbouring monitors
// don't show an identical clone. Output is extended-linear with sRGB primaries
// (1.0 = reference white).
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

// Molten lava ramp: deep red -> red-orange -> orange-gold. Warm hues only, so
// the mass reads as a lava lamp rather than a full-spectrum blob. s in [0,1].
vec3 lava(float s) {
    s = clamp(s, 0.0, 1.0);
    vec3 dark = vec3(0.20, 0.030, 0.020);
    vec3 mid  = vec3(0.65, 0.160, 0.040);
    vec3 hot  = vec3(1.00, 0.550, 0.150);
    return s < 0.5 ? mix(dark, mid, s * 2.0)
                   : mix(mid, hot, (s - 0.5) * 2.0);
}

void main() {
    vec2 p = (fragCoord - 0.5 * pc.iResolution) / pc.iResolution.y;
    float t = pc.iTime * 0.28;

    // Per-output desync: a stable seed from this output's place in the cluster,
    // so each monitor's lamp runs its own phase instead of an identical clone.
    float seed = fract(sin(dot(pc.iOutputOffset, vec2(0.0123, 0.0071)) + 0.5) * 43758.5453);

    // Accumulate a scalar field; each ball adds r^2 / dist^2.
    const int N = 6;
    float field = 0.0;
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        // Lissajous-ish orbit, each ball on a different frequency pair, with the
        // whole pattern phase-shifted per output.
        vec2 c = 0.55 * vec2(
            sin(t * (0.7 + 0.13 * fi) + fi * 1.7 + seed * 6.2831),
            cos(t * (0.6 + 0.11 * fi) + fi * 2.3 + seed * 9.1300)
        );
        float radius = 0.16 + 0.05 * sin(fi * 1.3 + seed * 3.0);
        vec2 r = p - c;
        field += (radius * radius) / (dot(r, r) + 0.0008);
    }

    // Isosurface, softened so the blobs read as a calm glowing mass rather than
    // high-contrast circles: a wide inside ramp and a gentle warm contour.
    float inside = smoothstep(0.55, 1.40, field);
    float rim = smoothstep(0.75, 1.05, field) * (1.0 - smoothstep(1.05, 1.7, field));

    vec3 bg = vec3(0.020, 0.012, 0.020);
    vec3 body = lava(clamp(0.16 + field * 0.16, 0.0, 1.0));

    vec3 col = mix(bg, body, inside);
    col += vec3(1.0, 0.5, 0.18) * rim * 0.18 * headroom();           // soft warm contour
    // Soft outer glow where the field is rising but not yet inside.
    col += body * smoothstep(0.25, 0.85, field) * 0.16 * headroom();

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
