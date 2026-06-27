// Gooey metaballs — soft blobs that orbit and merge into a single lava-lamp
// mass, isosurface shaded with an IQ palette. Uses iTime → animated. Cheap:
// a fixed loop of inverse-square field contributions, no noise.
//
//   prism-bg --shader examples/shaders/metaballs.frag
//
// Output is extended-linear with sRGB primaries (1.0 = reference white).
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 iResolution; float iTime; float _pad; } pc;

vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.45, 0.5);
    vec3 b = vec3(0.5, 0.45, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.0, 0.10, 0.20);
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    vec2 p = (fragCoord - 0.5 * pc.iResolution) / pc.iResolution.y;
    float t = pc.iTime * 0.4;

    // Accumulate a scalar field; each ball adds r^2 / dist^2.
    const int N = 6;
    float field = 0.0;
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        // Lissajous-ish orbit, each ball on a different frequency pair.
        vec2 c = 0.55 * vec2(
            sin(t * (0.7 + 0.13 * fi) + fi * 1.7),
            cos(t * (0.6 + 0.11 * fi) + fi * 2.3)
        );
        float radius = 0.16 + 0.05 * sin(fi * 1.3);
        vec2 r = p - c;
        field += (radius * radius) / (dot(r, r) + 0.0008);
    }

    // Isosurface: field ~1 is the blob boundary.
    float inside = smoothstep(0.8, 1.2, field);
    float rim = smoothstep(0.7, 1.0, field) * (1.0 - smoothstep(1.0, 1.6, field));

    vec3 bg = vec3(0.02, 0.015, 0.04);
    vec3 body = palette(0.15 + field * 0.12 + t * 0.05) * 0.5;

    vec3 col = mix(bg, body, inside);
    col += palette(0.5) * rim * 0.35; // brighter rim/contour
    // Soft outer glow where the field is rising but not yet inside.
    col += body * smoothstep(0.3, 0.8, field) * 0.12;

    outColor = vec4(col, 1.0);
}
