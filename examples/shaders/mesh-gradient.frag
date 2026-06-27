// Soft multi-point "mesh" gradient — the blurred, blended-color-blob look.
// No iTime, so prism-bg renders one frame and then idles exactly like an
// image wallpaper (zero per-frame GPU cost). Edit the control points below
// to recolor it.
//
//   prism-bg --shader examples/shaders/mesh-gradient.frag
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 iResolution; float iTime; float _pad; } pc;

void main() {
    // Aspect-correct so the blobs stay round on wide outputs.
    vec2 uv = fragCoord / pc.iResolution;
    vec2 p = uv;
    p.x *= pc.iResolution.x / pc.iResolution.y;
    float ar = pc.iResolution.x / pc.iResolution.y;

    // Control points: position (in the same aspect-corrected space), color,
    // and falloff radius. Weighted blend gives smooth gradient meshing.
    vec2 pts[4] = vec2[4](
        vec2(0.15 * ar, 0.20),
        vec2(0.85 * ar, 0.10),
        vec2(0.70 * ar, 0.85),
        vec2(0.25 * ar, 0.70)
    );
    vec3 cols[4] = vec3[4](
        vec3(0.18, 0.05, 0.28), // violet
        vec3(0.04, 0.12, 0.32), // deep blue
        vec3(0.32, 0.10, 0.20), // wine
        vec3(0.05, 0.20, 0.26)  // teal
    );
    float radii[4] = float[4](0.9, 1.0, 0.85, 0.95);

    vec3 col = vec3(0.0);
    float wsum = 0.0;
    for (int i = 0; i < 4; i++) {
        float d = length(p - pts[i]) / radii[i];
        // Smooth inverse-square-ish weight; the +eps avoids a singularity.
        float w = 1.0 / (d * d + 0.05);
        col += cols[i] * w;
        wsum += w;
    }
    col /= wsum;

    // Gentle center-weighted lift so it isn't uniformly flat.
    col *= 0.85 + 0.3 * smoothstep(1.0, 0.0, length(uv - 0.5));

    outColor = vec4(col, 1.0);
}
