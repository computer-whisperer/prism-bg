//!luminance bright
// Caustics — sunlit ripples dancing across a shallow pool floor. Bright aqua
// water webbed with brilliant white light lines that weave and breathe. Uses
// iTime -> animated. A luminous daytime wallpaper.
//
//   prism-bg --shader examples/shaders/caustics.frag
//
// Adapted from the classic iterated-distortion caustic (a widely-reproduced
// Shadertoy technique) into prism-bg's conventions: shaded in shared cluster
// space (continuous, aspect-correct coords) so the pool is one surface across a
// multi-monitor desktop, not a copy per output. Moderately heavy (a short loop
// of trig per pixel) — profile with --profile-gpu and trim ITER if needed.
// Output is extended-linear with sRGB primaries (1.0 = reference white); the
// brightest crests glint into HDR headroom.
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

#define ITER 6

void main() {
    // Global cluster space: one continuous, aspect-correct pool across outputs.
    vec2 local = fragCoord / pc.iResolution;
    vec2 gv = (pc.iOutputOffset + local * pc.iOutputSize) / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 gp = gv * vec2(aspect, 1.0);

    float time = pc.iTime * 0.22 + 23.0;
    // No mod() tiling (that would repeat) — a continuous global coordinate,
    // offset into a lively region of the field.
    vec2 p = gp * 5.0 - 250.0;
    vec2 i = p;
    float c = 1.0;
    float inten = 0.005;

    // Iterated domain distortion: each pass folds the coordinate through itself,
    // accumulating the inverse distance to a shifting interference lattice. The
    // per-pass time scale (some negative) is what makes the web shimmer.
    for (int n = 0; n < ITER; n++) {
        float t = time * (1.0 - 3.5 / float(n + 1));
        i = p + vec2(cos(t - i.x) + sin(t + i.y), sin(t - i.y) + cos(t + i.x));
        c += 1.0 / length(vec2(p.x / (sin(i.x + t) / inten), p.y / (cos(i.y + t) / inten)));
    }
    c /= float(ITER);
    c = 1.17 - pow(c, 1.4);
    float caustic = pow(abs(c), 8.0);

    // Bright aqua water (the teal base lifts the whole field) webbed with white
    // light; the brightest crests push into HDR for a sunlit glint.
    vec3 col = clamp(vec3(caustic) + vec3(0.0, 0.35, 0.5), 0.0, 1.0);
    col += vec3(0.4, 0.85, 1.0) * smoothstep(0.6, 1.0, caustic) * 0.6 * (headroom() - 1.0);

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
