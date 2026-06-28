// Ember ridge: layered procedural mountains with slow heat shimmer and HDR
// glints along the skyline. Uses iTime -> animated wallpaper.
//
//   prism-bg --shader examples/shaders/ember-ridge.frag
//
// The terrain is built in shared cluster space, so the ridge line continues
// across a multi-monitor desktop instead of restarting on each output.
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
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
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

float fbm(vec2 p) {
    float s = 0.0;
    float a = 0.55;
    for (int i = 0; i < 5; i++) {
        s += a * noise(p);
        p = p * 2.04 + vec2(17.0, 9.0);
        a *= 0.5;
    }
    return s;
}

float ridge(float x, float layer, float t) {
    float slow = t * (0.018 + layer * 0.006);
    float h = fbm(vec2(x * (0.85 + layer * 0.22) + slow, layer * 13.7));
    h += 0.35 * fbm(vec2(x * 2.4 - slow * 2.0, layer * 31.1));
    return 0.18 + layer * 0.14 + h * (0.18 + layer * 0.035);
}

void main() {
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 uv = g / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    float x = (uv.x - 0.5) * aspect * 3.0;

    vec3 sky = mix(vec3(0.012, 0.018, 0.045), vec3(0.11, 0.055, 0.035), local.y);
    vec3 col = sky;

    // Distant heat bands.
    float haze = fbm(vec2(x * 1.5 + pc.iTime * 0.03, local.y * 8.0));
    col += vec3(0.28, 0.10, 0.04) * haze * smoothstep(0.15, 0.75, local.y) * 0.18;

    for (int i = 0; i < 4; i++) {
        float layer = float(i);
        float h = ridge(x, layer, pc.iTime);
        float body = smoothstep(h + 0.015, h - 0.015, local.y);
        float edge = smoothstep(0.018, 0.0, abs(local.y - h));
        float shade = 1.0 - layer * 0.15;
        vec3 rock = mix(vec3(0.018, 0.021, 0.030), vec3(0.19, 0.060, 0.028), layer / 3.0);
        col = mix(col, rock * shade, body * (0.42 + layer * 0.16));

        float ember = pow(max(0.0, sin(x * 5.0 + layer * 2.4 + pc.iTime * 0.7)), 6.0);
        col += vec3(1.0, 0.30, 0.05) * edge * (0.18 + ember * 0.45) * headroom();
    }

    // Sparse sparks lifting off the lower ridge.
    vec2 sparkGrid = vec2(g.x * 0.018, g.y * 0.018 - pc.iTime * 0.55);
    vec2 cell = floor(sparkGrid);
    vec2 fp = fract(sparkGrid);
    float h = hash21(cell);
    vec2 pos = vec2(h, fract(h * 37.3));
    float spark = smoothstep(0.045, 0.0, length(fp - pos)) * step(0.965, h);
    spark *= smoothstep(0.0, 0.45, local.y) * (1.0 - smoothstep(0.70, 0.95, local.y));
    col += vec3(1.0, 0.45, 0.08) * spark * headroom();

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
