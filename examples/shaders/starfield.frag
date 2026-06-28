// Drifting parallax starfield. Uses iTime → animated. Three layers of stars
// scroll at different speeds for depth; the nearest twinkle. Cheap: a small
// fixed number of hash lookups per pixel, no noise octaves.
//
//   prism-bg --shader examples/shaders/starfield.frag
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

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

// One layer of stars on a grid of `density` cells. Each cell holds at most one
// star at a random position; brightness and twinkle phase are per-cell.
vec3 layer(vec2 uv, float density, float speed, float twinkle, float t) {
    uv += vec2(t * speed, t * speed * 0.3);
    vec2 g = uv * density;
    vec2 cell = floor(g);
    vec2 f = fract(g);

    float h = hash21(cell);
    vec2 pos = vec2(h, fract(h * 41.7));
    float d = length(f - pos);

    float bright = pow(hash21(cell + 3.1), 6.0);
    float tw = 0.6 + 0.4 * sin(t * twinkle + h * 6.2831);
    float star = smoothstep(0.06, 0.0, d) * bright * tw;

    vec3 hue = mix(vec3(1.0, 0.9, 0.75), vec3(0.7, 0.85, 1.0), fract(h * 7.3));
    return hue * star;
}

void main() {
    vec2 uv = fragCoord / pc.iResolution.y;
    float t = pc.iTime;

    // Deep-space gradient, slightly warmer toward the bottom.
    vec2 sc = fragCoord / pc.iResolution;
    vec3 col = mix(vec3(0.02, 0.02, 0.05), vec3(0.0, 0.0, 0.015), sc.y);

    col += layer(uv, 8.0, 0.010, 0.0, t) * 0.6 * headroom();  // far, static
    col += layer(uv, 14.0, 0.025, 2.5, t) * 0.8 * headroom(); // mid
    col += layer(uv, 22.0, 0.050, 4.0, t) * 1.1 * headroom(); // near, fast twinkle

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
