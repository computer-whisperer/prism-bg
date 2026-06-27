// Northern lights over a dark, star-flecked sky. Uses iTime → animated.
// Curtains of green/teal drift and shimmer; a faint magenta fringe rides
// the upper edge. Moderately cheap (a couple of fbm evaluations per pixel).
//
//   prism-bg --shader examples/shaders/aurora.frag
//
// Output is extended-linear with sRGB primaries (1.0 = reference white);
// prism-bg tags the surface and the compositor converts.
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 iResolution; float iTime; float _pad; } pc;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) {
        s += a * vnoise(p);
        p = p * 2.02 + 7.0;
        a *= 0.5;
    }
    return s;
}

void main() {
    vec2 uv = fragCoord / pc.iResolution;
    float t = pc.iTime * 0.08;

    // Night-sky vertical gradient: indigo at the horizon to near-black up top.
    vec3 col = mix(vec3(0.015, 0.03, 0.07), vec3(0.0, 0.005, 0.02), uv.y);

    // Sparse stars, only in the upper sky so the aurora reads clearly below.
    vec2 sp = fragCoord / pc.iResolution.y;
    float star = pow(hash21(floor(sp * 320.0)), 220.0);
    star *= smoothstep(0.35, 0.95, uv.y);
    col += vec3(0.8, 0.85, 1.0) * star;

    // Two drifting curtains. Each is a horizontal noise band whose height is
    // modulated by fbm; brightness falls off above and below the band.
    for (int k = 0; k < 2; k++) {
        float fk = float(k);
        float drift = t * (1.0 + fk * 0.6);
        float band = fbm(vec2(uv.x * 3.0 + drift, fk * 11.0));
        float height = 0.32 + 0.22 * fk + 0.16 * band;
        float thick = 0.12 + 0.05 * fk;
        float curtain = smoothstep(thick, 0.0, abs(uv.y - height));

        // Vertical filaments shimmering within the curtain.
        float fil = fbm(vec2(uv.x * 14.0, uv.y * 5.0 - drift * 6.0));
        curtain *= 0.55 + 0.65 * fil;

        vec3 tint = mix(vec3(0.05, 0.7, 0.35), vec3(0.1, 0.45, 0.8), fk);
        col += tint * curtain * 0.7;
    }

    // Magenta fringe skimming the top of the higher curtain.
    float fringe = smoothstep(0.6, 0.78, uv.y) * (1.0 - smoothstep(0.78, 0.95, uv.y));
    col += vec3(0.5, 0.1, 0.45) * fringe * 0.15;

    outColor = vec4(col, 1.0);
}
