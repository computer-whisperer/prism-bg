// Domain-warped fbm flow — slow, organic ribbons of color folding through
// one another. Uses iTime → animated. The priciest of the demos (fbm called
// several times per pixel for the warp), but still trivial at wallpaper rates.
//
//   prism-bg --shader examples/shaders/warp.frag
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
        p = p * 2.03 + 11.0;
        a *= 0.5;
    }
    return s;
}

void main() {
    // Aspect-correct coordinates centered on screen.
    vec2 p = (fragCoord - 0.5 * pc.iResolution) / pc.iResolution.y;
    p *= 2.4;
    float t = pc.iTime * 0.06;

    // Two stages of domain warping: q displaces p, r displaces again.
    vec2 q = vec2(fbm(p + vec2(0.0, t)), fbm(p + vec2(5.2, -t)));
    vec2 r = vec2(fbm(p + 3.0 * q + vec2(1.7, 9.2) + t),
                  fbm(p + 3.0 * q + vec2(8.3, 2.8) - t));
    float f = fbm(p + 3.5 * r);

    // Map the field through a smooth palette (deep violet → teal → warm gold).
    vec3 a = vec3(0.05, 0.03, 0.12);
    vec3 b = vec3(0.10, 0.45, 0.50);
    vec3 c = vec3(0.85, 0.55, 0.20);
    vec3 col = mix(a, b, smoothstep(0.2, 0.6, f));
    col = mix(col, c, smoothstep(0.55, 0.95, f) * length(r));

    // Lift the brightest folds a touch into HDR for a soft glow.
    col += vec3(0.15) * smoothstep(0.8, 1.1, f + length(q)) * headroom();

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
