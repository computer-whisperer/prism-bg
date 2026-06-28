// Solar wind: a bright off-screen star pushes magnetic ribbons through dusty
// space. Uses iTime -> animated wallpaper with HDR highlights.
//
//   prism-bg --shader examples/shaders/solar-wind.frag
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
    p = fract(p * vec2(43.21, 171.13));
    p += dot(p, p + 31.7);
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
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
        s += a * noise(p);
        p = mat2(1.62, 1.18, -1.18, 1.62) * p + 4.0;
        a *= 0.52;
    }
    return s;
}

vec3 palette(float x) {
    return 0.5 + 0.5 * cos(6.28318 * (x + vec3(0.02, 0.36, 0.68)));
}

void main() {
    vec2 uv = fragCoord / pc.iResolution;
    vec2 p = (fragCoord - 0.5 * pc.iResolution) / pc.iResolution.y;
    float t = pc.iTime * 0.10;

    vec2 sun = vec2(-0.62, 0.08);
    vec2 q = p - sun;
    float r = length(q);
    float a = atan(q.y, q.x);

    vec3 col = mix(vec3(0.010, 0.012, 0.030), vec3(0.035, 0.020, 0.040), uv.y);

    // Dusty nebula field, slowly advected away from the star.
    float dust = fbm(p * 2.0 + vec2(t * 0.35, -t * 0.12));
    col += mix(vec3(0.10, 0.045, 0.10), vec3(0.02, 0.08, 0.12), dust) * dust * 0.22;

    // Magnetic ribbons: angular bands whose phase bends with radius and noise.
    float bend = fbm(vec2(a * 1.8, r * 2.5 - t * 2.0));
    float bands = sin(a * 8.0 + r * 12.0 - pc.iTime * 1.1 + bend * 3.0);
    float ribbon = smoothstep(0.72, 1.0, bands) * smoothstep(1.25, 0.08, r);
    vec3 ribbonCol = palette(a * 0.18 + r * 0.12 + t);
    col += ribbonCol * ribbon * 0.42 * headroom();

    // Star core and corona. The core uses available HDR headroom but remains
    // clamped to the compositor-advertised peak.
    float core = smoothstep(0.20, 0.0, r);
    float corona = exp(-r * 2.0) * (0.35 + 0.65 * fbm(vec2(a * 3.0, r * 6.0 - t * 5.0)));
    col += vec3(1.0, 0.62, 0.28) * core * headroom();
    col += vec3(1.0, 0.30, 0.12) * corona * 0.35 * headroom();

    // Fine star flecks outside the corona.
    vec2 sp = p + vec2(t * 0.03, 0.0);
    float star = pow(hash21(floor(sp * 180.0)), 180.0) * smoothstep(0.45, 1.2, r);
    col += vec3(0.70, 0.85, 1.0) * star * headroom();

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
