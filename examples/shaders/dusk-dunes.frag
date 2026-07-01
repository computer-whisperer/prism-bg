//!luminance bright
// Dusk dunes: layered dune silhouettes receding into warm evening haze, with a
// low sun mastered into the HDR headroom. Each layer drifts at its own pace
// for a slow parallax; nearer ridges are darker and sharper, farther ones
// dissolve into the sky. Uses iTime -> animated.
//
//   prism-bg --shader examples/shaders/dusk-dunes.frag
//
// Drawn in shared cluster space: one horizon, one sun, one continuous range of
// dunes across a multi-monitor desktop. Cheap — five 4-octave fbm ridgelines
// per pixel and a couple of exponentials.
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
    p = fract(p * vec2(151.23, 337.19));
    p += dot(p, p + 23.45);
    return fract(p.x * p.y);
}

float noise1(float x, float seed) {
    float i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(hash21(vec2(i, seed)), hash21(vec2(i + 1.0, seed)), f);
}

float ridge(float x, float seed) {
    float s = 0.0;
    float a = 0.5;
    float fr = 1.0;
    for (int i = 0; i < 4; i++) {
        s += a * noise1(x * fr, seed + float(i) * 13.7);
        fr *= 2.1;
        a *= 0.48;
    }
    return s;
}

void main() {
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = g / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    float aa = 1.5 / max(pc.iGlobalResolution.y, 1.0);

    // Sky: apricot at the horizon lifting into dusty lavender overhead.
    float skyY = smoothstep(0.40, 1.0, gv.y);
    vec3 col = mix(vec3(0.88, 0.50, 0.30), vec3(0.33, 0.25, 0.42), skyY);

    // Low sun, shared across the cluster. The disc masters into the headroom;
    // the glow stays within diffuse range so the sky never reads blown-out.
    vec2 sunPos = vec2(0.615, 0.585);
    float sunD = length((gv - sunPos) * vec2(aspect, 1.0));
    col += vec3(1.0, 0.62, 0.32) * exp(-sunD * sunD * 34.0) * 0.55;
    float disc = smoothstep(0.030, 0.030 - aa * 2.0, sunD);
    col += vec3(1.0, 0.80, 0.55) * disc * (headroom() - 0.35);

    // Haze color the far layers dissolve into — warmer on the sun's side.
    float sunSide = exp(-sunD * sunD * 5.0);
    vec3 haze = mix(vec3(0.80, 0.52, 0.42), vec3(1.0, 0.70, 0.44), sunSide);

    // Five dune layers, far to near. Farther layers sit higher, drift slower,
    // and melt into the haze; the front layer is nearly black plum.
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        float drift = pc.iTime * 0.006 * (0.4 + fi * fi * 0.30);
        float x = gv.x * aspect * (1.3 + fi * 0.55) + drift + fi * 91.7;
        float base = 0.545 - fi * 0.105;
        float amp = 0.045 + fi * 0.022;
        float yr = base + amp * (ridge(x, fi * 47.3) - 0.5) * 2.0;

        float mask = smoothstep(yr + aa, yr - aa, gv.y);
        vec3 tone = mix(vec3(0.62, 0.36, 0.36), vec3(0.10, 0.06, 0.10), fi / 4.0);
        float hazeMix = 0.62 * pow(1.0 - fi / 4.0, 1.6);
        vec3 layer = mix(tone, haze, hazeMix);
        // A whisper of sun-warmed crest light just below each ridgeline.
        layer += vec3(0.30, 0.14, 0.06) * sunSide *
                 smoothstep(0.035, 0.0, yr - gv.y) * (1.0 - fi * 0.18);
        col = mix(col, layer, mask);
    }

    // Fine dither to keep the long sky gradient from banding.
    col += (hash21(g) - 0.5) * 0.004;

    outColor = vec4(clamp(col, vec3(0.0), vec3(headroom())), 1.0);
}
