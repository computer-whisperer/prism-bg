//!luminance dark
// Silk: slow satin folds under a single soft key light. A domain-warped height
// field is shaded as a surface — diffuse from the fold normals plus a tight
// anisotropic sheen that glides along the crests. Monochrome slate-blue with a
// faint warm rim, deliberately understated. Uses iTime -> animated.
//
//   prism-bg --shader examples/shaders/silk.frag
//
// Shaded in shared cluster space so the fabric drapes as one continuous sheet
// across a multi-monitor desktop. Cost is comparable to warp.frag: the height
// field is sampled three times (value + finite-difference normal) over one
// shared warp.
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
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 27.61);
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
    for (int i = 0; i < 4; i++) {
        s += a * noise(p);
        p = mat2(1.66, 0.94, -0.94, 1.66) * p + vec2(9.0, 3.0);
        a *= 0.5;
    }
    return s;
}

// Fold height at p given a precomputed warp. The warp varies far more slowly
// than the folds, so reusing it for the finite-difference taps is safe and
// saves two fbm evaluations per tap.
float height(vec2 p, vec2 warp) {
    // Stretch x so the folds hang as long draped ridges, not isotropic blobs.
    return fbm(vec2(p.x * 0.55, p.y * 1.25) + warp);
}

void main() {
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = g / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 p = (gv - 0.5) * vec2(aspect, 1.0);

    float t = pc.iTime * 0.045;
    vec2 q = p * 3.4;
    vec2 warp = 1.35 * vec2(fbm(q * 0.5 + vec2(t, -t * 0.6)),
                            fbm(q * 0.5 + vec2(-t * 0.8, t) + 7.3));

    float e = 0.018;
    float h = height(q, warp);
    float hx = height(q + vec2(e, 0.0), warp);
    float hy = height(q + vec2(0.0, e), warp);
    vec3 n = normalize(vec3(h - hx, h - hy, e * 1.6));

    vec3 keyDir = normalize(vec3(-0.45, 0.65, 0.62));
    vec3 rimDir = normalize(vec3(0.75, -0.30, 0.42));
    float diff = max(dot(n, keyDir), 0.0);
    float rim = max(dot(n, rimDir), 0.0);

    // Anisotropic sheen: a specular lobe flattened along the fold direction so
    // highlights run as long threads along the crests instead of round dots.
    vec3 view = vec3(0.0, 0.0, 1.0);
    vec3 hv = normalize(keyDir + view);
    float spec = pow(max(dot(n, hv), 0.0), 56.0);
    spec *= 0.55 + 0.45 * smoothstep(0.2, 0.8, h);

    vec3 base = mix(vec3(0.020, 0.026, 0.044), vec3(0.13, 0.17, 0.27), diff);
    base += vec3(0.10, 0.07, 0.05) * pow(rim, 3.0) * 0.60;

    vec3 sheen = vec3(0.72, 0.80, 0.94);
    vec3 col = base + sheen * spec * 1.05 * min(headroom(), 1.8);

    float vignette = smoothstep(1.45, 0.30, length((gv - 0.5) * vec2(aspect, 1.0)));
    col *= 0.72 + 0.28 * vignette;

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
