//!luminance bright
// Tidal slate: slow wave interference over a polished stone-and-water surface,
// with pearly highlights slipping through the troughs. Uses iTime -> animated.
//
//   prism-bg --shader examples/shaders/tidal-slate.frag
//
// Shaded in shared cluster space so the ripple field is one continuous surface
// across a multi-monitor desktop.
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
    p = fract(p * vec2(87.13, 219.47));
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
    float a = 0.52;
    for (int i = 0; i < 5; i++) {
        s += a * noise(p);
        p = mat2(1.74, -0.62, 0.62, 1.74) * p + vec2(13.0, 5.0);
        a *= 0.5;
    }
    return s;
}

void main() {
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = g / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 p = (gv - 0.5) * vec2(aspect, 1.0);

    float t = pc.iTime * 0.10;
    vec2 q = p * 4.6;
    float warp = fbm(q * 0.55 + vec2(t, -t * 0.7));
    q += vec2(0.45 * warp, 0.30 * fbm(q * 0.7 - vec2(t * 1.5, t)));

    float w1 = sin(dot(q, vec2(1.18, 0.34)) * 5.0 + t * 5.0);
    float w2 = sin(dot(q, vec2(-0.42, 1.05)) * 6.4 - t * 4.0);
    float w3 = sin(length(q + vec2(0.6, -0.2)) * 7.5 - t * 6.2);
    float h = (w1 + 0.75 * w2 + 0.55 * w3) / 2.3;

    float ridges = smoothstep(0.70, 0.96, abs(h));
    float fine = smoothstep(0.76, 0.97, abs(sin((h + warp * 0.45) * 15.0)));
    float foam = ridges * (0.35 + 0.65 * fine);

    vec3 slate = mix(vec3(0.20, 0.26, 0.30), vec3(0.54, 0.58, 0.56), gv.y);
    vec3 water = mix(vec3(0.04, 0.32, 0.42), vec3(0.30, 0.66, 0.72), 0.5 + 0.5 * h);
    vec3 col = mix(slate, water, 0.45 + 0.25 * warp);

    vec3 pearl = vec3(0.94, 0.98, 1.0);
    vec3 copper = vec3(0.95, 0.48, 0.20);
    col += mix(pearl, copper, 0.18 + 0.18 * sin(q.x + pc.iTime * 0.25)) * foam * 0.50 * headroom();
    col += vec3(0.12, 0.38, 0.48) * (1.0 - foam) * smoothstep(0.15, 0.95, warp) * 0.25;

    float vignette = smoothstep(1.35, 0.20, length((gv - 0.5) * vec2(aspect, 1.0)));
    col *= 0.78 + 0.24 * vignette;

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
