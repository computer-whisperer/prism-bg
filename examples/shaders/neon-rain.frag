// Neon rain on glass: falling droplets refract a soft city glow behind them.
// Uses iTime -> animated wallpaper. Procedural only; no texture channel.
//
//   prism-bg --shader examples/shaders/neon-rain.frag
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
    p = fract(p * vec2(234.13, 87.37));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

float dropLayer(vec2 uv, float cells, float speed, float t) {
    vec2 p = uv * vec2(cells, cells * 0.55);
    p.y += t * speed;
    vec2 id = floor(p);
    vec2 f = fract(p);
    float h = hash21(id);
    float lane = h * 0.72 + 0.14;
    float y = fract(h * 13.7 + t * speed * (0.25 + 0.5 * h));
    vec2 d = vec2((f.x - lane) * 2.4, f.y - y);
    float head = smoothstep(0.11, 0.0, length(d * vec2(1.0, 2.4)));
    float trail = smoothstep(0.055, 0.0, abs(f.x - lane));
    float behind = f.y - y;
    trail *= smoothstep(-0.55, -0.06, behind) * (1.0 - smoothstep(-0.02, 0.08, behind));
    return (head + trail * 0.45) * smoothstep(0.35, 1.0, h);
}

vec3 neonField(vec2 uv) {
    vec3 col = mix(vec3(0.015, 0.020, 0.040), vec3(0.030, 0.015, 0.035), uv.y);
    vec2 p = uv * vec2(10.0, 4.0);
    for (int i = 0; i < 6; i++) {
        float fi = float(i);
        vec2 c = vec2(1.0 + fi * 1.7, 0.65 + 0.22 * sin(fi * 4.1));
        float d = length((p - c) * vec2(0.65, 1.0));
        vec3 hue = 0.5 + 0.5 * cos(vec3(0.0, 2.1, 4.2) + fi * 1.3);
        col += hue * exp(-d * 2.6) * 0.20;
    }
    return col;
}

void main() {
    vec2 uv = fragCoord / pc.iResolution;
    float aspect = pc.iResolution.x / pc.iResolution.y;

    float d1 = dropLayer(vec2(uv.x * aspect, uv.y), 18.0, 0.45, pc.iTime);
    float d2 = dropLayer(vec2(uv.x * aspect + 7.3, uv.y + 2.1), 31.0, 0.85, pc.iTime);
    float drops = clamp(d1 + d2 * 0.7, 0.0, 1.0);

    vec2 refract = vec2(dFdx(drops), dFdy(drops)) * 0.075;
    vec3 bg = neonField(uv + refract);

    // Window streaks and wet specular edges.
    float streak = smoothstep(0.0, 0.9, drops);
    vec3 glass = vec3(0.38, 0.58, 0.75) * streak * 0.18;
    vec3 spec = vec3(0.9, 0.95, 1.0) * pow(streak, 5.0) * 0.45 * headroom();

    // Slight vignette keeps the center luminous without making the wallpaper
    // read like a full-screen flat gradient.
    float vig = smoothstep(0.95, 0.25, length(uv - 0.5));
    vec3 col = bg * (0.55 + 0.45 * vig) + glass + spec;

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
