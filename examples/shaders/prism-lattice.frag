//!luminance dark
// Prism lattice: a dark crystalline triangle field with slow chromatic glints
// travelling along its edges. Uses iTime -> animated wallpaper.
//
//   prism-bg --shader examples/shaders/prism-lattice.frag
//
// The lattice is built in shared cluster space, so edges and glints continue
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
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 41.3);
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
        p = mat2(1.62, 1.02, -1.02, 1.62) * p + vec2(9.0, 4.0);
        a *= 0.5;
    }
    return s;
}

float lineWave(float x, float width) {
    float d = abs(fract(x) - 0.5);
    return smoothstep(width, 0.0, d);
}

void main() {
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = g / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 p = (gv - 0.5) * vec2(aspect, 1.0);

    float t = pc.iTime * 0.075;
    float ct = cos(0.18), st = sin(0.18);
    vec2 q = mat2(ct, -st, st, ct) * p * 7.0;

    // Three stripe families at 60-degree angles form a triangular lattice.
    const vec2 A = vec2(1.0, 0.0);
    const vec2 B = vec2(0.5, 0.8660254);
    const vec2 C = vec2(-0.5, 0.8660254);
    float la = lineWave(dot(q, A) + 0.10 * fbm(q * 0.40 + t), 0.035);
    float lb = lineWave(dot(q, B) + 0.12 * fbm(q * 0.42 - t * 1.3), 0.035);
    float lc = lineWave(dot(q, C) + 0.10 * fbm(q * 0.38 + vec2(t, -t)), 0.035);
    float lines = clamp(la + lb + lc, 0.0, 1.0);
    float nodes = clamp(la * lb + lb * lc + lc * la, 0.0, 1.0);

    float glass = fbm(q * 0.28 + vec2(t * 0.7, -t));
    vec3 col = mix(vec3(0.006, 0.008, 0.014), vec3(0.018, 0.020, 0.032), glass);

    vec3 cyan = vec3(0.08, 0.78, 0.95);
    vec3 rose = vec3(0.95, 0.16, 0.44);
    vec3 gold = vec3(1.0, 0.72, 0.22);
    float chroma = 0.5 + 0.5 * sin(dot(q, vec2(0.21, 0.37)) - pc.iTime * 0.35);
    vec3 tint = mix(mix(cyan, rose, chroma), gold, 0.25 + 0.25 * sin(glass * 6.28318));

    float sweep = smoothstep(0.55, 1.0, sin(dot(q, vec2(0.70, -0.24)) - pc.iTime * 0.9));
    col += tint * lines * (0.09 + 0.28 * sweep) * headroom();
    col += vec3(0.85, 0.95, 1.0) * pow(nodes, 2.0) * (0.45 + 0.55 * sweep) * headroom();

    // A quiet centre glow keeps the desktop from reading as a flat wireframe.
    float centre = smoothstep(1.05, 0.10, length((gv - 0.5) * vec2(aspect, 1.0)));
    col += mix(vec3(0.02, 0.03, 0.06), vec3(0.06, 0.02, 0.04), chroma) * centre * 0.18;

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
