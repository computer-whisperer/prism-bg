// Organic cellular pattern with glowing edges — animated Voronoi. The cells
// breathe as their feature points drift on sine orbits. Uses iTime → animated.
// Moderately heavy: two neighborhood passes per pixel (nearest cell, then the
// distance to its borders), the classic two-pass Voronoi-edge approach.
//
//   prism-bg --shader examples/shaders/voronoi.frag
//
// Original implementation of a well-known technique (Inigo Quilez's Voronoi
// distance). Output is extended-linear with sRGB primaries (1.0 = white).
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 iResolution; float iTime; float _pad; } pc;

vec2 hash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453123);
}

// IQ cosine palette: smooth, loopable color ramp.
vec3 palette(float t) {
    vec3 a = vec3(0.45, 0.40, 0.55);
    vec3 b = vec3(0.30, 0.30, 0.35);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.0, 0.15, 0.35);
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    vec2 uv = (fragCoord - 0.5 * pc.iResolution) / pc.iResolution.y;
    vec2 p = uv * 5.0;
    float t = pc.iTime * 0.5;

    vec2 ip = floor(p), fp = fract(p);

    // Pass 1: nearest feature point and which cell it lives in.
    float md = 8.0;
    vec2 mr, mg;
    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
        vec2 g = vec2(i, j);
        vec2 o = hash22(ip + g);
        o = 0.5 + 0.5 * sin(t + 6.28318 * o);
        vec2 r = g + o - fp;
        float d = dot(r, r);
        if (d < md) { md = d; mr = r; mg = g; }
    }

    // Pass 2: distance to the borders with neighboring cells (perpendicular
    // bisector between the two nearest feature points).
    float me = 8.0;
    for (int j = -2; j <= 2; j++)
    for (int i = -2; i <= 2; i++) {
        vec2 g = mg + vec2(i, j);
        vec2 o = hash22(ip + g);
        o = 0.5 + 0.5 * sin(t + 6.28318 * o);
        vec2 r = g + o - fp;
        if (dot(mr - r, mr - r) > 0.00001)
            me = min(me, dot(0.5 * (mr + r), normalize(r - mr)));
    }

    // Cell fill: dark, tinted by the cell's hashed id.
    vec2 cellId = ip + mg;
    vec3 col = palette(dot(cellId, vec2(0.11, 0.07)) + t * 0.1) * 0.18;
    col *= 0.6 + 0.4 * smoothstep(0.0, 0.6, sqrt(md));

    // Glowing borders.
    float edge = smoothstep(0.04, 0.0, me);
    col += palette(0.6 + 0.05 * sin(t)) * edge * 0.5;

    outColor = vec4(col, 1.0);
}
