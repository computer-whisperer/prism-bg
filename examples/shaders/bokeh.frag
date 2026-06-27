// Dreamy floating bokeh — soft out-of-focus light discs drifting upward over
// a dark gradient, like backlit dust. Uses iTime → animated. Cheap: a fixed
// loop of cheap disc evaluations, no noise.
//
//   prism-bg --shader examples/shaders/bokeh.frag
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 iResolution; float iTime; float _pad; } pc;

float hash11(float n) {
    return fract(sin(n * 91.3458) * 47453.5453);
}

void main() {
    vec2 p = (fragCoord - 0.5 * pc.iResolution) / pc.iResolution.y;
    float t = pc.iTime * 0.05;

    // Background: cool dark gradient, a hint warmer low and to the left.
    vec2 uv = fragCoord / pc.iResolution;
    vec3 col = mix(vec3(0.02, 0.03, 0.06), vec3(0.05, 0.04, 0.09), uv.y);
    col += vec3(0.04, 0.02, 0.03) * (1.0 - uv.x) * (1.0 - uv.y);

    // 18 discs, each with its own column, speed, size and tint. They rise and
    // wrap, with a slow horizontal sway for life.
    const int N = 18;
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        float seed = hash11(fi + 1.0);
        float speed = 0.04 + 0.10 * hash11(fi + 7.0);
        float size = 0.05 + 0.16 * hash11(fi + 3.0);

        // Vertical position wraps through a span taller than the screen.
        float y = fract(seed + t * speed * 6.0) * 1.6 - 0.8;
        float x = (seed - 0.5) * 2.0 * (pc.iResolution.x / pc.iResolution.y);
        x += 0.10 * sin(t * 6.0 + seed * 6.2831);

        vec2 c = vec2(x, y);
        float d = length(p - c) / size;

        // Soft disc with a slightly brighter rim — classic bokeh look.
        float disc = smoothstep(1.0, 0.6, d);
        float rim = smoothstep(1.0, 0.9, d) * (1.0 - smoothstep(0.9, 0.6, d));
        float a = disc * 0.12 + rim * 0.10;

        vec3 tint = mix(vec3(0.5, 0.7, 1.0), vec3(1.0, 0.7, 0.5), hash11(fi + 5.0));
        col += tint * a;
    }

    outColor = vec4(col, 1.0);
}
