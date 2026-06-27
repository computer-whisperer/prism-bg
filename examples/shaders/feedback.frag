// Glowing trails: a wandering emitter painted over a slowly decaying, gently
// diffusing copy of the previous frame. A feedback shader — it samples its own
// last output via iPrevFrame, so motion leaves comet-like tails.
//
//   prism-bg --shader examples/shaders/feedback.frag
//
// FEEDBACK (iPrevFrame): referencing iPrevFrame makes prism-bg render this
// shader into a ping-pong buffer and feed last frame back in. The buffer is
// fp16 linear, so trails keep HDR range. NOTE the y-flip in prev() below:
// fragCoord is y-up (Shadertoy convention) but Vulkan textures sample y-down,
// so iPrevFrame must be read at vec2(uv.x, 1.0 - uv.y). Feedback shaders redraw
// every frame; cap with --fps if desired.
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
} pc;

// Previous frame's output (this buffer, last frame). Sampled y-flipped.
layout(set = 1, binding = 0) uniform sampler2D iPrevFrame;
vec3 prev(vec2 uv) { return texture(iPrevFrame, vec2(uv.x, 1.0 - uv.y)).rgb; }

vec3 palette(float t) {
    return 0.5 + 0.5 * cos(6.28318 * (t + vec3(0.0, 0.33, 0.67)));
}

void main() {
    vec2 uv = fragCoord / pc.iResolution;
    vec2 px = 1.0 / pc.iResolution;

    // Diffuse last frame (5-tap, weight sum 8) and decay slightly → soft trails
    // that fade instead of accumulating forever.
    vec3 acc = prev(uv) * 4.0;
    acc += prev(uv + vec2(px.x, 0.0));
    acc += prev(uv - vec2(px.x, 0.0));
    acc += prev(uv + vec2(0.0, px.y));
    acc += prev(uv - vec2(0.0, px.y));
    vec3 col = acc / 8.0 * 0.985;

    // A wandering emitter (a couple of incommensurate sinusoids → a Lissajous
    // path that doesn't obviously repeat).
    vec2 p = vec2(0.5 + 0.36 * cos(pc.iTime * 0.7), 0.5 + 0.30 * sin(pc.iTime * 1.1));
    float d = distance(uv, p);
    col += palette(pc.iTime * 0.05) * smoothstep(0.025, 0.0, d) * 1.5;

    outColor = vec4(col, 1.0);
}
