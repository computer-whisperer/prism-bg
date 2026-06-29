//!luminance dark
// Glowing trails: a wandering icy comet painted over a slowly decaying, gently
// diffusing copy of the previous frame. A feedback shader — it samples its own
// last output via iPrevFrame, so motion leaves comet-like tails. The emitter
// roams the whole desktop in cluster space, so one comet crosses between
// monitors rather than every output cloning its own.
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
    float iRefWhite;  // cd/m²: output value 1.0 = diffuse white
    float iMaxLum;    // cd/m²: peak to master against
} pc;

// Highlight headroom above white (≥ 1.0): how far past diffuse white we may
// push before the compositor's display LUT rolls off the overage. 1.0 on SDR.
float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

// Previous frame's output (this buffer, last frame). Sampled y-flipped.
layout(set = 1, binding = 0) uniform sampler2D iPrevFrame;
vec3 prev(vec2 uv) { return texture(iPrevFrame, vec2(uv.x, 1.0 - uv.y)).rgb; }

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

    // The emitter roams the whole desktop in aspect-correct global coords, so a
    // single comet crosses between monitors (each output accumulates its own
    // slice of the trail) rather than every output cloning its own comet.
    vec2 gv = (pc.iOutputOffset + uv * pc.iOutputSize) / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 gp = gv * vec2(aspect, 1.0);

    // A couple of incommensurate sinusoids → a Lissajous path that doesn't
    // obviously repeat, sweeping most of the desktop.
    vec2 e = vec2(aspect * (0.5 + 0.40 * cos(pc.iTime * 0.40)),
                            0.5 + 0.40 * sin(pc.iTime * 0.55));
    float d = distance(gp, e);

    // Icy comet: a hot white core with a cool cyan halo, driven into HDR so the
    // trails glow. The trails inherit this colour through the feedback decay.
    float core = smoothstep(0.022, 0.0, d);
    float halo = smoothstep(0.070, 0.0, d);
    col += vec3(1.0) * core * headroom();
    col += vec3(0.30, 0.65, 1.0) * (halo - core) * 0.6 * headroom();

    // Clamp the accumulator to the peak so additive trails can't creep past the
    // master target over many frames.
    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
