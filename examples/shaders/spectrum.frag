// Audio-reactive spectrum: a glowing equalizer along the bottom plus a
// bass-driven background bloom. Reads the live spectrum captured from the
// default audio sink — play something and it dances.
//
//   prism-bg --shader examples/shaders/spectrum.frag
//
// AUDIO UNIFORMS (set 0, binding 0): captured from the default sink's monitor.
// All values are 0..1. The capture only runs while a shader references them.
//   iAudioBins[8]  — 32 log-spaced magnitude bins, low→high, packed 4/vec4.
//                    Read bin i with audioBin(i) below.
//   iAudioLevel    — overall loudness.
//   iAudioBass/Mid/Treble — low/mid/high band energy, for quick reactions.
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

layout(set = 0, binding = 0, std140) uniform Audio {
    vec4 iAudioBins[8];   // 32 bins packed four-per-vec4
    float iAudioLevel;
    float iAudioBass;
    float iAudioMid;
    float iAudioTreble;
} au;

const int BINS = 32;

// Unpack spectrum bin i (0..31) from the vec4 array.
float audioBin(int i) {
    return au.iAudioBins[i >> 2][i & 3];
}

vec3 palette(float t) {
    vec3 a = vec3(0.5), b = vec3(0.5), c = vec3(1.0);
    vec3 d = vec3(0.0, 0.33, 0.67);
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    vec2 uv = fragCoord / pc.iResolution; // 0..1, y-up

    // Background: a slow drifting wash that brightens with bass.
    vec3 col = 0.04 * palette(pc.iTime * 0.02 + uv.y * 0.3);
    col += 0.25 * au.iAudioBass * palette(0.6 + uv.x * 0.2);

    // Equalizer: 32 bars across the width, height = that bin's magnitude.
    float x = uv.x * float(BINS);
    int bin = int(x);
    float frac = fract(x);
    // Small gap between bars.
    float bar = smoothstep(0.06, 0.12, frac) * smoothstep(0.06, 0.12, 1.0 - frac);

    float mag = audioBin(clamp(bin, 0, BINS - 1));
    float height = mag * 0.9;
    // Bright inside the bar up to its height, with a soft cap.
    float lit = bar * smoothstep(height + 0.01, height - 0.04, uv.y);
    vec3 barCol = palette(float(bin) / float(BINS));
    col += barCol * lit * (0.4 + 0.6 * mag);

    // A glowing crest line riding the top of each bar.
    float crest = bar * exp(-220.0 * (uv.y - height) * (uv.y - height));
    col += barCol * crest * (0.5 + mag);

    outColor = vec4(col, 1.0);
}
