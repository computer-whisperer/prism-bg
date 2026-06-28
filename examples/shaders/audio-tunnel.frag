// Audio tunnel — fly down an infinite ring tunnel whose walls ARE the live
// spectrum. Each angular sector glows with one frequency bin (low bins at the
// bottom, sweeping up around the ring); bass surges the camera forward and
// flares the vanishing point, treble sparkles across the walls. A more abstract
// companion to spectrum.frag's equalizer. Play something and fly.
//
//   prism-bg --shader examples/shaders/audio-tunnel.frag
//
// AUDIO UNIFORMS (set 0, binding 0): captured from the default sink's monitor,
// all 0..1. Capture only runs while a shader references them.
//   iAudioBins[8]  — 32 log-spaced magnitude bins, low→high, packed 4/vec4.
//   iAudioLevel    — overall loudness.
//   iAudioBass/Mid/Treble — band energies for quick reactions.
//
// Output is extended-linear with sRGB primaries (1.0 = reference white); ring
// crests and the bass flare are pushed into HDR headroom.
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

layout(set = 0, binding = 0, std140) uniform Audio {
    vec4 iAudioBins[8];   // 32 bins packed four-per-vec4
    float iAudioLevel;
    float iAudioBass;
    float iAudioMid;
    float iAudioTreble;
} au;

const int BINS = 32;
const float TAU = 6.28318531;

// Highlight headroom above diffuse white (>= 1.0; 1.0 on SDR).
float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

// Unpack spectrum bin i (0..31) from the vec4 array.
float audioBin(int i) { return au.iAudioBins[i >> 2][i & 3]; }

// Smoothly interpolate the spectrum at a fractional bin position, so the ring's
// colour/energy varies continuously around the wall instead of stair-stepping.
float spectrumAt(float fb) {
    fb = clamp(fb, 0.0, float(BINS - 1));
    int i = int(fb);
    float f = fract(fb);
    return mix(audioBin(i), audioBin(min(i + 1, BINS - 1)), f);
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

vec3 palette(float t) {
    return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67)));
}

void main() {
    vec2 p = (fragCoord - 0.5 * pc.iResolution) / pc.iResolution.y;
    float r = length(p);
    float a = atan(p.y, p.x);              // -PI..PI
    float u = a / TAU + 0.5;               // 0..1 around the ring

    // Tunnel projection: depth grows toward the centre (the vanishing point).
    float depth = 1.0 / (r + 0.12);
    // Forward motion; bass surges the flight speed.
    float z = depth + pc.iTime * 1.2 + au.iAudioBass * 2.0;

    // This wall sector's frequency bin (low at the bottom of the ring, u=0.25,
    // sweeping around). Map angle → bin and read the spectrum there.
    float fb = u * float(BINS);
    float mag = spectrumAt(fb);

    // Rings rushing toward the viewer; their crest brightness rides the bin.
    float rings = 0.5 + 0.5 * sin(z * 5.0);
    float crest = pow(rings, 3.0);

    vec3 col = vec3(0.0);
    vec3 wallCol = palette(u + z * 0.03);

    // Wall glow: base lit by the sector's energy, crests punched to HDR on loud
    // bins so the tunnel pulses with the music.
    col += wallCol * (0.08 + 1.4 * mag) * rings;
    col += wallCol * crest * mag * headroom();

    // Treble sparkle: flecks scattered on the walls, gated by high-band energy.
    float spk = pow(hash21(floor(vec2(u * 220.0, z * 6.0))), 40.0);
    col += vec3(1.0) * spk * au.iAudioTreble * 2.0 * headroom();

    // Distance fog: fade toward the dark vanishing point so depth reads.
    float fog = smoothstep(0.0, 0.45, r);
    col *= fog;

    // Bass flare blooming out of the centre.
    float flare = exp(-r * 5.0) * (0.3 + au.iAudioBass);
    col += palette(0.05 + pc.iTime * 0.05) * flare * headroom();

    // Gentle overall lift with loudness so quiet passages stay dim.
    col *= 0.5 + 0.8 * au.iAudioLevel;

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
