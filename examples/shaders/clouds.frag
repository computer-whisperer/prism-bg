//!luminance bright
// Clouds: slow cumulus drifting across a summer sky, built from fbm over the
// checked-in RGBA noise texture instead of ALU hash noise — the worked example
// for static texture channels (see README, "Static image channels"). The
// texture is declared raw (no srgb linearization) because its texels are data,
// not color. Uses iTime -> animated.
//
//   prism-bg --shader examples/shaders/clouds.frag
//
// One sky across the whole cluster. Cheap: two 5-octave fbm evaluations per
// pixel, each octave a single bilinear texture fetch.
/*!prism
{ "textures": { "noise": "../textures/rgba-noise.png" },
  "channels": { "image": { "0": "noise" } } }
*/
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
layout(set = 1, binding = 0) uniform sampler2D iChannel0;  // = noise (256², repeat)

float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

// Smooth value noise from the texture: quintic-ish fade between texel centres,
// letting the sampler's bilinear filter do the interpolation.
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return texture(iChannel0, (i + f + 0.5) / 256.0).r;
}

float fbm(vec2 p) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
        s += a * noise(p);
        p = mat2(1.62, 1.18, -1.18, 1.62) * p + vec2(31.0, 57.0);
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

    // Sky: deep blue zenith paling toward the horizon, with a soft sun.
    vec3 col = mix(vec3(0.66, 0.80, 0.94), vec3(0.22, 0.44, 0.78),
                   smoothstep(0.0, 1.0, gv.y));
    vec2 sunPos = vec2(0.30 * aspect, 0.38);
    float sunD = length(p - sunPos);
    col += vec3(1.0, 0.92, 0.75) * exp(-sunD * sunD * 16.0) * 0.25;

    // Cloud deck: domain-drifted fbm density, squashed vertically so the
    // billows read as flat-bottomed cumulus rather than round blobs.
    float t = pc.iTime * 0.012;
    vec2 q = vec2(p.x * 0.9 + t * 4.0, p.y * 1.9) * 2.2;
    float d = fbm(q);
    float cover = smoothstep(0.355, 0.58, d);

    // Light the deck by sampling the density a step toward the sun: thinner
    // material sunward means a brighter edge, thicker means a shaded base.
    float toward = fbm(q + normalize(sunPos - p) * 0.55);
    float lit = clamp(0.5 + 0.9 * (d - toward), 0.0, 1.0);
    vec3 cloud = mix(vec3(0.58, 0.63, 0.72), vec3(1.0, 0.99, 0.96), lit);
    cloud += vec3(0.35, 0.22, 0.10) * exp(-sunD * sunD * 4.0) * lit;

    col = mix(col, cloud, cover * 0.96);
    // The sun's disc burns through thin cover into the HDR headroom.
    col += vec3(1.0, 0.90, 0.70) * exp(-sunD * sunD * 260.0) *
           (1.0 - cover * 0.85) * (headroom() - 0.6);

    outColor = vec4(clamp(col, vec3(0.0), vec3(headroom())), 1.0);
}
