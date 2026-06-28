// Drifting clouds from a noise texture — a demo of static image channels. A
// fractal-Brownian-motion field is built by sampling a tiling RGBA noise image
// at several octaves and scrolling it over time, then shaded as soft clouds.
//
//   prism-bg --shader examples/shaders/clouds.frag
//
// TEXTURE (iChannelN): the /*!prism …*/ block below declares a `textures` map
// (name → path, resolved relative to this .frag) and routes the image pass's
// channel 0 to it. The image is uploaded once per GPU as an sRGB-sampled
// texture (so texture() returns linear light) with repeat wrap + linear filter.
// A shader can mix texture, buffer, and "self" channels freely; this one needs
// no //!pass sections (a plain body plus the metadata block is enough).
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
layout(set = 1, binding = 0) uniform sampler2D iChannel0; // noise

// Highlight headroom above diffuse white (>= 1.0; 1.0 on SDR).
float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

// Smooth value noise: bilinear-sample the noise texture's red channel. The
// texture is repeat-wrapped, so scaled coordinates tile seamlessly.
float noise(vec2 p) {
    return texture(iChannel0, p * (1.0 / 256.0)).r;
}

// 5-octave fbm — the canonical use of a noise texture.
float fbm(vec2 p) {
    float sum = 0.0, amp = 0.5;
    for (int i = 0; i < 5; i++) {
        sum += amp * noise(p);
        p = p * 2.02 + 7.0;
        amp *= 0.5;
    }
    return sum;
}

void main() {
    vec2 uv = fragCoord / pc.iResolution;
    float aspect = pc.iResolution.x / pc.iResolution.y;
    vec2 p = vec2(uv.x * aspect, uv.y) * 3.0;

    // Two layers scrolling at different speeds give parallax depth.
    float t = pc.iTime * 0.04;
    float base = fbm(p + vec2(t, 0.3 * t));
    float detail = fbm(p * 2.0 - vec2(2.0 * t, t));
    float density = clamp(base * 0.75 + detail * 0.35 - 0.25, 0.0, 1.0);

    // Sky gradient with the clouds lit on top; bright tops lifted into HDR.
    vec3 sky = mix(vec3(0.06, 0.10, 0.20), vec3(0.20, 0.32, 0.50), uv.y);
    vec3 cloud = mix(vec3(0.30, 0.32, 0.38), vec3(1.0), smoothstep(0.2, 1.0, density));
    vec3 col = mix(sky, cloud, smoothstep(0.1, 0.7, density));
    col += vec3(1.0, 0.95, 0.85) * smoothstep(0.8, 1.05, density) * (headroom() - 1.0);

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
