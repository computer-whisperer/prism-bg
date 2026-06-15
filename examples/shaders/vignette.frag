// Static radial vignette — no iTime, so prism-bg renders a single frame
// and then idles exactly like an image wallpaper (no per-frame GPU cost).
//
//   prism-bg --shader examples/shaders/vignette.frag
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 iResolution; float iTime; float _pad; } pc;

void main() {
    vec2 uv = fragCoord / pc.iResolution;
    float d = distance(uv, vec2(0.5));
    vec3 base = vec3(0.10, 0.12, 0.18);
    float falloff = smoothstep(0.75, 0.1, d);
    outColor = vec4(base * (0.25 + 0.75 * falloff), 1.0);
}
