// Gentle animated gradient. Uses iTime, so prism-bg animates it
// (vsync-paced, paused when the wallpaper is occluded).
//
//   prism-bg --shader examples/shaders/gradient.frag
//
// Output is extended-linear with sRGB primaries (1.0 = reference white);
// prism-bg tags the surface accordingly and the compositor converts.
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 iResolution; float iTime; float _pad; } pc;

void main() {
    vec2 uv = fragCoord / pc.iResolution;
    vec3 col = 0.5 + 0.5 * cos(pc.iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
    outColor = vec4(col, 1.0);
}
