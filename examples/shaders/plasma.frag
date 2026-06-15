// Classic animated plasma. Uses iTime → animated wallpaper.
//
//   prism-bg --shader examples/shaders/plasma.frag
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 iResolution; float iTime; float _pad; } pc;

void main() {
    vec2 uv = fragCoord / pc.iResolution;
    float t = pc.iTime * 0.5;
    float v = sin(uv.x * 10.0 + t)
            + sin(uv.y * 10.0 + t)
            + sin((uv.x + uv.y) * 10.0 + t)
            + sin(length(uv - 0.5) * 14.0 - t * 2.0);
    v *= 0.25;
    vec3 col = 0.5 + 0.5 * cos(vec3(0.0, 2.094, 4.188) + v * 3.1415 * 2.0);
    outColor = vec4(col, 1.0);
}
