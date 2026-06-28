// Cursor spotlight — a static, mouse-reactive shader (the Shadertoy iMouse
// feature). A soft glow tracks the pointer; while the left button is held the
// glow flares into HDR and concentric rings radiate from the click point.
//
//   prism-bg --shader examples/shaders/pointer.frag
//
// MOUSE (iMouse): referencing iMouse makes prism-bg bind a seat pointer and
// make this surface input-receiving, then redraw it on each pointer event —
// "repaint on motion". Because the shader has no iTime, it is otherwise static:
// it renders once and then only when the pointer moves or clicks, costing no
// GPU while the cursor is idle. iMouse follows the Shadertoy convention, in
// pixels with a y-up origin (matching fragCoord):
//   .xy  cursor position while a button is held (last drag position once up)
//   .zw  the position of the press; sign(.z) > 0 while the button is down,
//        sign(.w) > 0 only on the frame of the press
// All four are zero until the first click.
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
    vec4 iMouse;      // xy: cursor (while held); zw: click pos (sign = state)
} pc;

// Highlight headroom above diffuse white (>= 1.0; 1.0 on SDR).
float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

vec3 palette(float t) {
    return 0.5 + 0.5 * cos(6.28318 * (t + vec3(0.0, 0.33, 0.67)));
}

void main() {
    vec2 uv = fragCoord / pc.iResolution;

    // A calm base wash so the spotlight has something to reveal.
    vec3 col = mix(vec3(0.02, 0.03, 0.06), vec3(0.06, 0.05, 0.10), uv.y);

    // Cursor position, normalized. Before the first click iMouse is zero, so
    // park the spotlight off-screen until the user interacts.
    bool engaged = pc.iMouse.x != 0.0 || pc.iMouse.y != 0.0;
    vec2 m = engaged ? pc.iMouse.xy / pc.iResolution : vec2(-1.0);
    bool held = pc.iMouse.z > 0.0; // sign(.z): left button currently down

    // Aspect-correct distance so the glow stays round on wide outputs.
    float aspect = pc.iResolution.x / pc.iResolution.y;
    vec2 d = (uv - m) * vec2(aspect, 1.0);
    float r = length(d);

    // Soft spotlight that follows the cursor; brighter and tighter while held.
    float reach = held ? 0.18 : 0.28;
    float glow = smoothstep(reach, 0.0, r);
    float gain = held ? 1.0 : 0.45;
    col += palette(0.6 + 0.15 * uv.x) * glow * gain * headroom();

    // Rings centered on the last click position (.zw, magnitude is the pixel
    // coords). They sharpen while the button is held.
    if (pc.iMouse.z != 0.0 || pc.iMouse.w != 0.0) {
        vec2 c = abs(pc.iMouse.zw) / pc.iResolution;
        float rc = length((uv - c) * vec2(aspect, 1.0));
        float rings = 0.5 + 0.5 * cos(rc * 90.0);
        float falloff = smoothstep(0.4, 0.0, rc);
        col += vec3(0.6, 0.7, 1.0) * pow(rings, held ? 6.0 : 12.0)
             * falloff * 0.4 * headroom();
    }

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
