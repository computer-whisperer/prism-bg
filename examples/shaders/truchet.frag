// Flowing Truchet maze — each tile holds a randomly-rotated pair of quarter-
// arcs, and across the grid they connect into endless looping ribbons. The
// whole field drifts slowly and the ribbon color cycles. Uses iTime → animated.
// Cheap: one hash and a couple of distances per pixel.
//
//   prism-bg --shader examples/shaders/truchet.frag
//
// Original implementation of the classic Truchet-tile technique.
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

// Highlight headroom above diffuse white (>= 1.0; 1.0 on SDR).
float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec3 palette(float t) {
    vec3 a = vec3(0.5), b = vec3(0.5), c = vec3(1.0);
    vec3 d = vec3(0.0, 0.33, 0.67);
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    vec2 uv = (fragCoord - 0.5 * pc.iResolution) / pc.iResolution.y;
    float t = pc.iTime * 0.04;

    uv *= 5.0;
    uv += vec2(t * 2.0, t); // slow diagonal drift

    vec2 ip = floor(uv), fp = fract(uv);
    float h = hash21(ip);

    // Half the tiles flip, so the two quarter-arcs change which corners they
    // join — that random choice is what makes the ribbons weave.
    if (h < 0.5) fp.x = 1.0 - fp.x;

    // Distance to the nearer of two arcs (radius 0.5, centered on opposite
    // corners); abs(d - 0.5) is the distance to the ribbon centerline.
    float d = min(length(fp - vec2(0.0)), length(fp - vec2(1.0)));
    float w = abs(d - 0.5);

    // Position along the ribbon drives a flowing color.
    float flow = (fp.x + fp.y) * 0.5 + length(ip) * 0.15 + pc.iTime * 0.08;
    vec3 ribbon = palette(flow) * 0.55;

    float line = smoothstep(0.10, 0.04, w);
    float glow = smoothstep(0.30, 0.0, w) * 0.25;

    vec3 bg = vec3(0.02, 0.025, 0.04);
    vec3 col = bg + ribbon * (line + glow) * headroom();

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
