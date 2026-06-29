// Analog wall clock — a demo of iDate (local wall-clock time). Three hands
// sweep over a minimal dial; the second hand moves smoothly because iDate.w
// carries fractional seconds. Driven entirely by iDate (no iTime), so prism-bg
// treats it as animated and redraws it continuously to tick.
//
//   prism-bg --shader examples/shaders/clock.frag
//
// iDate = (year, month [0-11], day-of-month, seconds-since-midnight). Here only
// the seconds component is used, split into h/m/s angles.
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
    vec4 iMouse;
    vec4 iDate;       // (year, month, day, seconds-since-midnight)
    float iTimeDelta;
    int iFrame;
} pc;

const float PI = 3.14159265;
const float TAU = 6.28318531;

// Highlight headroom above diffuse white (>= 1.0; 1.0 on SDR).
float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

// Signed distance to a segment from the origin to `b`.
float segment(vec2 p, vec2 b) {
    float h = clamp(dot(p, b) / dot(b, b), 0.0, 1.0);
    return length(p - b * h);
}

// A hand pointing at clockwise angle `ang` (0 = up), length `len`, width `w`.
// Returns coverage in 0..1.
float hand(vec2 p, float ang, float len, float w) {
    vec2 dir = vec2(sin(ang), cos(ang)); // 0 = +y (up), clockwise
    return smoothstep(w, w * 0.5, segment(p, dir * len));
}

void main() {
    // Centered, aspect-correct coordinates; +y up, dial radius ~1.
    vec2 p = (2.0 * fragCoord - pc.iResolution) / min(pc.iResolution.x, pc.iResolution.y);

    vec3 col = mix(vec3(0.015, 0.02, 0.04), vec3(0.03, 0.035, 0.06), 0.5 + 0.5 * p.y);

    float r = length(p);

    // Dial rim and hour ticks.
    col += vec3(0.25) * smoothstep(0.02, 0.0, abs(r - 0.92));
    float a = atan(p.x, p.y);                 // 0 = up, clockwise
    float tick = abs(fract(a / TAU * 12.0 + 0.5) - 0.5) * (TAU / 12.0);
    float tickMask = smoothstep(0.03, 0.0, tick) * smoothstep(0.04, 0.06, 0.92 - r)
                   * step(0.82, r);
    col += vec3(0.3) * tickMask;

    // Split local seconds-since-midnight into hand angles.
    float secs = pc.iDate.w;
    float sAng = fract(secs / 60.0) * TAU;          // smooth (fractional) seconds
    float mAng = fract(secs / 3600.0) * TAU;        // minutes
    float hAng = fract(secs / 43200.0) * TAU;       // 12-hour

    // Hour and minute hands in soft white; second hand a thin warm highlight
    // pushed into HDR so it glints on capable outputs.
    col += vec3(0.85) * hand(p, hAng, 0.50, 0.022);
    col += vec3(0.95) * hand(p, mAng, 0.74, 0.014);
    col += vec3(1.0, 0.55, 0.35) * hand(p, sAng, 0.82, 0.006) * headroom();

    // Center hub.
    col += vec3(0.9) * smoothstep(0.03, 0.0, r);

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
