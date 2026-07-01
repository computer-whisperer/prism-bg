//!luminance bright
// Watercolor: translucent washes of pigment blooming slowly across cold-press
// paper. Each layer is an fbm field cut at a breathing threshold; pigment
// multiplies over the paper with the darkened rim real watercolor leaves as a
// wash dries, plus granulation settling into the paper tooth. Uses iTime ->
// animated (the washes migrate over a minute-scale cycle).
//
//   prism-bg --shader examples/shaders/watercolor.frag
//
// One sheet of paper across the whole cluster. Moderate cost: one 4-octave
// fbm per pigment layer (three layers) plus paper noise.
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

float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

float hash21(vec2 p) {
    p = fract(p * vec2(141.13, 289.97));
    p += dot(p, p + 24.31);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        s += a * noise(p);
        p = mat2(1.68, 1.02, -1.02, 1.68) * p + vec2(11.0, 47.0);
        a *= 0.5;
    }
    return s;
}

// One wash: pigment multiplied over `col`. The field is cut at a slowly
// breathing threshold; the rim just inside the cut gets the extra deposit.
vec3 wash(vec3 col, vec2 p, vec3 pigment, vec2 offset, float scale, float phase, float grain) {
    float t = pc.iTime * 0.010;
    float f = fbm(p * scale + offset + vec2(t, -t * 0.6) + phase);
    float th = 0.535 + 0.045 * sin(pc.iTime * 0.017 + phase * 5.0);

    float body = smoothstep(th, th + 0.10, f);
    float rim = smoothstep(th, th + 0.025, f) * (1.0 - smoothstep(th + 0.030, th + 0.13, f));
    float deposit = body * 0.30 + rim * 0.42;
    deposit *= 0.90 + 0.20 * grain;  // granulation settles into the tooth
    return col * mix(vec3(1.0), pigment, clamp(deposit, 0.0, 1.0));
}

void main() {
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gv = g / max(pc.iGlobalResolution, vec2(1.0));
    float aspect = pc.iGlobalResolution.x / max(pc.iGlobalResolution.y, 1.0);
    vec2 p = (gv - 0.5) * vec2(aspect, 1.0);

    // Cold-press paper: warm white, low-frequency mottle, fine tooth.
    float tooth = noise(g * 0.31);
    vec3 col = vec3(0.945, 0.925, 0.885);
    col *= 0.975 + 0.04 * noise(p * 6.0);
    col *= 0.985 + 0.024 * tooth;

    // Three overlapping pigments, coolest laid down last.
    col = wash(col, p, vec3(0.86, 0.56, 0.42), vec2(3.1, 9.7), 1.15, 0.0, tooth);   // burnt sienna
    col = wash(col, p, vec3(0.83, 0.70, 0.38), vec2(27.4, 1.3), 1.55, 1.7, tooth);  // yellow ochre
    col = wash(col, p, vec3(0.34, 0.52, 0.58), vec2(15.8, 33.2), 1.35, 3.9, tooth); // cerulean grey

    // Faint deckle-edge vignette, like the sheet lifting off the board.
    float vignette = smoothstep(1.6, 0.5, length(p));
    col *= 0.93 + 0.07 * vignette;

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
