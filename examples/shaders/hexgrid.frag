// Glowing hexagonal lattice with a pulse that ripples outward from the center.
// Each cell lights on its own phase as the wave passes. Uses iTime → animated.
// Cheap: a couple of mod()s and a distance per pixel.
//
//   prism-bg --shader examples/shaders/hexgrid.frag
//
// Original implementation of the standard hex-grid distance technique.
//
// CLUSTER TILING: instead of working in this output's own pixels, it builds a
// coordinate in the shared multi-monitor cluster space — so the lattice and
// the ripple form one continuous field across every monitor, with cells the
// same size everywhere. The cluster uniforms are y-up logical pixels with the
// origin at the cluster's bottom-left (same convention as fragCoord):
//   vec2 g = iOutputOffset + (fragCoord / iResolution) * iOutputSize;
// gives this fragment's position in cluster space; divide by iGlobalResolution
// for a 0..1 coordinate spanning the whole workspace.
#version 450
layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push {
    vec2 iResolution;        // this output, device px
    float iTime;
    float _pad;
    vec2 iOutputOffset;      // this output's bottom-left in cluster space, logical px
    vec2 iOutputSize;        // this output, logical px
    vec2 iGlobalResolution;  // whole cluster, logical px
} pc;

vec3 palette(float t) {
    vec3 a = vec3(0.5), b = vec3(0.5), c = vec3(1.0);
    vec3 d = vec3(0.15, 0.40, 0.65);
    return a + b * cos(6.28318 * (c * t + d));
}

// Tile the plane into hexagons. Returns the local coordinate within the cell
// (xy) and the cell's center id (zw).
vec4 hexCoords(vec2 uv) {
    vec2 r = vec2(1.0, 1.7320508);
    vec2 h = r * 0.5;
    vec2 a = mod(uv, r) - h;
    vec2 b = mod(uv - h, r) - h;
    vec2 gv = dot(a, a) < dot(b, b) ? a : b;
    return vec4(gv, uv - gv);
}

// Distance from a point to the hexagon center, in the hex metric: 0.5 at the
// edges.
float hexDist(vec2 p) {
    p = abs(p);
    return max(dot(p, vec2(0.5, 0.8660254)), p.x);
}

void main() {
    float t = pc.iTime;

    // Position in the shared cluster, centered on the whole workspace so the
    // ripple emanates from the middle of the desktop, not each monitor.
    vec2 g = pc.iOutputOffset + (fragCoord / pc.iResolution) * pc.iOutputSize;
    vec2 c = g - 0.5 * pc.iGlobalResolution;

    // Fixed cell size in logical px → identical hexagons on every monitor,
    // regardless of resolution or DPI. (Tune CELL to taste.)
    const float CELL = 200.0;
    vec4 hc = hexCoords(c / CELL);
    vec2 gv = hc.xy;
    vec2 id = hc.zw;

    // Border lines: bright where hexDist approaches the 0.5 edge.
    float edge = smoothstep(0.0, 0.06, 0.5 - hexDist(gv));
    float border = 1.0 - edge;

    // A ring expanding from the center; cells light as it reaches them.
    float dist = length(id) * 0.16;
    float wave = sin(dist * 4.0 - t * 1.5);
    float pulse = smoothstep(0.2, 1.0, wave);

    vec3 col = vec3(0.015, 0.02, 0.035);
    vec3 tint = palette(dist + t * 0.03);

    // Faint always-on cell fill, plus a stronger lit fill where the pulse is.
    col += tint * (0.05 + 0.4 * pulse) * edge;
    // Glowing borders, brightest on lit cells.
    col += tint * border * (0.15 + 0.5 * pulse);

    outColor = vec4(col, 1.0);
}
