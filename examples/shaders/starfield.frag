// Deep-sky starfield. Uses iTime → animated. Stars are drawn with a real
// magnitude distribution — most are faint sharp points, a rare few are bright
// with a soft halo that masters into the HDR highlight headroom and the
// brightest carry a restrained diffraction glint. A few depth planes parallax
// at once for genuine 3D depth, over a faint deep-blue→teal nebula wash.
//
// Shades in cluster space (iOutputOffset/iOutputSize/iGlobalResolution, y-up
// logical px like fragCoord) so one continuous sky spans every monitor instead
// of each output restarting the same field. See hexgrid.frag for the mapping.
//
//   prism-bg --shader examples/shaders/starfield.frag
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

const float TAU = 6.2831853;
const float U32 = 2.3283064e-10;  // 1 / 2^32

// Highlight headroom above diffuse white (>= 1.0; 1.0 on SDR).
float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

// PCG hashes (full period — the cheap fract(sin)/fract(p*k) hashes are visibly
// periodic when sampled with incrementing cell ids in parallel). pcg4d yields
// four decorrelated randoms in one mixing pass, so a star cell needs a single
// hash call instead of one per attribute.
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (s >> 22u) ^ s;
}
uvec4 pcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}
// Four randoms in [0,1) keyed on an integer cell.
vec4 cellRand(vec2 cell) {
    uvec2 q = uvec2(ivec2(cell));
    return vec4(pcg4d(uvec4(q, q.x ^ 0x9e3779b9u, q.y ^ 0x85ebca6bu))) * U32;
}
// Scalar hash for the nebula value noise.
float hash(vec2 p) {
    uvec2 q = uvec2(ivec2(floor(p)));
    return float(pcg(q.x ^ pcg(q.y))) * U32;
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i), b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0)), d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 3; i++) { s += a * vnoise(p); p = p * 2.02 + 11.3; a *= 0.5; }
    return s;
}

// One depth plane of stars at grid `density`, already scrolled.
//   reach=1, rich=1.0 : the near plane — accumulate the 3×3 neighbours so big
//                       stars' halos/glints cross cell boundaries seamlessly.
//   reach=0, rich=0.0 : far planes — tiny dense points, a single-cell lookup
//                       (center inset so the core can't clip) is identical for
//                       9× less work.
vec3 plane(vec2 uv, float density, float t, int reach, float rich) {
    vec2 gp = uv * density;
    vec2 base = floor(gp);
    vec3 acc = vec3(0.0);
    float hh = headroom();

    for (int dy = -reach; dy <= reach; dy++)
    for (int dx = -reach; dx <= reach; dx++) {
        vec2 cell = base + vec2(float(dx), float(dy));
        vec4 r = cellRand(cell);

        // Single-cell planes inset the center so the tiny core never clips at a
        // boundary; the 3×3 plane uses the full cell (neighbours cover spill).
        vec2 pos = mix(0.12 + 0.76 * r.xy, r.xy, rich);
        float d = length(gp - cell - pos);

        // Magnitude: pow skews the population faint, leaving a sparse few bright.
        float m2 = r.z * r.z; float mag = m2 * m2 * m2 * r.z;   // r.z^7
        float tw = 0.75 + 0.25 * sin(t * (0.6 + 1.4 * r.w) + r.x * TAU);

        // Rational falloffs (no exp): a tight core plus a soft brightness-scaled
        // halo (near plane only). Windowed to zero before the 3×3 boundary so no
        // faint contribution survives the sample-window edge as a square.
        float coreR = 0.035 * (0.7 + 0.8 * mag);
        float x = d / coreR;
        float core = 1.0 / (1.0 + x * x); core *= core;
        float halo = rich * mag * 0.12 / (1.0 + d * d * 45.0);
        float lum = (core + halo) * mag * tw;

        // Restrained 4-point diffraction glint on only the brightest few (near
        // plane). Same window keeps the arms from clipping into a square.
        float glint = rich * smoothstep(0.9, 1.0, mag);
        if (glint > 0.0) {
            vec2 rl = abs(gp - cell - pos);
            float spike = (1.0 / (1.0 + rl.x * 900.0)) / (1.0 + rl.y * 24.0)
                        + (1.0 / (1.0 + rl.y * 900.0)) / (1.0 + rl.x * 24.0);
            lum += glint * spike * 0.4 * mag * tw;
        }
        lum *= 1.0 - rich * smoothstep(0.6, 0.98, d);   // taper to 0 inside window

        // Star colour temperature (warm-white ↔ cool-blue), never rainbow.
        vec3 tint = mix(vec3(1.0, 0.92, 0.78), vec3(0.72, 0.85, 1.0), r.w);
        // Faint stars sit below white; the brightest master into the headroom.
        acc += tint * lum * mix(0.8, hh, mag);
    }
    return acc;
}

void main() {
    // Position in shared cluster space (y-up logical px), normalized by cluster
    // height so star size/density are consistent and the field is continuous
    // across the desktop. x then spans 0..aspect over the workspace.
    vec2 g = pc.iOutputOffset + (fragCoord / pc.iResolution) * pc.iOutputSize;
    vec2 uv = g / pc.iGlobalResolution.y;
    float t = pc.iTime;
    float hh = headroom();

    // Deep-space base, a touch warmer toward the bottom (global y → seamless).
    float gy = g.y / pc.iGlobalResolution.y;
    vec3 col = mix(vec3(0.020, 0.020, 0.045), vec3(0.0, 0.0, 0.012), gy);

    // Faint nebula wash: one low-frequency fbm in a coordinated cool palette,
    // slowly drifting. Colour comes from thresholds on the same field, so it
    // stays sub-white and reads as depth, not colour.
    float n = fbm(uv * 1.25 + vec2(0.015 * t, -0.008 * t));
    float wash = smoothstep(0.45, 1.0, n);
    vec3 nebCol = mix(vec3(0.02, 0.05, 0.12), vec3(0.03, 0.12, 0.15), smoothstep(0.3, 0.85, n));
    nebCol = mix(nebCol, vec3(0.06, 0.04, 0.13), smoothstep(0.75, 1.0, n));  // hint of violet
    col += nebCol * wash * (0.35 + 0.3 * n);

    // Parallax depth: one rich near plane (3×3, halos + glints) plus two cheap
    // far planes (single-cell points), near fast → far slow, grids decorrelated.
    col += plane(uv + vec2(1.0, 0.3) * (t * 0.060),        7.0, t, 1, 1.0) * 1.00;
    col += plane(uv + vec2(1.0, 0.3) * (t * 0.030) + 13.7, 14.0, t, 0, 0.0) * 0.70;
    col += plane(uv + vec2(1.0, 0.3) * (t * 0.014) + 27.4, 24.0, t, 0, 0.0) * 0.45;

    outColor = vec4(min(col, vec3(hh)), 1.0);
}
