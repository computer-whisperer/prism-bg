// Fog hills — an infinite raymarched terrain flythrough. The camera drifts
// forward over rolling fbm hills, a low sun bleeding HDR glow through layered
// haze; snow catches the high peaks. Genuine 3D: real perspective, parallax and
// distance fog, with the camera held ABOVE the heightfield by construction so it
// never clips through the ground. Uses iTime → animated.
//
//   prism-bg --shader examples/shaders/fog-hills.frag
//
// COST: a heightfield raymarch (~160 steps/pixel) plus a few fbm taps for
// shading — heavy at 4K. Cap the frame rate if your GPU runs warm:
//   prism-bg --shader examples/shaders/fog-hills.frag --fps 30
//
// Output is extended-linear with sRGB primaries (1.0 = reference white); the sun
// disk and its glow are pushed into HDR headroom.
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

// Low sun, slightly to the right and ahead — the glow source in the haze.
const vec3 SUN_DIR = vec3(0.30, 0.13, 0.945);   // normalized below in main use
// Terrain amplitude; the camera flies at an altitude safely above this.
const float AMP = 2.0;
const float CAM_Y = 3.3;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Terrain height at world (x,z). fbm, shaped toward broad valleys and sharper
// ridges, scaled into [~-0.3, AMP].
float terrainH(vec2 p) {
    p *= 0.16;
    float h = 0.0, a = 0.5, f = 1.0;
    for (int i = 0; i < 6; i++) {
        h += a * vnoise(p * f);
        f *= 2.02;
        a *= 0.5;
    }
    // h ∈ ~0..1; pow sharpens peaks and flattens valleys.
    return AMP * pow(clamp(h, 0.0, 1.0), 1.6) - 0.3;
}

vec3 terrainNormal(vec2 p) {
    vec2 e = vec2(0.03, 0.0);
    float hL = terrainH(p - e.xy), hR = terrainH(p + e.xy);
    float hD = terrainH(p - e.yx), hU = terrainH(p + e.yx);
    return normalize(vec3(hL - hR, 2.0 * e.x, hD - hU));
}

// Sky + sun, looking along rd.
vec3 sky(vec3 rd, vec3 sun) {
    float h = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 zenith  = vec3(0.16, 0.28, 0.52);
    vec3 horizon = vec3(0.62, 0.55, 0.52);
    vec3 col = mix(horizon, zenith, pow(h, 0.7));
    float s = max(dot(rd, sun), 0.0);
    // Tight HDR disk + a broad warm glow that hangs in the haze.
    col += vec3(1.0, 0.86, 0.62) * pow(s, 1800.0) * headroom() * 5.0;
    col += vec3(1.0, 0.66, 0.42) * pow(s, 6.0) * 0.55;
    col += vec3(1.0, 0.78, 0.55) * pow(s, 64.0) * 0.6;
    return col;
}

void main() {
    vec2 uv = (fragCoord - 0.5 * pc.iResolution) / pc.iResolution.y;
    float t = pc.iTime;
    vec3 sun = normalize(SUN_DIR);

    // Camera: forward flight with a lazy yaw sway, looking slightly downward.
    vec3 ro = vec3(0.0, CAM_Y, t * 3.0);
    float yaw = 0.18 * sin(t * 0.05);
    vec3 fwd = normalize(vec3(sin(yaw), -0.16, cos(yaw)));
    vec3 rgt = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
    vec3 up  = cross(fwd, rgt);
    vec3 rd  = normalize(uv.x * rgt + uv.y * up + 1.5 * fwd);

    // Heightfield raymarch: advance until the ray drops below the terrain. The
    // step is a fraction of the height gap (a conservative bound, since terrain
    // slopes can exceed 1), clamped so it never stalls or steps wildly far.
    float dist = 0.0;
    float maxd = 90.0;
    bool hit = false;
    vec3 pos = ro;
    for (int i = 0; i < 160; i++) {
        pos = ro + rd * dist;
        float gap = pos.y - terrainH(pos.xz);
        if (gap < 0.0015 * dist) { hit = true; break; }
        dist += max(gap * 0.4, 0.02 + 0.01 * dist);
        if (dist > maxd) break;
    }

    vec3 col;
    if (hit) {
        vec3 n = terrainNormal(pos.xz);
        float diff = max(dot(n, sun), 0.0);
        float skyl = 0.5 + 0.5 * n.y;                     // sky ambient
        float back = max(dot(n, normalize(vec3(-sun.x, 0.0, -sun.z))), 0.0) * 0.2;

        // Rock darkens in crevices, lightens on slopes; snow on high, flat-ish
        // ground.
        float height01 = clamp((pos.y + 0.3) / (AMP + 0.3), 0.0, 1.0);
        vec3 rock = mix(vec3(0.05, 0.055, 0.05), vec3(0.20, 0.17, 0.14), n.y);
        float snow = smoothstep(0.55, 0.8, height01) * smoothstep(0.55, 0.78, n.y);
        vec3 albedo = mix(rock, vec3(0.92, 0.94, 1.0), snow);

        vec3 sunCol = vec3(1.0, 0.82, 0.62);
        vec3 ambCol = vec3(0.30, 0.40, 0.60);
        col = albedo * (diff * sunCol * 1.3 + skyl * ambCol * 0.5 + back * sunCol);

        // Aerial perspective: melt distant terrain into the sky/haze.
        float fog = 1.0 - exp(-dist * 0.045);
        col = mix(col, sky(rd, sun), fog);
    } else {
        col = sky(rd, sun);
    }

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
