//!luminance dark
// Orrery: a quiet clockwork of orbits centred on the desktop, each ring
// carrying a small glowing planet with a fading trail. The orbital plane is
// viewed at a tilt chosen from the cluster's aspect ratio, so the ellipses
// stretch to fill the whole desktop — near-circular on a single portrait
// monitor, wide and shallow on a broad monitor grid. Planets run at
// Kepler-ish speeds (inner ones faster) and brighten slightly on the near
// side of the plane. Uses iTime -> animated.
//
//   prism-bg --shader examples/shaders/orrery.frag
//
// One instrument across the whole cluster: orbits are centred on the
// desktop's midpoint and sized against its full width; line widths are
// normalized through fwidth() so they stay hairline-thin on any monitor.
// Cheap: a fixed loop of eight ring evaluations, no noise.
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

const float TAU = 6.28318530718;
const float U32 = 2.3283064e-10;  // 1 / 2^32
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (s >> 22u) ^ s;
}
// Four decorrelated uniforms for ring index i.
vec4 ringRand(uint i) {
    uint h = pcg(i * 0x9e3779b9u + 0x85ebca6bu);
    return vec4(pcg(h), pcg(h ^ 0x68bc21ebu), pcg(h ^ 0x02e5be93u), pcg(h ^ 0x967a889bu)) * U32;
}
float hash21(vec2 p) {
    p = fract(p * vec2(173.93, 341.27));
    p += dot(p, p + 21.13);
    return fract(p.x * p.y);
}

void main() {
    vec2 local = fragCoord / pc.iResolution;
    vec2 g = pc.iOutputOffset + local * pc.iOutputSize;
    vec2 gres = max(pc.iGlobalResolution, vec2(1.0));
    // Logical pixels around the cluster centre, y-up.
    vec2 p = g - 0.5 * gres;

    // Viewing tilt of the orbital plane: pick the compression that makes the
    // outermost orbit span ~92% of the width and ~85% of the height, so the
    // instrument fills the desktop whatever shape the monitor grid is.
    float aspect = gres.x / gres.y;
    float tilt = clamp(0.93 / aspect, 0.30, 0.80);
    float rmax = 0.46 * gres.x;

    // Deep blue-black field with a faint glow pooled around the hub,
    // stretched gently along the plane.
    float dc = length(vec2(p.x, p.y / max(2.0 * tilt, 0.6))) / (0.55 * gres.x);
    vec3 col = mix(vec3(0.020, 0.026, 0.046), vec3(0.004, 0.006, 0.012),
                   smoothstep(0.0, 1.0, dc));

    vec3 lineCol = vec3(0.42, 0.52, 0.68);
    for (uint i = 0u; i < 8u; i++) {
        vec4 rnd = ringRand(i);
        float rn = (float(i) + 0.4 + 0.5 * (rnd.x - 0.5)) / 8.0;
        float r = rmax * (0.16 + 0.84 * rn);
        // All orbits share the plane, with a whisper of per-ring inclination.
        float ringTilt = tilt * (1.0 + 0.10 * (rnd.z - 0.5));
        vec2 e = vec2(p.x, p.y / ringTilt);
        float f = length(e) - r;
        // fwidth() converts stretched-space distance to true screen pixels,
        // so the line stays hairline all the way around the ellipse.
        float pxd = abs(f) / max(fwidth(f), 1e-3);
        float ring = smoothstep(1.6, 0.0, pxd - 0.5);
        float theta = atan(e.y, e.x);
        // The far side of the plane sits a little dimmer than the near side.
        float side = 0.5 - 0.5 * sin(theta);
        float fade = (0.9 - 0.07 * float(i)) * mix(0.72, 1.10, side);
        col += lineCol * ring * 0.16 * fade;

        // Planet: Kepler-ish angular speed, alternating direction.
        float dir = (rnd.w > 0.5) ? 1.0 : -1.0;
        float speed = dir * 0.40 / pow(0.30 + rn, 1.5);
        float aP = rnd.y * TAU + pc.iTime * 0.06 * speed;

        // Trail: brightness along the ring decaying behind the planet.
        float behind = fract(dir * (aP - theta) / TAU);
        col += lineCol * ring * exp(-behind * 7.0) * 0.55;

        // The planet itself — a small hot core with a soft halo, mastered
        // into the HDR headroom, swelling slightly on the near side.
        vec2 planet = vec2(cos(aP), sin(aP)) * r;
        float pd = length(vec2(e.x - planet.x, (e.y - planet.y) * ringTilt));
        float near = 0.5 - 0.5 * sin(aP);
        float glow = mix(0.70, 1.20, near);
        vec3 tint = mix(vec3(1.0, 0.82, 0.55), vec3(0.62, 0.80, 1.0), rnd.z);
        tint = mix(tint, vec3(0.95, 0.62, 0.66), step(0.8, rnd.y));
        col += tint * exp(-pd * pd / (12.0 + 10.0 * near)) * glow * (0.4 + 0.6 * headroom());
        col += tint * exp(-pd / 22.0) * 0.10 * glow;
    }

    // A small still sun at the hub.
    float hub = length(p);
    col += vec3(1.0, 0.86, 0.62) * exp(-hub * hub / 60.0) * (0.5 + 0.5 * headroom());
    col += vec3(0.9, 0.7, 0.45) * exp(-hub / 90.0) * 0.08;

    // Grain so the dark field doesn't band.
    col += (hash21(g) - 0.5) * 0.003;

    outColor = vec4(clamp(col, vec3(0.0), vec3(headroom())), 1.0);
}
