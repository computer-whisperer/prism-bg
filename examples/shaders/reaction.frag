// Reaction–diffusion — a living Gray-Scott membrane. Two virtual chemicals
// react and diffuse in a feedback buffer; coral / fingerprint / maze patterns
// nucleate from scattered seeds and crawl outward forever, never repeating.
// The simulation runs in an fp16 ping-pong buffer (one update per frame); the
// image pass colour-maps and shades it.
//
//   prism-bg --shader examples/shaders/reaction.frag
//
// MULTI-PASS + FEEDBACK: the "sim" buffer reads its own previous frame ("self")
// to step the PDE; "image" reads "sim" to display it. Buffers are fp16 and
// sampled y-flipped (fragCoord is y-up, textures y-down) — sim() flips so a
// pixel reads ITSELF and its true neighbours. The pattern scale is set by STEP
// (the Laplacian stride in texels): bigger STEP → coarser, more visible coral.
//
// Patterns grow over ~30–60 s from the seeds, then evolve indefinitely. Output
// is extended-linear with sRGB primaries (1.0 = reference white).
/*!prism
{
  "buffers": ["sim"],
  "channels": {
    "sim":   {"0": "self"},
    "image": {"0": "sim"}
  }
}
*/

//!common
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
    vec4 iDate;
    float iTimeDelta;
    int iFrame;       // frame counter — used to seed the first frames
} pc;

// Highlight headroom above diffuse white (>= 1.0; 1.0 on SDR).
float headroom() { return max(pc.iMaxLum / max(pc.iRefWhite, 1.0), 1.0); }

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

// Laplacian stride, in texels. Larger → coarser patterns (a few px per cell is
// invisible on 4K; this fattens the coral to a comfortable scale).
const float STEP = 2.0;

//!pass sim
#version 450
// One explicit Euler step of the Gray-Scott model. U,V are stored in R,G of the
// fp16 buffer. Read this pixel and its neighbours from the previous frame.
layout(set = 1, binding = 0) uniform sampler2D iChannel0; // self, previous frame

// Sample the sim buffer at a y-flipped uv (see header), so the feedback reads
// the matching texel rather than its vertical mirror.
vec2 chem(vec2 uv) { return texture(iChannel0, vec2(uv.x, 1.0 - uv.y)).rg; }

void main() {
    vec2 uv = fragCoord / pc.iResolution;
    vec2 px = STEP / pc.iResolution;

    // Seed the opening frames: U = 1 everywhere, V dabbed in scattered spots so
    // growth nucleates all across the screen.
    if (pc.iFrame < 4) {
        vec2 cell = floor(fragCoord / 64.0);
        float seed = step(0.88, hash21(cell));          // ~12% of cells
        // A soft disc of V within each seeded cell.
        vec2 c = (cell + 0.5) * 64.0;
        float disc = seed * smoothstep(20.0, 6.0, distance(fragCoord, c));
        outColor = vec4(1.0, 0.5 * disc, 0.0, 1.0);
        return;
    }

    vec2 c = chem(uv);
    // 3×3 Laplacian (Karl Sims weights: centre −1, orthogonal 0.2, diagonal 0.05).
    vec2 lap = -c
        + 0.20 * (chem(uv + vec2( px.x, 0.0)) + chem(uv + vec2(-px.x, 0.0))
                + chem(uv + vec2(0.0,  px.y)) + chem(uv + vec2(0.0, -px.y)))
        + 0.05 * (chem(uv + vec2( px.x,  px.y)) + chem(uv + vec2(-px.x,  px.y))
                + chem(uv + vec2( px.x, -px.y)) + chem(uv + vec2(-px.x, -px.y)));

    float U = c.r, V = c.g;
    const float Du = 1.0, Dv = 0.5, F = 0.0545, k = 0.062, dt = 1.0;
    float reaction = U * V * V;
    float dU = Du * lap.r - reaction + F * (1.0 - U);
    float dV = Dv * lap.g + reaction - (F + k) * V;

    U = clamp(U + dU * dt, 0.0, 1.0);
    V = clamp(V + dV * dt, 0.0, 1.0);
    outColor = vec4(U, V, 0.0, 1.0);
}

//!pass image
#version 450
// Colour-map V and fake-light it from its gradient for a wet, raised-membrane
// look.
layout(set = 1, binding = 0) uniform sampler2D iChannel0; // sim

float vAt(vec2 uv) { return texture(iChannel0, vec2(uv.x, 1.0 - uv.y)).g; }

vec3 palette(float t) {
    return 0.5 + 0.5 * cos(6.28318 * (t + vec3(0.0, 0.20, 0.45)));
}

void main() {
    vec2 uv = fragCoord / pc.iResolution;
    vec2 px = 1.0 / pc.iResolution;

    float v = vAt(uv);
    // Gradient of V → a surface normal, for shading the ridges.
    float gx = vAt(uv + vec2(px.x, 0.0)) - vAt(uv - vec2(px.x, 0.0));
    float gy = vAt(uv + vec2(0.0, px.y)) - vAt(uv - vec2(0.0, px.y));
    vec3 n = normalize(vec3(-gx * 7.0, -gy * 7.0, 1.0));

    // Curvature-based ambient occlusion: compare V to a wider-radius average.
    // Concave grooves and ring interiors (centre below their surroundings)
    // recess and darken; convex crests (centre above) round outward and catch
    // more light. This is what turns flat "cheerio" rings into raised tubes.
    vec2 aoR = 4.0 * px;
    float wide = 0.25 * (vAt(uv + vec2(aoR.x, 0.0)) + vAt(uv - vec2(aoR.x, 0.0))
                       + vAt(uv + vec2(0.0, aoR.y)) + vAt(uv - vec2(0.0, aoR.y)));
    float curv = v - wide;
    float occ   = smoothstep(0.0, -0.12, curv);   // 1 deep in a groove
    float crest = smoothstep(0.0,  0.12, curv);    // 1 on a ridge top
    float ao = mix(1.0, 0.30, occ);

    // Raking key light for relief + a soft hemispheric fill so valleys aren't
    // black. The grazing angle throws directional shading across the ridges.
    vec3 keyDir = normalize(vec3(0.82, 0.42, 0.30));
    float key  = max(dot(n, keyDir), 0.0);
    float fill = 0.5 + 0.5 * n.z;                  // ambient from above
    float spec = pow(max(dot(n, normalize(keyDir + vec3(0.0, 0.0, 1.0))), 0.0), 32.0);

    // Fresnel rim: steep ridge walls (normal tilted away from the viewer) catch
    // a wet glint that separates the tubes from the substrate.
    float rim = pow(1.0 - clamp(n.z, 0.0, 1.0), 3.0) * smoothstep(0.10, 0.30, v);

    // Deep substrate where V≈0, warm crests where V is high; drift the palette
    // slowly so the colour scheme breathes.
    vec3 mem = palette(0.55 + v * 1.4 + pc.iTime * 0.01);
    vec3 base = mix(vec3(0.02, 0.03, 0.06), mem, smoothstep(0.05, 0.35, v));

    vec3 col = base * (0.18 * fill + 0.90 * key) * ao;
    col += base * crest * 0.25;                                       // convex crests read brighter
    col += vec3(0.8, 0.9, 1.0) * rim * 0.35 * headroom();             // cool wet rim
    col += vec3(1.0, 0.95, 0.85) * spec * smoothstep(0.2, 0.45, v) * ao * headroom();

    outColor = vec4(min(col, vec3(headroom())), 1.0);
}
