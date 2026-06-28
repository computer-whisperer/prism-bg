//! Parse a `.frag` into a render graph: the single displayed pass plus any
//! offscreen feedback/buffer passes it depends on.
//!
//! Three authoring levels, in increasing power:
//!
//! 1. **Plain** — a lone fragment shader. One pass straight to screen (as the
//!    feature shipped originally).
//! 2. **`iPrevFrame`** — the shader samples its own previous frame. It becomes
//!    one ping-pong buffer fed back into itself, presented by a built-in blit.
//!    Zero config; covers trails/decay/reaction-diffusion.
//! 3. **Multi-pass** — a `/*!prism … */` JSON metadata block declares named
//!    buffer passes and per-pass channel routing, with the passes themselves in
//!    `//!pass <name>` sections (and optional shared `//!common` code). This is
//!    the Shadertoy-style multi-buffer model (fluid, separable blur, bloom, …).
//!
//! The result is a GPU-agnostic [`GraphSpec`] (GLSL strings + resolved routing)
//! the renderer turns into pipelines and ping-pong textures.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

/// Maximum channels per pass (`iChannel0..3`), matching Shadertoy.
pub const MAX_CHANNELS: u32 = 4;

/// Where a pass's input channel reads from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelSource {
    /// Another buffer's output: its *current* frame if that buffer renders
    /// earlier this frame, else its *previous* frame (resolved at render time
    /// from pass order). Index into [`GraphSpec::buffers`].
    Buffer(usize),
    /// The owning buffer's own previous frame (feedback).
    SelfPrev,
    /// A static uploaded image. Index into [`GraphSpec::textures`].
    Texture(usize),
}

/// A static image channel declared in the metadata `textures` map: a name (for
/// routing) and a path as written (resolved relative to the `.frag` file by the
/// loader, not here — the parser stays filesystem-agnostic for testability).
#[derive(Debug, Clone)]
pub struct TextureSpec {
    pub name: String,
    pub path: String,
}

/// One input channel binding: which `iChannelN` and what it samples.
#[derive(Debug, Clone, Copy)]
pub struct Channel {
    /// `iChannel` index, also the `set = 1` binding number.
    pub index: u32,
    pub source: ChannelSource,
}

/// A single compiled-as-its-own-shader pass.
#[derive(Debug, Clone)]
pub struct PassSpec {
    /// `"image"` or a buffer name; for diagnostics.
    pub name: String,
    /// Full fragment GLSL (shared `//!common` code already prepended).
    pub glsl: String,
    pub channels: Vec<Channel>,
}

/// How the final, displayed pass is produced.
#[derive(Debug, Clone)]
pub enum ImageSpec {
    /// An explicit fragment shader rendered straight into the dmabuf.
    Explicit(PassSpec),
    /// A built-in blit presenting `buffers[idx]` (the `iPrevFrame` shortcut,
    /// where the user shader *is* the buffer and just needs displaying).
    ImplicitBlit(usize),
}

/// A fragment shader resolved into its render graph.
#[derive(Debug, Clone)]
pub struct GraphSpec {
    /// Offscreen buffer passes, in render order (empty for a plain shader).
    pub buffers: Vec<PassSpec>,
    /// Static image channels, indexed by [`ChannelSource::Texture`]. Empty
    /// unless the metadata declares a `textures` map.
    pub textures: Vec<TextureSpec>,
    pub image: ImageSpec,
}

impl GraphSpec {
    /// True if any pass samples a channel (i.e. there are buffers to allocate).
    pub fn has_buffers(&self) -> bool {
        !self.buffers.is_empty()
    }
}

/// Raw JSON shape of the `/*!prism … */` block.
#[derive(Deserialize)]
struct RawMeta {
    /// Buffer pass names, in render order.
    #[serde(default)]
    buffers: Vec<String>,
    /// `pass name → (channel index string → source string)`.
    #[serde(default)]
    channels: HashMap<String, HashMap<String, String>>,
    /// `texture name → path` (path resolved relative to the `.frag` by the
    /// loader). A channel routes to a texture by naming it, same as a buffer.
    #[serde(default)]
    textures: HashMap<String, String>,
}

/// Parse `source` into a [`GraphSpec`]. Authoring-level detection (plain /
/// `iPrevFrame` feedback / `/*!prism …*/` multi-pass) happens here, so callers
/// just hand over the file.
pub fn parse(source: &str) -> Result<GraphSpec> {
    if let Some(meta_json) = extract_metadata(source) {
        return parse_multipass(source, &meta_json);
    }
    if source.contains("iPrevFrame") {
        // One self-feeding buffer, presented by the built-in blit.
        return Ok(GraphSpec {
            buffers: vec![PassSpec {
                name: "main".into(),
                glsl: source.to_string(),
                channels: vec![Channel {
                    index: 0,
                    source: ChannelSource::SelfPrev,
                }],
            }],
            textures: Vec::new(),
            image: ImageSpec::ImplicitBlit(0),
        });
    }
    // Plain single pass.
    Ok(GraphSpec {
        buffers: Vec::new(),
        textures: Vec::new(),
        image: ImageSpec::Explicit(PassSpec {
            name: "image".into(),
            glsl: source.to_string(),
            channels: Vec::new(),
        }),
    })
}

/// Pull the JSON between `/*!prism` and the next `*/`. Scans every `/*!prism`
/// occurrence and returns the first whose payload looks like a JSON object
/// (starts with `{`), so a shader that merely *mentions* `/*!prism …*/` in a
/// doc comment doesn't get mistaken for the real metadata block.
fn extract_metadata(source: &str) -> Option<String> {
    let mut from = 0;
    while let Some(rel) = source[from..].find("/*!prism") {
        let start = from + rel + "/*!prism".len();
        from = start;
        let Some(end) = source[start..].find("*/") else {
            break;
        };
        let payload = source[start..start + end].trim();
        if payload.starts_with('{') {
            return Some(payload.to_string());
        }
    }
    None
}

fn parse_multipass(source: &str, meta_json: &str) -> Result<GraphSpec> {
    let meta: RawMeta = serde_json::from_str(meta_json)
        .context("parsing /*!prism …*/ metadata as JSON")?;
    let sections = split_sections(source);
    let common = sections.get("common").map(String::as_str).unwrap_or("");

    // Build buffer passes in declared order; record name→index for routing.
    let mut index_of: HashMap<&str, usize> = HashMap::new();
    for (i, name) in meta.buffers.iter().enumerate() {
        if name == "image" || name == "common" {
            bail!("buffer name {name:?} is reserved");
        }
        if index_of.insert(name.as_str(), i).is_some() {
            bail!("duplicate buffer name {name:?}");
        }
    }

    // Textures, in name order so the index is deterministic (routing is by name,
    // so order is internal only). A name can't be both a buffer and a texture.
    let mut texture_names: Vec<&str> = meta.textures.keys().map(String::as_str).collect();
    texture_names.sort_unstable();
    let mut texture_of: HashMap<&str, usize> = HashMap::new();
    let mut textures = Vec::with_capacity(texture_names.len());
    for (i, name) in texture_names.iter().enumerate() {
        if *name == "image" || *name == "common" || *name == "self" {
            bail!("texture name {name:?} is reserved");
        }
        if index_of.contains_key(name) {
            bail!("name {name:?} is declared as both a buffer and a texture");
        }
        texture_of.insert(name, i);
        textures.push(TextureSpec {
            name: (*name).to_string(),
            path: meta.textures[*name].clone(),
        });
    }

    let resolve = |pass: &str, is_image: bool| -> Result<Vec<Channel>> {
        let Some(routes) = meta.channels.get(pass) else {
            return Ok(Vec::new());
        };
        let mut channels = Vec::with_capacity(routes.len());
        for (idx_str, src) in routes {
            let index: u32 = idx_str
                .parse()
                .with_context(|| format!("channel index {idx_str:?} for pass {pass:?} is not a number"))?;
            if index >= MAX_CHANNELS {
                bail!("pass {pass:?}: channel index {index} out of range 0..{MAX_CHANNELS}");
            }
            let source = if src == "self" {
                if is_image {
                    bail!("pass \"image\": \"self\" is invalid (the image pass has no stored previous frame)");
                }
                ChannelSource::SelfPrev
            } else if let Some(&j) = index_of.get(src.as_str()) {
                ChannelSource::Buffer(j)
            } else if let Some(&t) = texture_of.get(src.as_str()) {
                ChannelSource::Texture(t)
            } else {
                bail!("pass {pass:?}: channel {index} references unknown buffer/texture {src:?}");
            };
            channels.push(Channel { index, source });
        }
        // Stable order so binding setup is deterministic.
        channels.sort_by_key(|c| c.index);
        Ok(channels)
    };

    // A metadata block with no `//!pass` sections is a plain single-pass shader
    // that just wants textures: the whole source is the image pass. (Buffers
    // need sections, so they're disallowed in this shorthand.)
    if sections.is_empty() {
        if !meta.buffers.is_empty() {
            bail!("metadata declares buffers but the shader has no //!pass sections");
        }
        return Ok(GraphSpec {
            buffers: Vec::new(),
            textures,
            image: ImageSpec::Explicit(PassSpec {
                name: "image".into(),
                glsl: source.to_string(),
                channels: resolve("image", true)?,
            }),
        });
    }

    let mut buffers = Vec::with_capacity(meta.buffers.len());
    for name in &meta.buffers {
        let body = sections
            .get(name)
            .with_context(|| format!("metadata lists buffer {name:?} but there is no //!pass {name} section"))?;
        buffers.push(PassSpec {
            name: name.clone(),
            glsl: concat_glsl(common, body),
            channels: resolve(name, false)?,
        });
    }

    let image_body = sections
        .get("image")
        .context("multi-pass shader needs a //!pass image section")?;
    let image = ImageSpec::Explicit(PassSpec {
        name: "image".into(),
        glsl: concat_glsl(common, image_body),
        channels: resolve("image", true)?,
    });

    Ok(GraphSpec {
        buffers,
        textures,
        image,
    })
}

/// Prepend shared `//!common` code (after the first `#version` line, which must
/// stay first) to a pass body. If `common` is empty, return the body as-is.
fn concat_glsl(common: &str, body: &str) -> String {
    if common.trim().is_empty() {
        return body.to_string();
    }
    // The `#version` directive must be the first non-comment token, so splice
    // common in right after it.
    if let Some(version_end) = body.find('\n').filter(|_| body.trim_start().starts_with("#version")) {
        let (head, rest) = body.split_at(version_end + 1);
        format!("{head}{common}\n{rest}")
    } else {
        format!("{common}\n{body}")
    }
}

/// Split a source into named sections on `//!pass <name>` and `//!common`
/// marker lines. Text before the first marker (e.g. the metadata comment) is
/// ignored.
fn split_sections(source: &str) -> HashMap<String, String> {
    let mut sections: HashMap<String, String> = HashMap::new();
    let mut current: Option<String> = None;
    let mut buf = String::new();
    for line in source.lines() {
        let trimmed = line.trim_start();
        let marker = trimmed
            .strip_prefix("//!pass ")
            .map(str::trim)
            .or_else(|| trimmed.strip_prefix("//!common").map(|_| "common"));
        if let Some(name) = marker {
            if let Some(cur) = current.take() {
                sections.insert(cur, std::mem::take(&mut buf));
            }
            current = Some(name.to_string());
        } else if current.is_some() {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    if let Some(cur) = current.take() {
        sections.insert(cur, buf);
    }
    sections
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_shader_is_single_pass() {
        let g = parse("#version 450\nvoid main(){}").unwrap();
        assert!(g.buffers.is_empty());
        assert!(matches!(g.image, ImageSpec::Explicit(_)));
    }

    #[test]
    fn iprevframe_becomes_one_buffer_with_blit() {
        let g = parse("#version 450\nlayout(set=1,binding=0) uniform sampler2D iPrevFrame;\nvoid main(){}").unwrap();
        assert_eq!(g.buffers.len(), 1);
        assert_eq!(g.buffers[0].channels.len(), 1);
        assert_eq!(g.buffers[0].channels[0].source, ChannelSource::SelfPrev);
        assert!(matches!(g.image, ImageSpec::ImplicitBlit(0)));
    }

    #[test]
    fn multipass_parses_buffers_and_routing() {
        let src = r#"/*!prism
{ "buffers": ["velocity", "dye"],
  "channels": {
    "velocity": {"0": "velocity", "1": "dye"},
    "dye": {"0": "dye", "1": "velocity"},
    "image": {"0": "dye"} } }
*/
//!common
float helper(){ return 1.0; }
//!pass velocity
#version 450
void main(){}
//!pass dye
#version 450
void main(){}
//!pass image
#version 450
void main(){}
"#;
        let g = parse(src).unwrap();
        assert_eq!(g.buffers.len(), 2);
        assert_eq!(g.buffers[0].name, "velocity");
        // velocity channel 0 = self (own prev), channel 1 = dye buffer.
        let v = &g.buffers[0].channels;
        assert_eq!(v[0].index, 0);
        assert_eq!(v[0].source, ChannelSource::Buffer(0)); // "velocity" resolves to itself
        assert_eq!(v[1].source, ChannelSource::Buffer(1)); // "dye"
        // common code is prepended after #version.
        assert!(g.buffers[0].glsl.contains("helper"));
        assert!(g.buffers[0].glsl.starts_with("#version") || g.buffers[0].glsl.trim_start().starts_with("#version"));
        match &g.image {
            ImageSpec::Explicit(p) => {
                assert_eq!(p.channels.len(), 1);
                assert_eq!(p.channels[0].source, ChannelSource::Buffer(1));
            }
            _ => panic!("expected explicit image pass"),
        }
    }

    #[test]
    fn doc_comment_mention_is_not_mistaken_for_metadata() {
        // A header that mentions `/*!prism …*/` in prose must not be parsed as
        // the metadata block; the real block (starting with `{`) wins.
        let src = r#"// docs: a real shader uses /*!prism …*/ to declare buffers.
/*!prism
{ "buffers": ["a"], "channels": { "a": {"0": "self"} } }
*/
//!pass a
#version 450
void main(){}
//!pass image
#version 450
void main(){}
"#;
        let g = parse(src).unwrap();
        assert_eq!(g.buffers.len(), 1);
        assert_eq!(g.buffers[0].name, "a");
    }

    #[test]
    fn self_in_image_pass_is_rejected() {
        let src = r#"/*!prism
{ "buffers": [], "channels": { "image": {"0": "self"} } }
*/
//!pass image
#version 450
void main(){}
"#;
        assert!(parse(src).is_err());
    }

    #[test]
    fn unknown_buffer_reference_is_rejected() {
        let src = r#"/*!prism
{ "buffers": ["a"], "channels": { "a": {"0": "ghost"} } }
*/
//!pass a
#version 450
void main(){}
//!pass image
#version 450
void main(){}
"#;
        assert!(parse(src).is_err());
    }

    #[test]
    fn texture_channel_resolves_with_plain_body() {
        // No //!pass sections: the whole source is the image pass, and the
        // channel routes to the declared texture.
        let src = r#"/*!prism
{ "textures": { "noise": "noise.png" }, "channels": { "image": {"0": "noise"} } }
*/
#version 450
void main(){}
"#;
        let g = parse(src).unwrap();
        assert!(g.buffers.is_empty());
        assert_eq!(g.textures.len(), 1);
        assert_eq!(g.textures[0].name, "noise");
        assert_eq!(g.textures[0].path, "noise.png");
        match &g.image {
            ImageSpec::Explicit(p) => {
                assert_eq!(p.channels.len(), 1);
                assert_eq!(p.channels[0].source, ChannelSource::Texture(0));
            }
            _ => panic!("expected explicit image pass"),
        }
    }

    #[test]
    fn buffers_and_textures_share_a_channel_namespace() {
        let src = r#"/*!prism
{ "buffers": ["sim"],
  "textures": { "noise": "n.png" },
  "channels": { "sim": {"0": "noise"}, "image": {"0": "sim", "1": "noise"} } }
*/
//!pass sim
#version 450
void main(){}
//!pass image
#version 450
void main(){}
"#;
        let g = parse(src).unwrap();
        assert_eq!(g.buffers[0].channels[0].source, ChannelSource::Texture(0));
        match &g.image {
            ImageSpec::Explicit(p) => {
                assert_eq!(p.channels[0].source, ChannelSource::Buffer(0));
                assert_eq!(p.channels[1].source, ChannelSource::Texture(0));
            }
            _ => panic!("expected explicit image pass"),
        }
    }

    #[test]
    fn name_used_as_both_buffer_and_texture_is_rejected() {
        let src = r#"/*!prism
{ "buffers": ["x"], "textures": { "x": "x.png" },
  "channels": { "image": {"0": "x"} } }
*/
//!pass x
#version 450
void main(){}
//!pass image
#version 450
void main(){}
"#;
        assert!(parse(src).is_err());
    }

    #[test]
    fn missing_pass_section_is_rejected() {
        let src = r#"/*!prism
{ "buffers": ["a"], "channels": {} }
*/
//!pass image
#version 450
void main(){}
"#;
        assert!(parse(src).is_err());
    }
}
