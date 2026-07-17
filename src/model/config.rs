//! Model configuration parsed from a HuggingFace-style `config.json`.
//!
//! The VRAM profiler needs the shape parameters (hidden size, head counts,
//! layer count) to size both the streamed weight blocks and the KV cache.
//! Fields mirror the subset of `config.json` that `dlm` consumes in Phase 1.

use crate::error::{DlmError, Result};
use crate::forward::cpu::RopeScaling;
use serde::Deserialize;
use std::path::Path;

/// Numeric precision of the on-disk weights. Drives the bytes-per-parameter
/// term in the VRAM math.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// Full 32-bit weights (F32): 4 bytes per parameter.
    F32,
    /// Full 16-bit weights (FP16 / BF16): 2 bytes per parameter.
    Fp16,
    /// 8-bit quantization: 1 byte per parameter.
    Int8,
    /// 4-bit AWQ / GPTQ: 0.5 bytes per parameter (the Phase 1 default).
    Int4,
}

impl QuantScheme {
    /// Average bytes occupied by a single weight parameter under this scheme.
    /// Returned as `f64` because 4-bit packing is a fractional 0.5.
    pub fn bytes_per_param(self) -> f64 {
        match self {
            QuantScheme::F32 => 4.0,
            QuantScheme::Fp16 => 2.0,
            QuantScheme::Int8 => 1.0,
            QuantScheme::Int4 => 0.5,
        }
    }
}

/// Raw deserialization target matching HuggingFace `config.json` key names.
/// Kept private; callers get the validated [`ModelConfig`] instead.
#[derive(Debug, Deserialize)]
struct RawConfig {
    hidden_size: u32,
    num_attention_heads: u32,
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    num_hidden_layers: u32,
    vocab_size: u32,
    #[serde(default)]
    intermediate_size: Option<u32>,
    #[serde(default)]
    max_position_embeddings: Option<u32>,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rms_norm_eps: Option<f32>,
    /// Explicit per-head dimension. Most models omit it (it is then
    /// `hidden_size / num_attention_heads`), but some declare a `head_dim` that
    /// is *not* that quotient, and assuming the quotient loads them mis-shaped.
    #[serde(default)]
    head_dim: Option<u32>,
    /// RoPE frequency scaling. Long-context models are trained with this; it is
    /// not optional decoration, and ignoring it corrupts every position.
    #[serde(default)]
    rope_scaling: Option<RawRopeScaling>,
    /// EOS token id(s). HF configs use either a single int or an array (e.g.
    /// Llama-3 lists `<|eot_id|>` and `<|end_of_text|>`).
    #[serde(default)]
    eos_token_id: Option<EosField>,
    /// Quantization metadata for GPTQ/AWQ checkpoints, when present. Used only
    /// to reject formats the dequantizer would otherwise mis-decode silently.
    #[serde(default)]
    quantization_config: Option<QuantizationConfig>,
}

/// The subset of HF's `quantization_config` block dlm needs to decide whether it
/// can dequantize a checkpoint correctly. The dequantizer models canonical 4-bit
/// GPTQ (sequential nibble order, no act-order); anything else is refused up front
/// rather than producing plausible-looking garbage.
#[derive(Debug, Deserialize)]
struct QuantizationConfig {
    #[serde(default)]
    quant_method: Option<String>,
    #[serde(default)]
    bits: Option<u32>,
    #[serde(default)]
    group_size: Option<i64>,
    /// GPTQ act-order: weights are permuted by `g_idx` and must be un-permuted.
    #[serde(default)]
    desc_act: Option<bool>,
    /// `gptq_v2` stores the true zero-point; classic `gptq` stores `zero - 1`.
    #[serde(default)]
    checkpoint_format: Option<String>,
}

/// A packed-quantized checkpoint dlm can decode, as declared by `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedQuant {
    /// Weights per quantization group along the input dimension.
    pub group_size: usize,
}

/// Refuse quantized checkpoints the dequantizer can't decode correctly, with a
/// message naming the working alternative. Silent wrong output is worse than a
/// clear error — see [`crate::quant`] for what the canonical path handles.
/// Decide whether a packed-quantized checkpoint is one dlm can decode correctly,
/// returning its layout when it is.
///
/// Only the case that has been **validated against a real export** is accepted:
/// 4-bit GPTQ, `desc_act: false`, classic (`v1`) checkpoint format. Everything
/// else is refused by name rather than guessed at, because wrong-but-plausible
/// weights generate fluent nonsense — the worst failure mode there is.
fn check_quant_supported(q: &QuantizationConfig) -> Result<Option<PackedQuant>> {
    let method = q.quant_method.as_deref().unwrap_or("").to_ascii_lowercase();
    if method.is_empty() {
        return Ok(None); // a plain float checkpoint
    }
    if let Some(bits) = q.bits {
        if bits != 4 {
            return Err(DlmError::UnsupportedQuant(format!(
                "{bits}-bit {method} checkpoint; dlm decodes 4-bit only.                  Use an fp16/bf16 or 4-bit GPTQ (desc_act=false) checkpoint."
            )));
        }
    }
    match method.as_str() {
        "gptq" => {
            // act-order permutes rows by `g_idx`; decoding without un-permuting
            // silently scrambles every weight.
            if q.desc_act == Some(true) {
                return Err(DlmError::UnsupportedQuant(
                    "GPTQ checkpoint uses desc_act (act-order): its rows are permuted by                      g_idx and dlm does not un-permute them. Use a desc_act=false GPTQ                      export, or an fp16/bf16 checkpoint."
                        .into(),
                ));
            }
            // `gptq_v2` stores the true zero-point; classic `gptq` stores zero-1.
            // dlm's decoder assumes the classic convention (verified against a real
            // export) and has no v2 fixture to check the other against.
            match q.checkpoint_format.as_deref().map(|f| f.to_ascii_lowercase()) {
                None => {}
                Some(ref f) if f == "gptq" => {}
                Some(other) => {
                    return Err(DlmError::UnsupportedQuant(format!(
                        "GPTQ checkpoint_format {other:?} is not supported; dlm decodes the                          classic `gptq` format, whose zero-point convention it has been                          validated against."
                    )))
                }
            }
            let group_size = q.group_size.unwrap_or(-1);
            if group_size <= 0 {
                return Err(DlmError::UnsupportedQuant(format!(
                    "GPTQ checkpoint declares group_size {group_size}; dlm needs a positive                      per-group size (act-order/whole-row grouping is not supported)."
                )));
            }
            Ok(Some(PackedQuant { group_size: group_size as usize }))
        }
        "awq" => Err(DlmError::UnsupportedQuant(
            "AWQ packs its nibbles in a permuted order dlm does not unpack, and no real AWQ              fixture has been validated against. Use a 4-bit GPTQ (desc_act=false) or              fp16/bf16 checkpoint."
                .into(),
        )),
        other => Err(DlmError::UnsupportedQuant(format!(
            "unrecognized quant_method {other:?}; dlm loads fp16/bf16 and 4-bit GPTQ              (desc_act=false) checkpoints."
        ))),
    }
}

/// Raw `rope_scaling` block. HF spells the discriminant `rope_type` on newer
/// configs and `type` on older ones; accept either.
#[derive(Debug, Deserialize)]
struct RawRopeScaling {
    #[serde(default, alias = "type")]
    rope_type: Option<String>,
    #[serde(default)]
    factor: Option<f32>,
    #[serde(default)]
    low_freq_factor: Option<f32>,
    #[serde(default)]
    high_freq_factor: Option<f32>,
    #[serde(default)]
    original_max_position_embeddings: Option<u32>,
}

/// Convert a declared `rope_scaling` block into the [`RopeScaling`] the block
/// kernel applies.
///
/// A scaling type we do not implement is a hard error, never a silent skip: the
/// model was *trained* with that scaling, so ignoring it yields fluent-looking
/// garbage rather than an obvious failure. An explicit refusal is the only safe
/// behavior — see the `dlm` README on supported architectures.
fn parse_rope_scaling(r: &RawRopeScaling) -> Result<Option<RopeScaling>> {
    let kind = r.rope_type.as_deref().unwrap_or("").to_ascii_lowercase();
    let factor = r.factor.unwrap_or(1.0);
    match kind.as_str() {
        // `default`/absent means "no scaling" — plain RoPE.
        "" | "default" => Ok(None),
        "linear" => Ok(Some(RopeScaling::Linear { factor })),
        "llama3" => Ok(Some(RopeScaling::Llama3 {
            factor,
            low_freq_factor: r.low_freq_factor.unwrap_or(1.0),
            high_freq_factor: r.high_freq_factor.unwrap_or(4.0),
            original_max_position: r.original_max_position_embeddings.unwrap_or(8192) as f32,
        })),
        other => Err(DlmError::InvalidConfig(format!(
            "rope_scaling type {other:?} is not implemented; dlm supports \"linear\" and \
             \"llama3\". Running this model without its trained RoPE scaling would produce \
             incoherent output, so it is refused rather than silently mis-run."
        ))),
    }
}

/// `eos_token_id` as it appears in `config.json`: one id or a list of them.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosField {
    One(u32),
    Many(Vec<u32>),
}

/// Validated model geometry consumed by the profiler and storage planner.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model embedding / residual stream width (`d_model`).
    pub hidden_size: u32,
    /// Number of query attention heads.
    pub num_attention_heads: u32,
    /// Number of key/value heads. Equals `num_attention_heads` for vanilla MHA;
    /// smaller under Grouped-Query Attention (GQA), which shrinks the KV cache.
    pub num_kv_heads: u32,
    /// Number of transformer blocks — the layers `dlm` streams in and out.
    pub num_layers: u32,
    /// Vocabulary size (drives embedding + LM head parameter counts).
    pub vocab_size: u32,
    /// FFN inner dimension; falls back to `4 * hidden_size` when absent.
    pub intermediate_size: u32,
    /// Model's own maximum context, if declared.
    pub max_position_embeddings: Option<u32>,
    /// EOS token id(s) declared by the model; empty when the config omits them.
    /// Generation stops when any of these is produced.
    pub eos_token_ids: Vec<u32>,
    /// RoPE base frequency (default 10000).
    pub rope_theta: f32,
    /// RMSNorm epsilon (default 1e-5).
    pub rms_eps: f32,
    /// RoPE frequency scaling declared by the model, if any.
    pub rope_scaling: Option<RopeScaling>,
    /// Explicit per-head dim when the config declares one; otherwise `None` and
    /// [`head_dim`](Self::head_dim) falls back to `hidden_size / num_heads`.
    pub explicit_head_dim: Option<u32>,
    /// On-disk weight precision.
    pub quant: QuantScheme,
    /// Set when the checkpoint ships **already** packed-quantized (4-bit GPTQ)
    /// rather than as floats. Its codes are decoded as they are — no
    /// re-quantization — so the calibration the export paid for survives.
    pub packed_quant: Option<PackedQuant>,
}

impl ModelConfig {
    /// Load and validate a `config.json` from a model directory or file path.
    /// If `path` is a directory, `config.json` inside it is used — and any
    /// `generation_config.json` beside it is merged in (see [`merge_generation_config`]).
    pub fn from_path(path: impl AsRef<Path>, quant: QuantScheme) -> Result<Self> {
        let path = path.as_ref();
        let config_path = if path.is_dir() {
            path.join("config.json")
        } else {
            path.to_path_buf()
        };

        let bytes = std::fs::read(&config_path).map_err(|source| DlmError::Io {
            path: config_path.clone(),
            source,
        })?;

        let mut config = Self::from_json_bytes(&bytes, quant)?;

        // HuggingFace treats `generation_config.json` as authoritative for
        // generation parameters, and models routinely declare a *larger* EOS set
        // there than in config.json. Qwen2.5 lists only `<|im_end|>` in
        // config.json but both `<|im_end|>` and `<|endoftext|>` in the generation
        // config — miss the second and the model never stops: it emits the token,
        // generation runs on to the token limit, and the special token itself
        // leaks into the reply.
        if let Some(dir) = config_path.parent() {
            let gen_path = dir.join("generation_config.json");
            if let Ok(gen_bytes) = std::fs::read(&gen_path) {
                config.merge_generation_config(&gen_bytes)?;
            }
        }
        Ok(config)
    }

    /// Merge a `generation_config.json` over this config: union the EOS ids it
    /// declares into [`eos_token_ids`](Self::eos_token_ids). Stopping on any of
    /// them is correct, so a union (rather than a replace) is the safe merge.
    pub fn merge_generation_config(&mut self, bytes: &[u8]) -> Result<()> {
        #[derive(Deserialize)]
        struct GenConfig {
            #[serde(default)]
            eos_token_id: Option<EosField>,
        }
        let gen: GenConfig = serde_json::from_slice(bytes).map_err(|source| DlmError::Json {
            context: "generation_config.json".to_string(),
            source,
        })?;
        let extra = match gen.eos_token_id {
            Some(EosField::One(id)) => vec![id],
            Some(EosField::Many(ids)) => ids,
            None => Vec::new(),
        };
        for id in extra {
            if !self.eos_token_ids.contains(&id) {
                self.eos_token_ids.push(id);
            }
        }
        Ok(())
    }

    /// Parse a config from raw JSON bytes. Separated from [`from_path`] so it
    /// can be unit-tested without touching the filesystem.
    pub fn from_json_bytes(bytes: &[u8], quant: QuantScheme) -> Result<Self> {
        let raw: RawConfig = serde_json::from_slice(bytes).map_err(|source| DlmError::Json {
            context: "config.json".to_string(),
            source,
        })?;

        // Reject quant formats the decoder would silently mis-decode; keep the
        // layout of the one it can.
        let packed_quant = match &raw.quantization_config {
            Some(qc) => check_quant_supported(qc)?,
            None => None,
        };

        let config = ModelConfig {
            hidden_size: raw.hidden_size,
            num_attention_heads: raw.num_attention_heads,
            // Default to full multi-head attention when kv-heads is unspecified.
            num_kv_heads: raw.num_key_value_heads.unwrap_or(raw.num_attention_heads),
            num_layers: raw.num_hidden_layers,
            vocab_size: raw.vocab_size,
            intermediate_size: raw
                .intermediate_size
                .unwrap_or(raw.hidden_size.saturating_mul(4)),
            max_position_embeddings: raw.max_position_embeddings,
            eos_token_ids: match raw.eos_token_id {
                Some(EosField::One(id)) => vec![id],
                Some(EosField::Many(ids)) => ids,
                None => Vec::new(),
            },
            rope_theta: raw.rope_theta.unwrap_or(10000.0),
            rms_eps: raw.rms_norm_eps.unwrap_or(1e-5),
            rope_scaling: match &raw.rope_scaling {
                Some(r) => parse_rope_scaling(r)?,
                None => None,
            },
            explicit_head_dim: raw.head_dim,
            quant,
            packed_quant,
        };

        config.validate()?;
        Ok(config)
    }

    /// Reject configs that would make the VRAM math divide by zero or produce
    /// nonsense head dimensions.
    fn validate(&self) -> Result<()> {
        if self.num_attention_heads == 0 {
            return Err(DlmError::InvalidConfig(
                "num_attention_heads must be > 0".into(),
            ));
        }
        if self.num_layers == 0 {
            return Err(DlmError::InvalidConfig("num_hidden_layers must be > 0".into()));
        }
        if self.hidden_size == 0 {
            return Err(DlmError::InvalidConfig("hidden_size must be > 0".into()));
        }
        // Only the derived head_dim needs the divisibility guarantee; a config
        // that states head_dim outright is free to break the quotient relation.
        if self.explicit_head_dim.is_none() && self.hidden_size % self.num_attention_heads != 0 {
            return Err(DlmError::InvalidConfig(format!(
                "hidden_size ({}) is not divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.head_dim() % 2 != 0 {
            return Err(DlmError::InvalidConfig(format!(
                "head_dim ({}) must be even (RoPE rotates dimension pairs)",
                self.head_dim()
            )));
        }
        Ok(())
    }

    /// Per-head dimension: the config's explicit `head_dim` when it declares one,
    /// else `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> u32 {
        self.explicit_head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Rough total parameter count for the whole model, used to estimate the
    /// average size of one streamed transformer block.
    ///
    /// Approximates each transformer layer as attention projections
    /// (`4 * hidden^2`, folding GQA into a smaller KV share) plus the FFN
    /// (`3 * hidden * intermediate`, covering gate/up/down in SwiGLU MLPs),
    /// and adds the tied embedding + LM head (`2 * vocab * hidden`).
    pub fn estimated_total_params(&self) -> u64 {
        let h = self.hidden_size as u64;
        let inter = self.intermediate_size as u64;
        let kv_ratio = self.num_kv_heads as f64 / self.num_attention_heads as f64;

        // q (h*h) + o (h*h) + k,v scaled by the GQA ratio (2 * kv_ratio * h*h).
        let attn = (2.0 * (h * h) as f64 + 2.0 * kv_ratio * (h * h) as f64) as u64;
        let ffn = 3 * h * inter;
        let per_layer = attn + ffn;

        let blocks = per_layer * self.num_layers as u64;
        let embed_and_head = 2 * self.vocab_size as u64 * h;
        blocks + embed_and_head
    }
}
