//! Load a runnable CPU model from a memory-mapped safetensors checkpoint.
//!
//! Bridges the storage layer to the compute path: it reads the standard
//! HuggingFace-named tensors out of an [`MmapStore`], converts them to `f32`
//! ([`bytes_to_f32`] handles F32/F16/BF16), and assembles a
//! [`Generator`] over a [`CpuKernel`] ready to run [`generate`].
//!
//! This is the connector that lets `flip generate --model-path <dir>` run a real
//! (small) model on CPU. Quantized checkpoints (AWQ/GPTQ `qweight` triplets)
//! would be materialized through the [`quant`](crate::quant) dequant kernel here
//! — this loader covers the float dtypes small models ship in.
//!
//! [`bytes_to_f32`]: crate::storage::bytes_to_f32
//! [`generate`]: crate::generate::Generator::generate

use crate::cache::KvCacheConfig;
use crate::error::{FlipError, Result};
use crate::forward::{BlockConfig, CpuKernel, LayerTensors};
use crate::generate::Generator;
use crate::model::ModelConfig;
use crate::storage::{bytes_to_f32, MmapStore};

/// Read a named tensor as `f32`, verifying it has exactly `expected_len` elements.
fn load_tensor(store: &MmapStore, name: &str, expected_len: usize) -> Result<Vec<f32>> {
    let (shard, info) = store
        .locate(name)
        .ok_or_else(|| FlipError::UnknownTensor(name.to_string()))?;
    let values = bytes_to_f32(shard.tensor_bytes(name)?, info.dtype)?;
    if values.len() != expected_len {
        return Err(FlipError::InvalidConfig(format!(
            "tensor {name:?}: expected {expected_len} elements, got {}",
            values.len()
        )));
    }
    Ok(values)
}

/// Build a CPU [`Generator`] from a mapped checkpoint and its config.
///
/// `max_context` sizes the KV block pool (tokens the sequence may reach). Uses
/// the standard `model.layers.{i}.*`, `model.embed_tokens.weight`,
/// `model.norm.weight`, and `lm_head.weight` names; a tied LM head (missing
/// `lm_head.weight`) falls back to the embedding matrix.
pub fn load_generator(
    store: &MmapStore,
    config: &ModelConfig,
    max_context: u32,
) -> Result<Generator<CpuKernel>> {
    let hidden = config.hidden_size as usize;
    let num_heads = config.num_attention_heads as usize;
    let num_kv_heads = config.num_kv_heads as usize;
    let head_dim = config.head_dim() as usize;
    let intermediate = config.intermediate_size as usize;
    let vocab = config.vocab_size as usize;

    let cfg = BlockConfig {
        hidden_size: hidden,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size: intermediate,
        rope_theta: config.rope_theta,
        rms_eps: config.rms_eps,
    };
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();

    let mut layers = Vec::with_capacity(config.num_layers as usize);
    for i in 0..config.num_layers {
        let name = |suffix: &str| format!("model.layers.{i}.{suffix}");
        let tensors = LayerTensors {
            q_proj: load_tensor(store, &name("self_attn.q_proj.weight"), q_dim * hidden)?,
            k_proj: load_tensor(store, &name("self_attn.k_proj.weight"), kv_dim * hidden)?,
            v_proj: load_tensor(store, &name("self_attn.v_proj.weight"), kv_dim * hidden)?,
            o_proj: load_tensor(store, &name("self_attn.o_proj.weight"), hidden * q_dim)?,
            gate_proj: load_tensor(store, &name("mlp.gate_proj.weight"), intermediate * hidden)?,
            up_proj: load_tensor(store, &name("mlp.up_proj.weight"), intermediate * hidden)?,
            down_proj: load_tensor(store, &name("mlp.down_proj.weight"), hidden * intermediate)?,
            input_layernorm: load_tensor(store, &name("input_layernorm.weight"), hidden)?,
            post_attention_layernorm: load_tensor(
                store,
                &name("post_attention_layernorm.weight"),
                hidden,
            )?,
        };
        tensors.validate(&cfg)?;
        layers.push(tensors);
    }
    let kernel = CpuKernel::new(cfg, layers)?;

    let embedding = load_tensor(store, "model.embed_tokens.weight", vocab * hidden)?;
    let final_norm = load_tensor(store, "model.norm.weight", hidden)?;
    // Weight tying: reuse the embedding when there is no separate LM head.
    let lm_head = if store.locate("lm_head.weight").is_some() {
        load_tensor(store, "lm_head.weight", vocab * hidden)?
    } else {
        embedding.clone()
    };

    let kv_config = KvCacheConfig {
        num_layers: config.num_layers,
        num_kv_heads: num_kv_heads as u32,
        head_dim: head_dim as u32,
        block_size: 16,
    };
    let kv_blocks = (max_context as u64).div_ceil(16) as u32 + 2;

    Generator::new(
        kernel,
        embedding,
        final_norm,
        lm_head,
        vocab,
        config.rms_eps,
        kv_config,
        kv_blocks,
    )
}
