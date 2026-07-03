//! Speculative decoding must produce exactly the target-greedy sequence.

use flip::cache::KvCacheConfig;
use flip::forward::{BlockConfig, CpuKernel, LayerTensors};
use flip::generate::{GenerationConfig, Generator, Sampler};
use flip::speculative::SpeculativeDecoder;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn vec(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * s)
            .collect()
    }
}

/// Deterministic small generator keyed by `seed` (same seed → identical model).
fn build_generator(seed: u64) -> Generator<CpuKernel> {
    let vocab = 32usize;
    let hidden = 16usize;
    let cfg = BlockConfig {
        hidden_size: hidden,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
    };
    let mut rng = Rng::new(seed);
    let s = 0.05;
    let layers = vec![LayerTensors {
        q_proj: rng.vec(cfg.q_dim() * hidden, s),
        k_proj: rng.vec(cfg.kv_dim() * hidden, s),
        v_proj: rng.vec(cfg.kv_dim() * hidden, s),
        o_proj: rng.vec(hidden * cfg.q_dim(), s),
        gate_proj: rng.vec(cfg.intermediate_size * hidden, s),
        up_proj: rng.vec(cfg.intermediate_size * hidden, s),
        down_proj: rng.vec(hidden * cfg.intermediate_size, s),
        input_layernorm: vec![1.0; hidden],
        post_attention_layernorm: vec![1.0; hidden],
    }];
    let kernel = CpuKernel::new(cfg, layers).unwrap();
    let embedding = rng.vec(vocab * hidden, s);
    let lm_head = rng.vec(vocab * hidden, s);
    Generator::new(
        kernel,
        embedding,
        vec![1.0; hidden],
        lm_head,
        vocab,
        1e-5,
        KvCacheConfig { num_layers: 1, num_kv_heads: 2, head_dim: 4, block_size: 16 },
        64,
    )
    .unwrap()
}

fn greedy(gen: &Generator<CpuKernel>, prompt: &[u32], n: usize) -> Vec<u32> {
    gen.generate(
        prompt,
        &GenerationConfig { max_new_tokens: n, eos_token: None, sampler: Sampler::Greedy },
    )
    .unwrap()
}

#[test]
fn speculative_output_equals_target_greedy() {
    let prompt = [1u32, 2, 3];
    let n = 8;

    // Different draft (seed 2) vs. target (seed 1): output must still be exact.
    let decoder = SpeculativeDecoder::new(build_generator(1), build_generator(2), 4);
    let spec = decoder.generate(&prompt, n).unwrap();
    let reference = greedy(&build_generator(1), &prompt, n);

    assert_eq!(spec.tokens, reference, "speculative diverged from target-greedy");
    assert_eq!(spec.tokens.len(), n);
}

#[test]
fn identical_draft_is_fully_accepted() {
    let prompt = [5u32, 6];
    let n = 6;

    // Draft == target → every proposal matches, 100% acceptance.
    let decoder = SpeculativeDecoder::new(build_generator(1), build_generator(1), 4);
    let spec = decoder.generate(&prompt, n).unwrap();

    assert_eq!(spec.tokens, greedy(&build_generator(1), &prompt, n));
    assert!(spec.proposed > 0);
    assert_eq!(spec.accepted, spec.proposed);
    assert!((spec.acceptance_rate() - 1.0).abs() < 1e-9);
}
