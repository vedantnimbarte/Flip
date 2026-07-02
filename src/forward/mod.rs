//! Forward-pass orchestration skeleton.
//!
//! Ties streaming, dequantization, the compute kernel, the residual activation
//! pool, and the paged KV cache into a per-layer forward pass. The transformer
//! math is abstracted behind [`ComputeKernel`] so a real GPU kernel replaces the
//! [`StubKernel`] without touching the orchestration.

pub mod cpu;
pub mod kernel;
pub mod orchestrator;

pub use cpu::{decode_block, BlockConfig, KvLayerCache, LayerTensors};
pub use kernel::{ComputeKernel, LayerWeights, StubKernel};
pub use orchestrator::{ForwardConfig, ForwardOrchestrator};
