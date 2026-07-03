//! Weight quantization kernels — CPU dequantization of streamed 4-bit weights.

pub mod dequant;
pub mod packed;

pub use dequant::{pack_codes, quantize_affine, Quant4Tensor};
pub use packed::{dequantize_gptq_4bit, pack_gptq_4bit, PackedQuantConfig};
