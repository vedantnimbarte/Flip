//! Weight quantization kernels — CPU dequantization of streamed 4-bit weights.

pub mod dequant;

pub use dequant::{pack_codes, quantize_affine, Quant4Tensor};
