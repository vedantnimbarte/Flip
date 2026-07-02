//! Storage engine: memory-mapped, zero-copy access to safetensors weights.

pub mod mmap_store;
pub mod safetensors;

pub use mmap_store::{MmapShard, MmapStore};
pub use safetensors::{Dtype, SafetensorsHeader, TensorInfo};
