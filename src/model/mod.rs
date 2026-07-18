//! Model description types (geometry + quantization) shared across the engine.

pub mod config;
pub mod naming;

pub use config::{ModelConfig, MoeConfig, MoeNaming, PackedQuant, QuantScheme};
pub use naming::{classify, TensorRole};
