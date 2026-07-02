//! Dynamic VRAM profiling: how many transformer blocks fit resident at once.

pub mod vram;

pub use vram::{VramPlan, VramProfiler, DEFAULT_SAFETY_MARGIN_BYTES};
