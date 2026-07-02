//! Safe CUDA runtime surface.
//!
//! Everything here compiles in both modes:
//! * With `--features cuda`, calls dispatch to real `libcudart` symbols.
//! * Without it, GPU-only operations return [`FlipError::CudaUnavailable`] so
//!   the storage engine and VRAM math remain fully exercisable off-GPU.

use crate::error::{FlipError, Result};

#[cfg(feature = "cuda")]
pub mod ffi;

/// Free/total device memory in bytes, as reported by `cudaMemGetInfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemory {
    pub free: u64,
    pub total: u64,
}

/// Query the current device's free and total VRAM.
///
/// This is the runtime source of `M_free` in the profiler's `Layers To Load`
/// formula (`specs.md` §3.1).
#[cfg(feature = "cuda")]
pub fn mem_get_info() -> Result<DeviceMemory> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    // SAFETY: both out-pointers reference live stack storage for the call.
    let code = unsafe { ffi::cudaMemGetInfo(&mut free, &mut total) };
    if code != ffi::CUDA_SUCCESS {
        return Err(FlipError::Cuda {
            api: "cudaMemGetInfo",
            code,
        });
    }
    Ok(DeviceMemory {
        free: free as u64,
        total: total as u64,
    })
}

/// Off-GPU build: no device to query.
#[cfg(not(feature = "cuda"))]
pub fn mem_get_info() -> Result<DeviceMemory> {
    Err(FlipError::CudaUnavailable("cudaMemGetInfo"))
}

/// Whether this binary was compiled with real CUDA support.
pub const fn is_available() -> bool {
    cfg!(feature = "cuda")
}
