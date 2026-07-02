//! Raw CUDA runtime FFI bindings, compiled only under the `cuda` feature.
//!
//! These map directly onto the CUDA Runtime API symbols in `libcudart`. Higher
//! layers never touch these declarations — they go through the safe wrappers in
//! [`crate::cuda`] and [`crate::memory::pinned`]. Keeping the `unsafe extern`
//! surface isolated to one file is what lets the rest of the crate stay
//! `#![forbid(unsafe_code)]`-clean in spirit.

#![allow(non_camel_case_types)]

use std::os::raw::{c_int, c_uint, c_void};

/// `cudaError_t`. `0` == `cudaSuccess`.
pub type cudaError_t = c_int;

/// `cudaSuccess`.
pub const CUDA_SUCCESS: cudaError_t = 0;

/// Flag for [`cudaHostAlloc`]: default page-locked, non-portable allocation.
pub const CUDA_HOST_ALLOC_DEFAULT: c_uint = 0x00;
/// Flag: memory is considered pinned by all CUDA contexts.
pub const CUDA_HOST_ALLOC_PORTABLE: c_uint = 0x01;
/// Flag: maps the allocation into the CUDA address space (zero-copy).
pub const CUDA_HOST_ALLOC_MAPPED: c_uint = 0x02;
/// Flag: allocation is write-combined — faster host→device DMA, slow host reads.
pub const CUDA_HOST_ALLOC_WRITE_COMBINED: c_uint = 0x04;

extern "C" {
    /// Query free and total device memory for the current device.
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;

    /// Allocate `size` bytes of page-locked (pinned) host memory suitable for
    /// asynchronous `cudaMemcpyAsync` DMA transfers.
    pub fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

    /// Free host memory previously allocated with [`cudaHostAlloc`].
    pub fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;

    /// Page-lock an existing host allocation in place (alternative to
    /// `cudaHostAlloc`; used when pinning an already-mmapped region).
    pub fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

    /// Un-pin a region previously registered with [`cudaHostRegister`].
    pub fn cudaHostUnregister(ptr: *mut c_void) -> cudaError_t;

    /// Return the string name of an error code (for diagnostics).
    pub fn cudaGetErrorString(error: cudaError_t) -> *const std::os::raw::c_char;
}
