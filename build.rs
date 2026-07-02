//! Build script for `flip`.
//!
//! When the `cuda` feature is enabled we link against the CUDA runtime (`cudart`)
//! so the FFI hooks in `src/cuda/ffi.rs` (cudaMemGetInfo / cudaHostAlloc / ...)
//! resolve at link time. Honour `CUDA_PATH` if the toolkit lives outside the
//! default search path. Without the feature this is a no-op and the crate builds
//! anywhere — the storage engine and VRAM math have no GPU dependency.

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            // Windows toolkit ships libs under lib\x64; Linux under lib64.
            println!("cargo:rustc-link-search=native={cuda_path}/lib64");
            println!("cargo:rustc-link-search=native={cuda_path}/lib/x64");
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
