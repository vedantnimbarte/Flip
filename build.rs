//! Build script for `flip`.
//!
//! Links the selected GPU runtime so the FFI in `src/gpu/` resolves at link
//! time: `cuda` → `cudart` (honours `CUDA_PATH`), `rocm` → `amdhip64` (honours
//! `ROCM_PATH`). With neither feature this is a no-op and the crate builds
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

    if std::env::var("CARGO_FEATURE_ROCM").is_ok() {
        // ROCm default install is /opt/rocm; libs live under lib.
        let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        println!("cargo:rustc-link-search=native={rocm_path}/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
}
