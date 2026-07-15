# Releasing

Releases are cut by pushing a `v*` tag; `.github/workflows/release.yml` then
builds the prebuilt binaries (CPU + static-CUDA) and attaches them to the GitHub
Release. `install.sh` / `install.ps1` download those assets.

## Before you tag

CI covers the CPU path, real-model correctness, the CUDA **FFI** (`cargo check`),
and clippy — but **CI has no GPU**, so the CUDA kernels are never *executed* in
CI. That verification is manual and must happen every release:

On a machine with an NVIDIA GPU + CUDA toolkit (nvcc):

```sh
# 1. GPU↔CPU parity — the CUDA kernels must match the CPU oracle.
cargo test --release --features cuda-kernels --test gpu_parity

# 2. End-to-end on a real model, streaming through VRAM on the GPU.
cargo run --release --features cuda-kernels -- \
  serve --model-path models/<a-real-model> --device gpu --stream \
  --resident-layers 4 --safety-margin-gb 0.5 --port 8234
#    then, from another shell, confirm a correct completion:
curl -s http://127.0.0.1:8234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Capital of France? One word."}],"max_tokens":8,"temperature":0}'
```

Both must pass on real hardware. A green CI run alone does **not** prove the
shipped `-cuda-static` binary produces correct output — only the above does.

## Tag

```sh
# bump version in Cargo.toml first, commit, then:
git tag vX.Y.Z && git push origin vX.Y.Z
```
