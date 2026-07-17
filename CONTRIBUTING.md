# Contributing to dlm

Thanks for your interest in contributing!

## Getting started

1. Fork and clone the repo.
2. Build: `cargo build --release` (add `--features cuda-kernels` if you have an NVIDIA GPU and the CUDA toolkit).
3. Test: `cargo test`. GPU-dependent tests only run with `--features cuda-kernels` on a machine with a CUDA GPU.

## Making changes

- Open an issue first for anything non-trivial so we can discuss the approach.
- Keep PRs focused — one change per PR.
- Run `cargo fmt` and `cargo clippy` before pushing.
- Add or update tests for behavior changes (integration tests live in `tests/`).
- Use clear commit messages (`fix(stream): ...`, `feat(quant): ...`, `docs: ...` — match the existing history).

## Reporting bugs

Open an issue with your OS, GPU, dlm version, the exact command you ran, and the full output.

## License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 license](LICENSE.md).
