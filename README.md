# Qwen TTS Native

Rust workspace for on-device **Qwen3 TTS** using [ggml-org/ggml](https://github.com/ggml-org/ggml) and **GGUF** weights. The project references [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) for architecture and tensor naming, but does **not** link against it.

## Crates

| Crate | Role |
|-------|------|
| `ggml-sys` | CMake + bindgen FFI to `vendor/ggml` ([ggml](https://github.com/ggml-org/ggml) Git submodule) |
| `ggml` | Thin wrappers + `sys` re-export |
| `qwen3-tts` | GGUF loader, tokenizer, public synthesis API, direct GGML pipeline in progress |

## Prerequisites

- **CMake** on the PATH (for `ggml-sys` building the `vendor/ggml` submodule).

## Build

Fetch ggml first (submodule):

```bash
git submodule update --init --recursive
```

Then:

```bash
cargo build --workspace
cargo test --workspace
```

Optional Hugging Face helpers:

```bash
cargo build -p qwen3-tts --features hf
```

GPU / accelerator backends are Cargo features on `ggml-sys` (see [VERSIONS.md](VERSIONS.md)).

Common backend builds:

```bash
# Apple GPU path
cargo build -p qwen3-tts --features metal

# Cross-platform GPU path (requires Vulkan SDK / loader + glslc)
cargo build -p qwen3-tts --features vulkan
```

Runtime backend preference is automatic:

- Apple builds with `metal` prefer Metal and fall back to CPU if initialization fails.
- Non-Apple builds with `vulkan` prefer Vulkan and fall back to CPU if initialization fails.
- Builds without GPU features use CPU only.

## Models

Documented GGUF links and directory layout: [docs/models.md](docs/models.md).

## Tests

Fast tests run in CI; model-backed tests are opt-in: [docs/testing.md](docs/testing.md).

Criterion benchmarks can be driven through `xtask`, including backend-specific runs:

```bash
cargo run -p xtask -- bench cpu
cargo run -p xtask -- bench metal
cargo run -p xtask -- bench vulkan
```

## Godot / gdext

The `qwen3-tts` crate is a normal Rust library (`rlib`). A future Godot project can depend on it directly from a `gdext` crate without a separate `cdylib` ABI layer.
