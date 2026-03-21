# Testing

## Layer A — always-on (`cargo test --workspace`)

- `ggml-sys`: minimal GGML graph smoke test.
- `ggml`: wrapper smoke test.
- `qwen3-tts`: request defaults, `ModelPaths` helpers.

No network, no large files.

## Layer B — local / optional CI (`#[ignore]`)

The direct Rust path currently validates GGUF files, loads tokenizer metadata, and exercises the public synthesis entrypoint up to the not-yet-ported transformer/vocoder stages.

Set `QWEN3_TTS_MODEL_DIR` to a directory containing:

- `qwen3-tts-0.6b-f16.gguf`
- `qwen3-tts-tokenizer-f16.gguf`

Then:

```bash
export QWEN3_TTS_MODEL_DIR=/path/to/models
cargo test -p qwen3-tts -- --ignored
```

Backend-specific compile checks:

```bash
# Apple
cargo check -p qwen3-tts --no-default-features --features metal

# Linux / Windows with Vulkan SDK + glslc available
cargo check -p qwen3-tts --no-default-features --features vulkan
```

## Layer C — golden numerics (planned)

Copy `reference/*.bin` from [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) and add component tests with explicit tolerances (tokenizer, encoder, transformer, vocoder).

## Layer D — heavy E2E (manual)

Python or upstream `compare_e2e.py`-style checks should stay out of default PR CI.

## Benchmarks

Use `xtask` to run Criterion with the intended backend feature enabled:

```bash
cargo run -p xtask -- bench cpu
cargo run -p xtask -- bench metal
cargo run -p xtask -- bench vulkan
```

The Vulkan path requires a working Vulkan SDK / loader and `glslc` on the machine performing the build.

## `testdata/`

See [testdata/README.md](../testdata/README.md). Large or generated files under `testdata/` are listed in `.gitignore`; only `testdata/README.md` is intended to be versioned.
