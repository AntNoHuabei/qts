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

The ignored `qwen3-tts` tests now also cover WAV-only reference-audio conditioning:

- `integration_reference_wav_changes_conditioning` generates a small synthetic WAV, verifies that it produces a non-zero speaker embedding, and checks that conditioned synthesis still succeeds.
- No extra speaker-encoder model asset is required for this path.

Backend-specific compile checks:

```bash
# Apple
cargo check -p qwen3-tts --no-default-features --features metal

# Linux / Windows with Vulkan SDK + glslc available
cargo check -p qwen3-tts --no-default-features --features vulkan

# CLI crate (same feature flags pass through to qwen3-tts)
cargo check -p qwen3-tts-cli
```

## Layer C — golden numerics (planned)

Copy `reference/*.bin` from [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) and add component tests with explicit tolerances (tokenizer, encoder, transformer, vocoder).

## Layer D — heavy E2E (manual)

Python or upstream `compare_e2e.py`-style checks should stay out of default PR CI.

## Benchmarks

Use the `cargo xtask` alias (see `.cargo/config.toml`) to run Criterion with the intended backend feature enabled:

```bash
cargo xtask bench cpu
cargo xtask bench metal
cargo xtask bench vulkan
```

The Vulkan path requires a working Vulkan SDK / loader and `glslc` on the machine performing the build.

Runtime backend is controlled by **`QWEN3_TTS_BACKEND`** (`auto`, `cpu`, `metal`, `vulkan`). `cargo xtask profile vulkan` sets `QWEN3_TTS_BACKEND=vulkan` so macOS can use MoltenVK when the binary is built with `--features vulkan`.

### Synthesis stage profile (wall clock)

To see approximate time spent in tokenizer vs transformer (codec rollout) vs vocoder for one or more end-to-end runs:

```bash
cargo xtask profile cpu --model-dir "$QWEN3_TTS_MODEL_DIR" --text "hello" --frames 32 --runs 5
```

Optional: `--out target/profile-run1.wav` writes WAV from the first run only. The table is printed to stderr; use `--runs N` to average stage times over multiple iterations (useful after a warmup run with `--runs 1` first if you care about steady state).

## CLI smoke checks

With a valid model directory and a small reference WAV:

```bash
cargo run -p qwen3-tts-cli -- speaker-bin \
  --model-dir "$QWEN3_TTS_MODEL_DIR" \
  --wav testdata/hello.wav \
  --out target/hello.speaker.bin

cargo run -p qwen3-tts-cli -- synthesize \
  --model-dir "$QWEN3_TTS_MODEL_DIR" \
  --text "hello" \
  --speaker-bin target/hello.speaker.bin \
  --out target/hello.wav
```

Stage-2 prompt import can be smoke-tested with a protobuf prompt exported by the upstream helper:

```bash
uv sync
uv run export-voice-clone-prompt \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio testdata/hello.wav \
  --ref-text "hello" \
  --out target/hello.voice-clone-prompt.pb

cargo run -p qwen3-tts-cli -- synthesize \
  --model-dir "$QWEN3_TTS_MODEL_DIR" \
  --text "hello" \
  --voice-clone-prompt target/hello.voice-clone-prompt.pb \
  --out target/hello.from-prompt.wav

cargo run -p qwen3-tts-cli -- speaker-bin \
  --model-dir "$QWEN3_TTS_MODEL_DIR" \
  --voice-clone-prompt target/hello.voice-clone-prompt.pb \
  --out target/hello.from-prompt.speaker.bin
```

## `testdata/`

See [testdata/README.md](../testdata/README.md). Large or generated files under `testdata/` are listed in `.gitignore`; only `testdata/README.md` is intended to be versioned.
