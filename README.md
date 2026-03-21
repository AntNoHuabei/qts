# Qwen TTS Native

Rust workspace for on-device **Qwen3 TTS** using [ggml-org/ggml](https://github.com/ggml-org/ggml) and **GGUF** weights. The project references [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) for architecture and tensor naming, but does **not** link against it.

## Crates

| Crate | Role |
|-------|------|
| `ggml-sys` | CMake + bindgen FFI to `vendor/ggml` ([ggml](https://github.com/ggml-org/ggml) Git submodule) |
| `ggml` | Thin wrappers + `sys` re-export |
| `qwen3-tts` | Pure Rust `rlib` for GGUF loading, speaker encoding, and synthesis |
| `qwen3-tts-cli` | Command-line interface for generating `speaker.bin` and WAV output |

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

Python helper scripts under `scripts/` are managed with `uv` from the workspace root:

```bash
uv sync
uv run export-voice-clone-prompt --help
uv run export-speaker-bin --help
```

Optional Hugging Face helpers:

```bash
cargo build -p qwen3-tts --features hf
```

CLI builds the same engine as the library and supports the same backend features:

```bash
cargo build -p qwen3-tts-cli
cargo build -p qwen3-tts-cli --features metal
cargo build -p qwen3-tts-cli --features vulkan
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

## Reference Audio

`SynthesizeRequest.reference_wav_bytes` now drives a built-in speaker-conditioning path for tone transfer.

- Input format is WAV only in this first pass.
- No extra speaker-model file is required; the library derives a deterministic speaker embedding from reference audio at runtime.
- If `reference_wav_bytes` is absent, synthesis keeps the previous zero-speaker fallback.

For better alignment with upstream `QwenLM/Qwen3-TTS`, stage 1 also supports importing a Python-exported voice-clone prompt and consuming only its `ref_spk_embedding` on the native side.

### Stage 1: Python prompt export, native speaker consumption

Use the repository's `uv`-managed Python environment to export a prompt JSON from `create_voice_clone_prompt(...)`:

```bash
uv sync

uv run export-voice-clone-prompt \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio testdata/hello.wav \
  --ref-text "hello" \
  --out target/hello.voice-clone-prompt.json
```

The legacy script path still works too:

```bash
uv run python scripts/export_voice_clone_prompt.py --help
uv run python scripts/export_speaker_bin.py --help
```

If you only need the upstream speaker embedding as a raw `speaker.bin`, export it directly:

```bash
uv run export-speaker-bin \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio testdata/hello.wav \
  --ref-text "hello" \
  --out target/hello.python.speaker.bin
```

Then consume that prompt from `qwen3-tts-cli`. In stage 1, the native engine reads the JSON and uses only `ref_spk_embedding`; `ref_code` and `ref_text` are preserved for future work but are not yet injected into the transformer:

```bash
cargo run -p qwen3-tts-cli -- synthesize \
  --model-dir models/volko76-q4k-q8 \
  --text "hello" \
  --voice-clone-prompt target/hello.voice-clone-prompt.json \
  --out target/hello-from-prompt.wav

cargo run -p qwen3-tts-cli -- speaker-bin \
  --model-dir models/volko76-q4k-q8 \
  --voice-clone-prompt target/hello.voice-clone-prompt.json \
  --out target/hello.from-prompt.speaker.bin
```

The CLI can also materialize that embedding as a standalone `speaker.bin` so it can be cached and reused:

```bash
cargo run -p qwen3-tts-cli -- speaker-bin \
  --model-dir models/volko76-q4k-q8 \
  --wav testdata/hello.wav \
  --out target/hello.speaker.bin

cargo run -p qwen3-tts-cli -- synthesize \
  --model-dir models/volko76-q4k-q8 \
  --text "hello" \
  --speaker-bin target/hello.speaker.bin \
  --out target/hello-from-speaker-bin.wav
```

### Stage 2 Goal

The next milestone is to consume the full upstream voice-clone prompt on the native side:

- `ref_spk_embedding`
- `ref_code`
- `ref_text`
- `icl_mode` / `x_vector_only_mode`

That will let `qwen3-tts-native` move from speaker-only conditioning toward the upstream Base model's full voice-clone behavior instead of the current stage-1 compatibility path.

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
