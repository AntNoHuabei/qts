# Qwen TTS Native

Rust workspace for on-device **Qwen3 TTS** using [ggml-org/ggml](https://github.com/ggml-org/ggml) for the main transformer and **ONNX Runtime** for the exported vocoder. The project references [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) for architecture and tensor naming, but does **not** link against it.

## Crates

| Crate | Role |
|-------|------|
| `ggml-sys` | CMake + bindgen FFI to `vendor/ggml` ([ggml](https://github.com/ggml-org/ggml) Git submodule) |
| `ggml` | Thin wrappers + `sys` re-export |
| `qwen3-tts` | Pure Rust `rlib` for GGUF loading, speaker encoding, and synthesis |
| `qwen3-tts-cli` | Command-line interface for synthesis, profiling, and optional `speaker.bin` extraction from voice-clone prompts |

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
uv run export-model-artifacts --help
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

For better alignment with upstream `QwenLM/Qwen3-TTS`, the native path also supports a stage-2 protobuf voice-clone prompt that carries the full upstream prompt semantics.

### Stage 2: Protobuf prompt export and native consumption

Use the repository's `uv`-managed Python environment to export a single protobuf prompt from `create_voice_clone_prompt(...)`:

```bash
uv sync

uv run export-voice-clone-prompt \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio testdata/hello.wav \
  --ref-text "hello" \
  --out target/hello.voice-clone-prompt.pb
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

Then consume that prompt from `qwen3-tts-cli`. In stage 2, the native engine reads the protobuf and uses:

- `ref_spk_embedding`
- `ref_code`
- `ref_text`
- `icl_mode` / `x_vector_only_mode`

```bash
cargo run -p qwen3-tts-cli -- synthesize \
  --model-dir models/volko76-q4k-q8 \
  --text "hello" \
  --voice-clone-prompt target/hello.voice-clone-prompt.pb \
  --out target/hello-from-prompt.wav

cargo run -p qwen3-tts-cli -- speaker-bin \
  --model-dir models/volko76-q4k-q8 \
  --voice-clone-prompt target/hello.voice-clone-prompt.pb \
  --out target/hello.from-prompt.speaker.bin
```

You can reuse a `speaker.bin` from `export-speaker-bin` (above) or from `speaker-bin` on a prompt:

```bash
cargo run -p qwen3-tts-cli -- synthesize \
  --model-dir models/volko76-q4k-q8 \
  --text "hello" \
  --speaker-bin target/hello.python.speaker.bin \
  --out target/hello-from-speaker-bin.wav
```

## Interactive TUI

For interactive latency demos, the CLI also has a `tui` mode that loads the model once, lets you type successive utterances, and streams audio directly to the default output device through `cpal`.

```bash
cargo run -p qwen3-tts-cli -- tui \
  --model-dir models/volko76-q4k-q8 \
  --speaker-bin target/hello.python.speaker.bin \
  --language en \
  --chunk-size 4
```

Notes:

- Press `Enter` to synthesize the current line.
- Press `F2` to cycle the synthesis language between English, Chinese, and Japanese.
- Press `Esc`, `Ctrl-C`, or type `:q` to quit.
- The TUI header shows both the transformer backend (`ggml`) and the active vocoder EP (`ORT/CPU` or `ORT/CoreML`).
- `--backend` and `--vocoder-ep` let you choose the transformer backend and vocoder EP directly from the command line.
- `--backend-fallback` and `--vocoder-ep-fallback` accept comma-separated fallback chains used when the corresponding selector is `auto`.
- `--language en|zh|ja` is the friendly startup flag; `--language-id` still works if you want to pass a raw codec language id.
- `--chunk-size` controls how many codec frames are vocoded per playback chunk. Smaller values reduce startup latency, while larger values reduce scheduling overhead.
- `--reference-wav`, `--speaker-bin`, and `--voice-clone-prompt` work the same way as in `synthesize`, but are loaded once up front and then reused for each prompt.

On Apple platforms, the ONNX vocoder can use CoreML:

```bash
cargo run -p qwen3-tts-cli -- tui \
  --model-dir models/qwen3-tts-f16-onnx \
  --backend auto \
  --backend-fallback metal,vulkan,cpu \
  --vocoder-ep coreml \
  --chunk-size 4
```

Defaults:

- Apple transformer `auto`: `metal,vulkan,cpu`
- Other transformer `auto`: `vulkan,cpu`
- Apple vocoder `auto`: `coreml,cpu`
- Other vocoder `auto`: `cpu`

## Tests

Fast tests run in CI; model-backed tests are opt-in: [docs/testing.md](docs/testing.md).

Criterion benchmarks and synthesis profiling are driven through the `xtask` Cargo alias (see [`.cargo/config.toml`](.cargo/config.toml)):

```bash
cargo xtask bench cpu
cargo xtask bench metal
cargo xtask bench vulkan
```

Stage timings (tokenizer, prefill build, codec rollout / transformer, vocoder, etc.) for a real synthesis pass:

```bash
cargo xtask profile cpu --model-dir models/volko76-q4k-q8 --text "hello" --frames 64 --runs 3
cargo xtask profile metal --model-dir models/volko76-q4k-q8 --text "hello" --frames 64
cargo xtask profile vulkan --model-dir models/volko76-q4k-q8 --text "hello" --frames 64
```

`cargo xtask profile` sets **`QWEN3_TTS_BACKEND`** for the child to match the first token (`cpu` / `metal` / `vulkan`), so **Cargo features and the actual GGML primary backend stay aligned** (including Vulkan on macOS when you choose the `vulkan` profile).

For `cargo run -p qwen3-tts-cli` directly, set the backend explicitly, for example:

```bash
QWEN3_TTS_BACKEND=vulkan cargo run -p qwen3-tts-cli --features vulkan -- profile --text "hello" --model-dir models/volko76-q4k-q8 --frames 64
```

This runs `qwen3-tts-cli profile`, which prints per-stage milliseconds and percentage of total wall time. Use `--out run1.wav` to keep audio from the first run while profiling.

**Transformer backend selection:** use `--backend` / `--backend-fallback` on the CLI, or `QWEN3_TTS_BACKEND` / `QWEN3_TTS_BACKEND_FALLBACK` if you prefer environment variables. The default `auto` chain is `metal,vulkan,cpu` on Apple and `vulkan,cpu` elsewhere.

GGML picks GPU devices by **registry** name plus index: use **`QWEN3_TTS_GPU_DEVICE`** (default `0`) to choose among multiple Vulkan or Metal adapters (`Vulkan0` / `MTL0`, …).

**Vocoder EP selection:** use `--vocoder-ep` / `--vocoder-ep-fallback` on the CLI, or `QWEN3_TTS_VOCODER_EP` / `QWEN3_TTS_VOCODER_EP_FALLBACK` from the environment. The default `auto` chain is `coreml,cpu` on Apple and `cpu` elsewhere.

## Godot / gdext

The `qwen3-tts` crate is a normal Rust library (`rlib`). A future Godot project can depend on it directly from a `gdext` crate without a separate `cdylib` ABI layer.
