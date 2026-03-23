# Model artifacts

This repository does **not** ship weights. Download or export the model artifacts separately and point the API at local paths.

## Suggested layout

```
models/
  qwen3-tts-0.6b-f16.gguf
  qwen3-tts-vocoder.onnx
  qwen3-tts-vocoder.onnx.data
```

Names may differ if you pass explicit [`ModelPaths`](../crates/qwen3-tts/src/model/paths.rs).

The Rust runtime requires the ONNX vocoder artifact `qwen3-tts-vocoder.onnx` alongside the main GGUF checkpoint.

## Where to get artifacts

Export them with the repository's Python helper:

```bash
uv sync
uv run export-model-artifacts \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --out-dir models/qwen3-tts-f16-onnx \
  --main-type f16
```

Legacy/community GGUF conversions (verify checksums before trusting):

- [HaujetZhao/Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) — community GGUF builds and related notes.

Official base model (PyTorch / safetensors, for reference — convert elsewhere if needed):

- [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)

## Optional download in Rust

Enable feature `hf` on `qwen3-tts`, then use `qwen3_tts::hf::download_hf_file` with a repo id and file path (see `hf-hub` cache semantics).

Example (pseudo-repo — replace with the GGUF repo/file you actually use):

```rust
qwen3_tts::hf::download_hf_file(
    "namespace/Qwen3-TTS-GGUF",
    "qwen3-tts-0.6b-f16.gguf",
    std::path::Path::new("models"),
)?;
```
