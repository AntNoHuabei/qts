# GGUF models

This repository does **not** ship weights. Download GGUF files separately and point the API at local paths.

## Suggested layout

```
models/
  qwen3-tts-0.6b-f16.gguf
  qwen3-tts-tokenizer-f16.gguf
```

Names may differ if you pass explicit [`ModelPaths`](../crates/qwen3-tts/src/model/paths.rs).

## Where to get GGUF

Community conversions (verify checksums before trusting):

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
