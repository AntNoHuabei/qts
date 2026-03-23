# Model artifacts

This repository does **not** ship weights. Download or export the model artifacts separately and point the API at local paths.

## Suggested layout

```
models/
  qwen3-tts-0.6b-f16.gguf
  qwen3-tts-vocoder.onnx
```

Names may differ if you pass explicit [`ModelPaths`](../crates/qwen3-tts/src/model/paths.rs).

The Rust runtime requires the self-contained ONNX vocoder artifact
`qwen3-tts-vocoder.onnx` alongside the main GGUF checkpoint.

## Where to get artifacts

Export them with the repository's Python helper:

```bash
uv sync
uv run export-model-artifacts \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --out-dir models/qwen3-tts-f16-onnx \
  --main-type f16
```

To export multiple GGUF variants into the same directory while reusing a single
unquantized vocoder ONNX, repeat `--main-type` or pass a comma-separated list:

```bash
uv run export-model-artifacts \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --out-dir models/qwen3-tts-bundle \
  --main-type f16 \
  --main-type q8_0
```

Currently supported `--main-type` values are `f16` and `q8_0`.
The exporter does not emit `speaker_encoder` weights because the native runtime
builds its own reference-audio encoder at runtime instead of loading those
tensors from GGUF.

Legacy/community GGUF conversions (verify checksums before trusting):

- [HaujetZhao/Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) — community GGUF builds and related notes.

Official base model (PyTorch / safetensors, for reference — convert elsewhere if needed):

- [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
