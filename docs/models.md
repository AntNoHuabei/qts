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

Recommended public artifact repository:

- Hugging Face: [`dsh0416/Qwen3-TTS-12Hz-0.6B-Base-QTS`](https://huggingface.co/dsh0416/Qwen3-TTS-12Hz-0.6B-Base-QTS)

Repository roles:

- GitHub [`yet-another-ai/qts`](https://github.com/yet-another-ai/qts) is the source of truth for source code, export logic, and documentation.
- The Hugging Face repository above is the distribution target for exported GGUF and ONNX artifacts.

The recommended maintenance flow is:

1. Change exporter behavior and documentation in the GitHub repository.
2. Export stable artifacts from a known Git commit.
3. Upload only the built model files to the Hugging Face repository root.
4. Keep one shared `qwen3-tts-vocoder.onnx` and multiple GGUF variants side by side so the default Rust model-path resolution works unchanged.

If you want a copy-ready model card for the Hugging Face repository, use
[`docs/huggingface-model-card.md`](huggingface-model-card.md) from this
repository as the starting point.

To automate Hugging Face release packaging from an artifact directory, run:

```bash
cargo xtask hf-release --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
```

By default this prepares `target/hf-qts-release/` with:

- freshly exported `f16` and `q8_0` artifacts produced via `uv run export-model-artifacts`
- copied `qwen3-tts-0.6b-*.gguf` files
- copied `qwen3-tts-vocoder.onnx`
- generated `README.md`
- generated `SHA256SUMS`
- generated `.gitattributes` that routes `*.gguf` and `*.onnx` through Hugging Face Xet

If you already have the Hugging Face repository cloned locally, pass
`--hf-repo-dir /path/to/cloned-hf-repo`. By default, `xtask` will then reuse
that repository root as both the export destination and the packaged release
directory, so the managed files are generated in place instead of being copied
through an extra staging directory. You can still override `--artifacts-dir` or
`--out-dir` if you explicitly want a separate staging layout.

For official releases, this repository now treats GitHub Actions as the source
of truth for publication:

- `.github/workflows/hf-release.yml` runs on pushed `v*` tags, exports the model with `uv`, packages it with `cargo xtask hf-release`, syncs the managed files into the Hugging Face checkout, and pushes to `dsh0416/Qwen3-TTS-12Hz-0.6B-Base-QTS`
- `.github/workflows/model-integration.yml` is the manual preview path that stages the same release bundle and uploads it as a GitHub Actions artifact without publishing
- repository secret `HF_TOKEN` must be set with write access to the Hugging Face model repository before tagged releases can publish successfully

In other words, local `cargo xtask hf-release ...` is now the preview/staging
path, while tagged GitHub Actions runs are the official publishing path.

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
