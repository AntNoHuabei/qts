---
license: apache-2.0
language:
- zh
- en
- ja
- ko
- de
- fr
- ru
- pt
- es
- it
base_model:
- Qwen/Qwen3-TTS-12Hz-0.6B-Base
pipeline_tag: text-to-speech
quantized_by: dsh0416
tags:
- audio
- tts
- voice-clone
---

# Qwen3-TTS-12Hz-0.6B-Base-QTS

`Qwen3-TTS-12Hz-0.6B-Base-QTS` is a distribution repository for model artifacts
produced by [`yet-another-ai/qts`](https://github.com/yet-another-ai/qts).

This Hugging Face repository is intended to contain stable, downloadable runtime
artifacts only:

- one shared `qwen3-tts-vocoder.onnx`
- one or more GGUF variants such as `qwen3-tts-0.6b-f16.gguf`
- optional additional GGUF variants such as `qwen3-tts-0.6b-q8_0.gguf`

It is not the source-of-truth repository for code, export logic, or developer
documentation. Those live in [`yet-another-ai/qts`](https://github.com/yet-another-ai/qts).

## Relationship To `qts`

- GitHub [`yet-another-ai/qts`](https://github.com/yet-another-ai/qts): source code, export scripts, tests, and documentation
- Hugging Face [`dsh0416/Qwen3-TTS-12Hz-0.6B-Base-QTS`](https://huggingface.co/dsh0416/Qwen3-TTS-12Hz-0.6B-Base-QTS): exported runtime artifacts

Recommended maintenance flow:

1. Change behavior in the GitHub repository first.
2. Export artifacts from a known Git commit.
3. Publish only the built model files to this Hugging Face repository, preferably from the tagged GitHub Actions release workflow in `yet-another-ai/qts`.
4. Keep this model card aligned with the GitHub docs, but do not treat this repository as a second source repository.

## Included Files

Expected root layout:

```text
{{ROOT_LAYOUT}}
README.md
SHA256SUMS
```

Notes:

- `qwen3-tts-vocoder.onnx` is shared across all GGUF variants in this repository.
- The Rust runtime in `qts` expects the GGUF and vocoder files to live in the same directory by default.
- Not every release must ship every quantization variant.
- For the current artifact set, `q8_0` is the recommended default download and `f16` is the reference-quality export.

## Current Quantization Support

At the moment, the `qts` exporter supports:

{{QUANTIZATION_LIST}}

Other quantization types may appear in future releases once the export and
validation pipeline is ready.

## Usage With `qts`

See the source repository for current usage and export documentation:

- GitHub: [`yet-another-ai/qts`](https://github.com/yet-another-ai/qts)
- Models guide: [`docs/models.md`](https://github.com/yet-another-ai/qts/blob/main/docs/models.md)

Typical local layout:

```text
models/
  qwen3-tts-0.6b-f16.gguf
  qwen3-tts-vocoder.onnx
```

Example CLI usage:

```bash
cargo run -p qwen3-tts-cli -- synthesize \
  --model-dir /path/to/models \
  --text "hello" \
  --out target/hello.wav
```

## Provenance

Current source repository snapshot:

- GitHub commit: `{{SOURCE_COMMIT}}`

Current artifact checksums:

{{CHECKSUM_LIST}}

For future releases, it is recommended to record:

- source GitHub commit SHA from `yet-another-ai/qts`
- exported file list
- SHA256 checksums
- any release-specific notes such as added or removed quantization variants

## Base Model

Base upstream model:

- [`Qwen/Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
