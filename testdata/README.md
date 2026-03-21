# Test data

This directory is for **local** fixtures only: everything here except this file is ignored by git (see root `.gitignore`).

- **`minimal.gguf`** (optional): tiny synthetic GGUF for parser experiments — generate with your preferred tooling and keep under ~512 KiB.
- **Golden vectors** (optional): for layer-B numerics, add `reference/*.bin` from an upstream Qwen3 TTS reference build (e.g. [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp)) — not vendored in this repo.

Integration tests use real checkpoints via `QWEN3_TTS_MODEL_DIR` (see `docs/testing.md`).
