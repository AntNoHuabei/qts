//! Layer-B style checks: requires real GGUF files on disk.
//!
//! ```text
//! export QWEN3_TTS_MODEL_DIR=/path/to/models   # contains qwen3-tts-0.6b-f16.gguf + qwen3-tts-tokenizer-f16.gguf
//! cargo test -p qwen3-tts integration_ -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use qwen3_tts::{Qwen3TtsEngine, SynthesizeRequest};

fn require_model_dir() -> PathBuf {
    std::env::var("QWEN3_TTS_MODEL_DIR")
        .map(PathBuf::from)
        .expect("QWEN3_TTS_MODEL_DIR must be set when running ignored integration tests")
}

#[test]
#[ignore = "set QWEN3_TTS_MODEL_DIR to run"]
fn integration_loads_models() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load models");
    assert!(engine.model_paths().main_exists());
    assert!(engine.model_paths().vocoder_exists());
    assert!(!engine.encode_for_tts("hello").is_empty());
}

#[test]
#[ignore = "set QWEN3_TTS_MODEL_DIR to run"]
fn integration_synthesize_direct_path_audio() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load");
    let req = SynthesizeRequest {
        text: "hello".into(),
        max_audio_frames: 4,
        ..Default::default()
    };
    let result = engine.synthesize(&req).expect("synthesize audio");
    assert_eq!(result.sample_rate_hz, 24_000);
    assert!(result.generated_frames > 0);
    assert!(!result.pcm_f32.is_empty());
}
