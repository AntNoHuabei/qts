//! Layer-B style checks: requires real model artifacts on disk.
//!
//! ```text
//! export QWEN3_TTS_MODEL_DIR=/path/to/models   # contains qwen3-tts-0.6b-f16.gguf + qwen3-tts-vocoder.onnx
//! cargo test -p qwen3-tts integration_ -- --ignored --nocapture
//! ```

use std::io::Cursor;
use std::path::PathBuf;

use hound::{SampleFormat, WavSpec, WavWriter};
use qwen3_tts::{
    PrefillConditioning, Qwen3TtsEngine, SynthesizeRequest, TensorF32, TensorI32,
    VoiceClonePromptV2, VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
};

fn require_model_dir() -> PathBuf {
    std::env::var("QWEN3_TTS_MODEL_DIR")
        .map(PathBuf::from)
        .expect("QWEN3_TTS_MODEL_DIR must be set when running ignored integration tests")
}

fn synthetic_voice_like_wav(sample_rate_hz: u32, seconds: f32) -> Vec<u8> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).expect("wav writer");
    let total_samples = (sample_rate_hz as f32 * seconds) as usize;
    for idx in 0..total_samples {
        let t = idx as f32 / sample_rate_hz as f32;
        let envelope = ((idx as f32 / total_samples as f32) * std::f32::consts::PI)
            .sin()
            .max(0.1);
        let sample = ((2.0 * std::f32::consts::PI * 140.0 * t).sin() * 0.55
            + (2.0 * std::f32::consts::PI * 280.0 * t).sin() * 0.25
            + (2.0 * std::f32::consts::PI * 560.0 * t).sin() * 0.15)
            * envelope;
        writer
            .write_sample((sample * i16::MAX as f32) as i16)
            .expect("write wav sample");
    }
    writer.finalize().expect("finalize wav");
    cursor.into_inner()
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

#[test]
#[ignore = "set QWEN3_TTS_MODEL_DIR to run"]
fn integration_reference_wav_changes_conditioning() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load");
    let reference_wav = synthetic_voice_like_wav(16_000, 1.25);
    let speaker = engine
        .encode_reference_speaker(&reference_wav)
        .expect("encode reference speaker");
    let hidden_size = engine.transformer().config().hidden_size as usize;
    assert_eq!(speaker.len(), hidden_size);
    assert!(speaker.iter().any(|value| value.abs() > 1e-4));
    let tokens = engine.encode_for_tts("hello");
    let zero_speaker = vec![0.0f32; hidden_size];
    let baseline_prefill = engine
        .transformer()
        .build_prefill_inputs(
            PrefillConditioning {
                text_tokens: &tokens,
                speaker_embd: Some(&zero_speaker),
                ref_codebook_0: &[],
                language_id: 2050,
            },
            1,
        )
        .expect("baseline prefill");
    let conditioned_prefill = engine
        .transformer()
        .build_prefill_inputs(
            PrefillConditioning {
                text_tokens: &tokens,
                speaker_embd: Some(&speaker),
                ref_codebook_0: &[],
                language_id: 2050,
            },
            1,
        )
        .expect("conditioned prefill");
    assert_ne!(
        baseline_prefill.prefill_embd,
        conditioned_prefill.prefill_embd
    );

    let baseline = SynthesizeRequest {
        text: "hello".into(),
        max_audio_frames: 16,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        ..Default::default()
    };
    let conditioned = SynthesizeRequest {
        reference_wav_bytes: Some(reference_wav),
        ..baseline.clone()
    };

    let baseline_audio = engine.synthesize(&baseline).expect("baseline synthesize");
    let conditioned_audio = engine
        .synthesize(&conditioned)
        .expect("conditioned synthesize");
    assert_eq!(
        baseline_audio.sample_rate_hz,
        conditioned_audio.sample_rate_hz
    );
    assert_eq!(
        baseline_audio.generated_frames,
        conditioned_audio.generated_frames
    );
    assert!(!baseline_audio.pcm_f32.is_empty());
    assert!(!conditioned_audio.pcm_f32.is_empty());
}

#[test]
#[ignore = "set QWEN3_TTS_MODEL_DIR to run"]
fn integration_voice_clone_prompt_xvector_mode() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load");
    let reference_wav = synthetic_voice_like_wav(16_000, 1.25);
    let speaker = engine
        .encode_reference_speaker(&reference_wav)
        .expect("encode reference speaker");
    let prompt = VoiceClonePromptV2 {
        schema_version: VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
        source: "integration-test".into(),
        model_id: "local".into(),
        speaker_encoder_sample_rate_hz: 16_000,
        x_vector_only_mode: true,
        icl_mode: false,
        ref_text: String::new(),
        ref_code: None,
        ref_spk_embedding: Some(TensorF32 {
            shape: vec![speaker.len() as u32],
            values: speaker,
        }),
    };
    let req = SynthesizeRequest {
        text: "hello".into(),
        max_audio_frames: 4,
        ..Default::default()
    };
    let result = engine
        .synthesize_with_voice_clone_prompt(&req, &prompt)
        .expect("synthesize xvector prompt");
    assert_eq!(result.sample_rate_hz, 24_000);
    assert!(result.generated_frames > 0);
    assert!(!result.pcm_f32.is_empty());
}

#[test]
#[ignore = "set QWEN3_TTS_MODEL_DIR to run"]
fn integration_voice_clone_prompt_icl_mode() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load");
    let reference_wav = synthetic_voice_like_wav(16_000, 1.25);
    let speaker = engine
        .encode_reference_speaker(&reference_wav)
        .expect("encode reference speaker");
    let n_codebooks = engine.transformer().config().n_codebooks as usize;
    let prompt = VoiceClonePromptV2 {
        schema_version: VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
        source: "integration-test".into(),
        model_id: "local".into(),
        speaker_encoder_sample_rate_hz: 16_000,
        x_vector_only_mode: false,
        icl_mode: true,
        ref_text: "hello".into(),
        ref_code: Some(TensorI32 {
            shape: vec![2, n_codebooks as u32],
            values: vec![0; 2 * n_codebooks],
        }),
        ref_spk_embedding: Some(TensorF32 {
            shape: vec![speaker.len() as u32],
            values: speaker,
        }),
    };
    let req = SynthesizeRequest {
        text: "world".into(),
        max_audio_frames: 4,
        ..Default::default()
    };
    let result = engine
        .synthesize_with_voice_clone_prompt(&req, &prompt)
        .expect("synthesize icl prompt");
    assert_eq!(result.sample_rate_hz, 24_000);
    assert!(result.generated_frames > 0);
    assert!(!result.pcm_f32.is_empty());
}
