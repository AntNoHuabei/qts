//! Qwen3 TTS (GGUF + GGML) — native inference library.
//!
//! Output sample rate for the published Qwen3-TTS checkpoints.
pub const SAMPLE_RATE_HZ: u32 = 24_000;

mod error;
mod model;
pub mod pipeline;

#[cfg(feature = "hf")]
pub mod hf;

pub use error::Qwen3TtsError;
pub use model::{load_and_validate, GgufFile, ModelPaths};
pub use pipeline::tokenizer::{TextTokenizer, TokenizerConfig};
pub use pipeline::tts_transformer::{
    CodecRollout, PreparedPrefillInputs, PrefillForwardOutputs, SelectedCodecFrame, TtsTransformer,
    TtsTransformerConfig,
};
pub use pipeline::vocoder::{Vocoder, VocoderConfig};

/// User-facing synthesis parameters (stable for future `gdext` bindings).
#[derive(Debug, Clone)]
pub struct SynthesizeRequest {
    pub text: String,
    pub reference_wav_bytes: Option<Vec<u8>>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub max_audio_frames: usize,
    pub thread_count: usize,
    pub repetition_penalty: f32,
    /// Codec language id (e.g. 2050=en, 2055=zh, 2058=ja).
    pub language_id: i32,
}

impl Default for SynthesizeRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            reference_wav_bytes: None,
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            max_audio_frames: 4096,
            thread_count: 4,
            repetition_penalty: 1.05,
            language_id: 2050,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynthesizeResult {
    pub pcm_f32: Vec<f32>,
    pub sample_rate_hz: u32,
    /// Best-effort: frame count not yet exposed by C API; 0 for now.
    pub generated_frames: usize,
}

pub struct Qwen3TtsEngine {
    paths: ModelPaths,
    tokenizer: TextTokenizer,
    transformer: TtsTransformer,
    vocoder: Vocoder,
}

impl Qwen3TtsEngine {
    pub fn load(paths: ModelPaths) -> Result<Self, Qwen3TtsError> {
        load_and_validate(&paths)?;
        let main = GgufFile::open(&paths.main_gguf)?;
        let vocoder_gguf = GgufFile::open(&paths.vocoder_gguf)?;
        let tokenizer = TextTokenizer::load_from_gguf(&main)?;
        let transformer = TtsTransformer::load_from_gguf(&main)?;
        let vocoder = Vocoder::load_from_gguf(&vocoder_gguf)?;

        Ok(Self {
            paths,
            tokenizer,
            transformer,
            vocoder,
        })
    }

    pub fn from_model_dir(dir: impl AsRef<std::path::Path>) -> Result<Self, Qwen3TtsError> {
        Self::load(ModelPaths::from_model_dir(dir))
    }

    pub fn model_paths(&self) -> &ModelPaths {
        &self.paths
    }

    #[must_use]
    pub fn tokenizer(&self) -> &TextTokenizer {
        &self.tokenizer
    }

    #[must_use]
    pub fn transformer(&self) -> &TtsTransformer {
        &self.transformer
    }

    #[must_use]
    pub fn vocoder(&self) -> &Vocoder {
        &self.vocoder
    }

    #[must_use]
    pub fn encode_for_tts(&self, text: &str) -> Vec<i32> {
        self.tokenizer.encode_for_tts(text)
    }

    pub fn synthesize(&self, req: &SynthesizeRequest) -> Result<SynthesizeResult, Qwen3TtsError> {
        let tokens = self.tokenizer.encode_for_tts(&req.text);
        let zero_speaker = vec![0.0f32; self.transformer.config().hidden_size as usize];
        let prepared_inputs = self.transformer.build_prefill_inputs(
            &tokens,
            Some(&zero_speaker),
            req.language_id,
            req.thread_count,
        )?;
        let codec_rollout = self.transformer.rollout_codec_frames_kv(
            &prepared_inputs.prefill_embd,
            &prepared_inputs.trailing_text_hidden,
            &prepared_inputs.tts_pad_embed,
            req.thread_count,
            req.max_audio_frames,
            req.repetition_penalty,
            req.temperature,
            req.top_k,
            req.top_p,
        )?;
        let generated_frames = codec_rollout.frames.len();
        let flattened_codes = codec_rollout
            .frames
            .iter()
            .flat_map(|frame| frame.codebook_tokens.iter().copied())
            .collect::<Vec<_>>();
        let pcm_f32 = self
            .vocoder
            .decode(&flattened_codes, generated_frames, req.thread_count)?;
        let _ = req.reference_wav_bytes.as_ref();

        Ok(SynthesizeResult {
            pcm_f32,
            sample_rate_hz: self.vocoder.config().sample_rate as u32,
            generated_frames,
        })
    }
}

/// Placeholder for future frame/chunk streaming (e.g. Godot audio stream).
pub trait StreamingSynthesis {
    fn next_pcm_chunk(&mut self) -> Option<Result<Vec<f32>, Qwen3TtsError>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults() {
        let r = SynthesizeRequest::default();
        assert_eq!(r.temperature, 0.9);
        assert_eq!(r.top_k, 50);
        assert_eq!(r.language_id, 2050);
    }
}
