//! Qwen3 TTS (GGUF + GGML) — native inference library.
//!
//! Output sample rate for the published Qwen3-TTS checkpoints.
pub const SAMPLE_RATE_HZ: u32 = 24_000;

mod error;
mod model;
pub mod pipeline;
mod voice_clone_prompt;

#[cfg(feature = "hf")]
pub mod hf;

pub use error::Qwen3TtsError;
pub use model::{load_and_validate, GgufFile, ModelPaths};
pub use pipeline::speaker_encoder::{SpeakerEncoder, SpeakerEncoderConfig};
pub use pipeline::tokenizer::{TextTokenizer, TokenizerConfig};
pub use pipeline::tts_transformer::{
    CodecRollout, PrefillForwardOutputs, PreparedPrefillInputs, SelectedCodecFrame, TtsTransformer,
    TtsTransformerConfig,
};
pub use pipeline::vocoder::{Vocoder, VocoderConfig};
pub use voice_clone_prompt::{PromptTensorI32, VoiceClonePromptV1, VOICE_CLONE_PROMPT_V1_SCHEMA};

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
    speaker_encoder: SpeakerEncoder,
}

impl Qwen3TtsEngine {
    pub fn load(paths: ModelPaths) -> Result<Self, Qwen3TtsError> {
        load_and_validate(&paths)?;
        let main = GgufFile::open(&paths.main_gguf)?;
        let vocoder_gguf = GgufFile::open(&paths.vocoder_gguf)?;
        let tokenizer = TextTokenizer::load_from_gguf(&main)?;
        let transformer = TtsTransformer::load_from_gguf(&main)?;
        let vocoder = Vocoder::load_from_gguf(&vocoder_gguf)?;
        let speaker_encoder = SpeakerEncoder::new(transformer.config().hidden_size as usize)?;

        Ok(Self {
            paths,
            tokenizer,
            transformer,
            vocoder,
            speaker_encoder,
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
    pub fn speaker_encoder(&self) -> &SpeakerEncoder {
        &self.speaker_encoder
    }

    #[must_use]
    pub fn encode_for_tts(&self, text: &str) -> Vec<i32> {
        self.tokenizer.encode_for_tts(text)
    }

    pub fn encode_reference_speaker(&self, wav_bytes: &[u8]) -> Result<Vec<f32>, Qwen3TtsError> {
        self.speaker_encoder.encode_wav_bytes(wav_bytes)
    }

    #[must_use]
    pub fn speaker_embedding_size(&self) -> usize {
        self.transformer.config().hidden_size as usize
    }

    pub fn encode_reference_speaker_bin(&self, wav_bytes: &[u8]) -> Result<Vec<u8>, Qwen3TtsError> {
        let embedding = self.encode_reference_speaker(wav_bytes)?;
        Ok(embedding
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>())
    }

    pub fn decode_speaker_embedding_bin(
        &self,
        bin_bytes: &[u8],
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        if bin_bytes.len() % std::mem::size_of::<f32>() != 0 {
            return Err(Qwen3TtsError::InvalidInput(
                "speaker.bin must be a raw little-endian f32 array".into(),
            ));
        }

        let embedding = bin_bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().expect("speaker f32 chunk");
                f32::from_le_bytes(bytes)
            })
            .collect::<Vec<_>>();
        if embedding.len() != self.speaker_embedding_size() {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "speaker.bin must contain {} f32 values",
                self.speaker_embedding_size()
            )));
        }
        Ok(embedding)
    }

    pub fn decode_voice_clone_prompt_json(
        &self,
        json_bytes: &[u8],
    ) -> Result<VoiceClonePromptV1, Qwen3TtsError> {
        let prompt = VoiceClonePromptV1::from_json_bytes(json_bytes)?;
        self.validate_speaker_embedding(prompt.speaker_embedding())?;
        Ok(prompt)
    }

    pub fn synthesize_with_speaker_embedding(
        &self,
        req: &SynthesizeRequest,
        speaker_embedding: &[f32],
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.validate_speaker_embedding(speaker_embedding)?;
        self.synthesize_impl(req, Some(speaker_embedding))
    }

    pub fn synthesize_with_voice_clone_prompt(
        &self,
        req: &SynthesizeRequest,
        prompt: &VoiceClonePromptV1,
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.synthesize_with_speaker_embedding(req, prompt.speaker_embedding())
    }

    pub fn synthesize(&self, req: &SynthesizeRequest) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.synthesize_impl(req, None)
    }

    fn synthesize_impl(
        &self,
        req: &SynthesizeRequest,
        speaker_embedding_override: Option<&[f32]>,
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        let tokens = self.tokenizer.encode_for_tts(&req.text);
        let encoded_speaker;
        let zero_speaker;
        let speaker_embedding = if let Some(speaker_embedding) = speaker_embedding_override {
            speaker_embedding
        } else if let Some(wav_bytes) = req.reference_wav_bytes.as_deref() {
            encoded_speaker = self.encode_reference_speaker(wav_bytes)?;
            &encoded_speaker
        } else {
            zero_speaker = vec![0.0f32; self.speaker_embedding_size()];
            &zero_speaker
        };
        let prepared_inputs = self.transformer.build_prefill_inputs(
            &tokens,
            Some(speaker_embedding),
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

        Ok(SynthesizeResult {
            pcm_f32,
            sample_rate_hz: self.vocoder.config().sample_rate as u32,
            generated_frames,
        })
    }

    fn validate_speaker_embedding(&self, speaker_embedding: &[f32]) -> Result<(), Qwen3TtsError> {
        let expected = self.speaker_embedding_size();
        if speaker_embedding.len() != expected {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "speaker embedding must have {expected} elements"
            )));
        }
        Ok(())
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
