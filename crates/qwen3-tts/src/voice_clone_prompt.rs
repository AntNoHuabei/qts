use serde::{Deserialize, Serialize};

use crate::Qwen3TtsError;

pub const VOICE_CLONE_PROMPT_V1_SCHEMA: &str = "qwen3_tts.voice_clone_prompt.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptTensorI32 {
    pub shape: Vec<usize>,
    pub values: Vec<i32>,
}

impl PromptTensorI32 {
    fn validate(&self, field_name: &str) -> Result<(), Qwen3TtsError> {
        if self.shape.is_empty() {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "{field_name}.shape must not be empty"
            )));
        }
        let element_count = self
            .shape
            .iter()
            .copied()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
            .ok_or_else(|| {
                Qwen3TtsError::InvalidInput(format!("{field_name}.shape overflows element count"))
            })?;
        if element_count != self.values.len() {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "{field_name}.values length {} does not match shape product {}",
                self.values.len(),
                element_count
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceClonePromptV1 {
    pub schema: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub speaker_encoder_sample_rate_hz: Option<u32>,
    #[serde(default)]
    pub x_vector_only_mode: bool,
    #[serde(default)]
    pub icl_mode: bool,
    #[serde(default)]
    pub ref_text: Option<String>,
    #[serde(default)]
    pub ref_code: Option<PromptTensorI32>,
    pub ref_spk_embedding: Vec<f32>,
}

impl VoiceClonePromptV1 {
    pub fn from_json_bytes(json_bytes: &[u8]) -> Result<Self, Qwen3TtsError> {
        let prompt = serde_json::from_slice::<Self>(json_bytes).map_err(|err| {
            Qwen3TtsError::InvalidInput(format!("invalid voice clone prompt json: {err}"))
        })?;
        prompt.validate()?;
        Ok(prompt)
    }

    pub fn to_json_vec_pretty(&self) -> Result<Vec<u8>, Qwen3TtsError> {
        self.validate()?;
        serde_json::to_vec_pretty(self).map_err(|err| {
            Qwen3TtsError::InvalidInput(format!("failed to encode voice clone prompt json: {err}"))
        })
    }

    pub fn validate(&self) -> Result<(), Qwen3TtsError> {
        if self.schema != VOICE_CLONE_PROMPT_V1_SCHEMA {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "unsupported voice clone prompt schema: {}",
                self.schema
            )));
        }
        if self.ref_spk_embedding.is_empty() {
            return Err(Qwen3TtsError::InvalidInput(
                "voice clone prompt ref_spk_embedding must not be empty".into(),
            ));
        }
        if let Some(ref_code) = &self.ref_code {
            ref_code.validate("voice clone prompt ref_code")?;
        }
        Ok(())
    }

    #[must_use]
    pub fn speaker_embedding(&self) -> &[f32] {
        &self.ref_spk_embedding
    }
}

#[cfg(test)]
mod tests {
    use super::{PromptTensorI32, VoiceClonePromptV1, VOICE_CLONE_PROMPT_V1_SCHEMA};

    #[test]
    fn parses_prompt_json() {
        let prompt = VoiceClonePromptV1 {
            schema: VOICE_CLONE_PROMPT_V1_SCHEMA.into(),
            source: Some("unit-test".into()),
            model_id: Some("Qwen/Qwen3-TTS-12Hz-0.6B-Base".into()),
            speaker_encoder_sample_rate_hz: Some(24_000),
            x_vector_only_mode: false,
            icl_mode: true,
            ref_text: Some("hello".into()),
            ref_code: Some(PromptTensorI32 {
                shape: vec![2, 3],
                values: vec![1, 2, 3, 4, 5, 6],
            }),
            ref_spk_embedding: vec![0.1, 0.2, 0.3, 0.4],
        };
        let json = prompt.to_json_vec_pretty().unwrap();
        let parsed = VoiceClonePromptV1::from_json_bytes(&json).unwrap();
        assert_eq!(parsed, prompt);
    }

    #[test]
    fn rejects_bad_schema() {
        let prompt = br#"{
            "schema": "wrong.schema",
            "ref_spk_embedding": [0.1, 0.2]
        }"#;
        let err = VoiceClonePromptV1::from_json_bytes(prompt).unwrap_err();
        assert!(err
            .to_string()
            .contains("unsupported voice clone prompt schema"));
    }
}
