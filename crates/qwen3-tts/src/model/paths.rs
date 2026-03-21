use std::path::{Path, PathBuf};

/// Paths to the two GGUF artifacts (same layout as [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) `models/`).
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// Main TTS checkpoint (e.g. `qwen3-tts-0.6b-f16.gguf`).
    pub main_gguf: PathBuf,
    /// Vocoder / audio tokenizer weights (e.g. `qwen3-tts-tokenizer-f16.gguf`).
    pub vocoder_gguf: PathBuf,
}

impl ModelPaths {
    pub fn new(main_gguf: PathBuf, vocoder_gguf: PathBuf) -> Self {
        Self {
            main_gguf,
            vocoder_gguf,
        }
    }

    /// Resolve under a single directory using the conventional filenames.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self {
            main_gguf: choose_model_file(
                dir,
                &[
                    "qwen3-tts-0.6b-f16.gguf",
                    "qwen3-tts-0.6b-q8_0.gguf",
                    "qwen3-tts-0.6b-q6_k.gguf",
                    "qwen3-tts-0.6b-q5_k.gguf",
                    "qwen3-tts-0.6b-q4_k.gguf",
                ],
            ),
            vocoder_gguf: choose_model_file(
                dir,
                &[
                    "qwen3-tts-tokenizer-f16.gguf",
                    "qwen3-tts-tokenizer-q8_0.gguf",
                    "qwen3-tts-tokenizer-q6_k.gguf",
                    "qwen3-tts-tokenizer-q5_k.gguf",
                    "qwen3-tts-tokenizer-q4_k.gguf",
                ],
            ),
        }
    }

    pub fn main_exists(&self) -> bool {
        self.main_gguf.is_file()
    }

    pub fn vocoder_exists(&self) -> bool {
        self.vocoder_gguf.is_file()
    }
}

fn choose_model_file(dir: &Path, candidates: &[&str]) -> PathBuf {
    for candidate in candidates {
        let path = dir.join(candidate);
        if path.is_file() {
            return path;
        }
    }

    dir.join(candidates[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_model_dir_names() {
        let p = ModelPaths::from_model_dir("/models");
        assert!(p.main_gguf.ends_with("qwen3-tts-0.6b-f16.gguf"));
        assert!(p.vocoder_gguf.ends_with("qwen3-tts-tokenizer-f16.gguf"));
    }

    #[test]
    fn from_model_dir_prefers_existing_quantized_files() {
        let dir = std::env::temp_dir().join("qwen_tts_native_paths_quantized");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("qwen3-tts-0.6b-q4_k.gguf"), b"").unwrap();
        std::fs::write(dir.join("qwen3-tts-tokenizer-q8_0.gguf"), b"").unwrap();

        let p = ModelPaths::from_model_dir(&dir);
        assert!(p.main_gguf.ends_with("qwen3-tts-0.6b-q4_k.gguf"));
        assert!(p.vocoder_gguf.ends_with("qwen3-tts-tokenizer-q8_0.gguf"));

        let _ = std::fs::remove_dir_all(dir);
    }
}
