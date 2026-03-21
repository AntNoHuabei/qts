//! Optional helpers to populate a local directory from Hugging Face (behind feature `hf`).
//!
//! URLs and repo IDs live in `docs/models.md` at the workspace root.

use std::path::Path;

use hf_hub::api::sync::ApiBuilder;

use crate::Qwen3TtsError;

/// Download a single file from a Hugging Face repo into `dest_dir` and return its path.
pub fn download_hf_file(
    repo: &str,
    filename: &str,
    dest_dir: &Path,
) -> Result<std::path::PathBuf, Qwen3TtsError> {
    let api = ApiBuilder::new().with_progress(true).build()?;
    let p = api.model(repo.to_string()).get(filename)?;
    let name = std::path::Path::new(filename)
        .file_name()
        .ok_or_else(|| Qwen3TtsError::InvalidPath)?;
    std::fs::create_dir_all(dest_dir)?;
    let out = dest_dir.join(name);
    std::fs::copy(&p, &out)?;
    Ok(out)
}
