use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use qwen3_tts::{Qwen3TtsEngine, SynthesizeRequest};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("speaker-bin") => run_speaker_bin(args.collect()),
        Some("synthesize") => run_synthesize(args.collect()),
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn run_speaker_bin(args: Vec<String>) -> Result<()> {
    let mut model_dir = default_model_dir()?;
    let mut wav_path = None;
    let mut prompt_path = None;
    let mut out_path = None;

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--model-dir" => {
                model_dir = PathBuf::from(value_arg(&args, &mut idx, "--model-dir")?);
            }
            "--wav" => {
                wav_path = Some(PathBuf::from(value_arg(&args, &mut idx, "--wav")?));
            }
            "--voice-clone-prompt" => {
                prompt_path = Some(PathBuf::from(value_arg(
                    &args,
                    &mut idx,
                    "--voice-clone-prompt",
                )?));
            }
            "--out" => {
                out_path = Some(PathBuf::from(value_arg(&args, &mut idx, "--out")?));
            }
            other => bail!("unknown speaker-bin argument: {other}"),
        }
    }

    let out_path = out_path.context("--out is required")?;
    if wav_path.is_some() == prompt_path.is_some() {
        bail!("exactly one of --wav or --voice-clone-prompt is required");
    }
    let engine = load_engine(&model_dir)?;
    let speaker_bin = match (wav_path, prompt_path) {
        (Some(wav_path), None) => {
            let wav_bytes = fs::read(&wav_path)
                .with_context(|| format!("failed to read {}", wav_path.display()))?;
            engine.encode_reference_speaker_bin(&wav_bytes)?
        }
        (None, Some(prompt_path)) => {
            let prompt_bytes = fs::read(&prompt_path)
                .with_context(|| format!("failed to read {}", prompt_path.display()))?;
            let prompt = engine.decode_voice_clone_prompt_json(&prompt_bytes)?;
            prompt
                .speaker_embedding()
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        }
        _ => unreachable!("validated exclusive prompt input"),
    };
    fs::write(&out_path, &speaker_bin)
        .with_context(|| format!("failed to write {}", out_path.display()))?;

    eprintln!(
        "wrote speaker.bin: path={} dim={} bytes={}",
        out_path.display(),
        engine.speaker_embedding_size(),
        speaker_bin.len()
    );
    Ok(())
}

fn run_synthesize(args: Vec<String>) -> Result<()> {
    let mut model_dir = default_model_dir()?;
    let mut text = None;
    let mut out_path = None;
    let mut reference_wav = None;
    let mut speaker_bin = None;
    let mut voice_clone_prompt = None;
    let mut thread_count = 4usize;
    let mut max_audio_frames = 4096usize;
    let mut temperature = 0.9f32;
    let mut top_k = 50i32;
    let mut top_p = 1.0f32;
    let mut repetition_penalty = 1.05f32;
    let mut language_id = 2050i32;

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--model-dir" => {
                model_dir = PathBuf::from(value_arg(&args, &mut idx, "--model-dir")?);
            }
            "--text" => {
                text = Some(value_arg(&args, &mut idx, "--text")?);
            }
            "--out" => {
                out_path = Some(PathBuf::from(value_arg(&args, &mut idx, "--out")?));
            }
            "--reference-wav" => {
                reference_wav = Some(PathBuf::from(value_arg(
                    &args,
                    &mut idx,
                    "--reference-wav",
                )?));
            }
            "--speaker-bin" => {
                speaker_bin = Some(PathBuf::from(value_arg(&args, &mut idx, "--speaker-bin")?));
            }
            "--voice-clone-prompt" => {
                voice_clone_prompt = Some(PathBuf::from(value_arg(
                    &args,
                    &mut idx,
                    "--voice-clone-prompt",
                )?));
            }
            "--threads" => {
                thread_count = parse_value_arg(&args, &mut idx, "--threads")?;
            }
            "--frames" => {
                max_audio_frames = parse_value_arg(&args, &mut idx, "--frames")?;
            }
            "--temperature" => {
                temperature = parse_value_arg(&args, &mut idx, "--temperature")?;
            }
            "--top-k" => {
                top_k = parse_value_arg(&args, &mut idx, "--top-k")?;
            }
            "--top-p" => {
                top_p = parse_value_arg(&args, &mut idx, "--top-p")?;
            }
            "--repetition-penalty" => {
                repetition_penalty = parse_value_arg(&args, &mut idx, "--repetition-penalty")?;
            }
            "--language-id" => {
                language_id = parse_value_arg(&args, &mut idx, "--language-id")?;
            }
            other => bail!("unknown synthesize argument: {other}"),
        }
    }

    let prompt_inputs = usize::from(reference_wav.is_some())
        + usize::from(speaker_bin.is_some())
        + usize::from(voice_clone_prompt.is_some());
    if prompt_inputs > 1 {
        bail!("--reference-wav, --speaker-bin, and --voice-clone-prompt are mutually exclusive");
    }

    let text = text.context("--text is required")?;
    let out_path = out_path.context("--out is required")?;
    let engine = load_engine(&model_dir)?;

    let request = SynthesizeRequest {
        text,
        reference_wav_bytes: match reference_wav.as_ref() {
            Some(path) => {
                Some(fs::read(path).with_context(|| format!("failed to read {}", path.display()))?)
            }
            None => None,
        },
        temperature,
        top_p,
        top_k,
        max_audio_frames,
        thread_count,
        repetition_penalty,
        language_id,
    };

    let result = if let Some(path) = voice_clone_prompt {
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let prompt = engine.decode_voice_clone_prompt_json(&bytes)?;
        engine.synthesize_with_voice_clone_prompt(&request, &prompt)?
    } else if let Some(path) = speaker_bin {
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let embedding = engine.decode_speaker_embedding_bin(&bytes)?;
        engine.synthesize_with_speaker_embedding(&request, &embedding)?
    } else {
        engine.synthesize(&request)?
    };

    write_wav_f32(&out_path, result.sample_rate_hz, &result.pcm_f32)?;
    eprintln!(
        "wrote synthesis: path={} sample_rate={} frames={} samples={}",
        out_path.display(),
        result.sample_rate_hz,
        result.generated_frames,
        result.pcm_f32.len()
    );
    Ok(())
}

fn load_engine(model_dir: &Path) -> Result<Qwen3TtsEngine> {
    Qwen3TtsEngine::from_model_dir(model_dir)
        .with_context(|| format!("failed to load model dir {}", model_dir.display()))
}

fn write_wav_f32(path: &Path, sample_rate_hz: u32, pcm_f32: &[f32]) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("failed to create {}", path.display()))?;
    for sample in pcm_f32.iter().copied() {
        let clamped = sample.clamp(-1.0, 1.0);
        writer
            .write_sample((clamped * i16::MAX as f32) as i16)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("failed to finalize {}", path.display()))?;
    Ok(())
}

fn default_model_dir() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("qwen3-tts-cli manifest has no workspace parent")?;
    Ok(workspace_root.join("models/volko76-q4k-q8"))
}

fn value_arg(args: &[String], idx: &mut usize, flag: &str) -> Result<String> {
    *idx += 1;
    let value = args
        .get(*idx)
        .with_context(|| format!("missing value for {flag}"))?
        .clone();
    *idx += 1;
    Ok(value)
}

fn parse_value_arg<T>(args: &[String], idx: &mut usize, flag: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = value_arg(args, idx, flag)?;
    value
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid value for {flag}: {err}"))
}

fn print_usage() {
    eprintln!(
        "usage:\n  cargo run -p qwen3-tts-cli -- synthesize --text TEXT --out OUT.wav [--model-dir DIR] [--reference-wav REF.wav | --speaker-bin speaker.bin | --voice-clone-prompt prompt.json] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language-id N]\n  cargo run -p qwen3-tts-cli -- speaker-bin (--wav REF.wav | --voice-clone-prompt prompt.json) --out speaker.bin [--model-dir DIR]"
    );
}
