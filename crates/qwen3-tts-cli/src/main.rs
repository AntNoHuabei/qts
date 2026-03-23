use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use qwen3_tts::{Qwen3TtsEngine, SynthesisStageTimings, VoiceClonePromptV2};

mod cli_support;
mod tui;

use crate::cli_support::{
    default_model_dir, load_engine, parse_value_arg, value_arg, CommonSynthesisArgs,
    RuntimeBackendOverrides,
};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("speaker-bin") => run_speaker_bin(args.collect()),
        Some("synthesize") => run_synthesize(args.collect()),
        Some("profile") => run_profile(args.collect()),
        Some("tui") => tui::run(args.collect()),
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn run_speaker_bin(args: Vec<String>) -> Result<()> {
    let mut model_dir = default_model_dir()?;
    let mut prompt_path = None;
    let mut out_path = None;
    let mut runtime_backends = RuntimeBackendOverrides::default();

    let mut idx = 0;
    while idx < args.len() {
        if runtime_backends.parse_flag(&args, &mut idx)? {
            continue;
        }
        match args[idx].as_str() {
            "--model-dir" => {
                model_dir = PathBuf::from(value_arg(&args, &mut idx, "--model-dir")?);
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
    let prompt_path = prompt_path.context("--voice-clone-prompt is required")?;
    let engine = load_engine(&model_dir, &runtime_backends)?;
    let prompt_bytes = fs::read(&prompt_path)
        .with_context(|| format!("failed to read {}", prompt_path.display()))?;
    let prompt = engine.decode_voice_clone_prompt(&prompt_bytes)?;
    let speaker_bin = prompt
        .speaker_embedding()
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
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
    let mut common = CommonSynthesisArgs::new()?;

    let mut idx = 0;
    while idx < args.len() {
        if common.parse_flag(&args, &mut idx)? {
            continue;
        }
        bail!("unknown synthesize argument: {}", args[idx]);
    }

    common.validate_conditioning()?;
    let text = common.require_text()?;
    let out_path = common.require_out_path()?;
    let engine = load_engine(&common.model_dir, &common.runtime_backends)?;
    let request = common.build_request(text)?;
    let conditioning = load_synthesis_conditioning(&engine, &common)?;

    let result = if let LoadedConditioning::VoiceClonePrompt(prompt) = &conditioning {
        engine.synthesize_with_voice_clone_prompt(&request, prompt)?
    } else if let LoadedConditioning::SpeakerEmbedding(embedding) = &conditioning {
        engine.synthesize_with_speaker_embedding(&request, embedding)?
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

fn run_profile(args: Vec<String>) -> Result<()> {
    if args
        .iter()
        .any(|a| matches!(a.as_str(), "--help" | "-h"))
    {
        eprintln!(
            "qwen3-tts-cli profile — print per-stage synthesis timings (wall clock)\n\n\
             usage:\n  profile --text TEXT [--model-dir DIR] [--runs N] [--out OUT.wav] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language-id N] [--vocoder-threads N] [--chunk-size N] [--reference-wav PATH | --speaker-bin PATH | --voice-clone-prompt PATH] [--backend auto|cpu|metal|vulkan] [--backend-fallback LIST] [--vocoder-ep auto|cpu|coreml] [--vocoder-ep-fallback LIST]\n\n\
             CLI flags override environment variables.\n\
             Default transformer auto chain: Apple = metal,vulkan,cpu ; others = vulkan,cpu.\n\
             Default vocoder auto chain: Apple = coreml,cpu ; others = cpu.\n\n\
             If --frames is omitted, the CLI derives a text-length-based max frame budget.\n\
             --runs N averages stage times over N full synthesize passes (default 1).\n\
             --out writes WAV from the first pass only when --runs > 1."
        );
        return Ok(());
    }

    let mut common = CommonSynthesisArgs::new()?;
    let mut runs = 1usize;

    let mut idx = 0;
    while idx < args.len() {
        if common.parse_flag(&args, &mut idx)? {
            continue;
        }
        match args[idx].as_str() {
            "--runs" => {
                runs = parse_value_arg(&args, &mut idx, "--runs")?;
            }
            other => bail!("unknown profile argument: {other}"),
        }
    }

    if runs == 0 {
        bail!("--runs must be >= 1");
    }

    common.validate_conditioning()?;
    let text = common.require_text()?;
    let engine = load_engine(&common.model_dir, &common.runtime_backends)?;
    let request = common.build_request(text)?;
    let conditioning = load_synthesis_conditioning(&engine, &common)?;

    let mut samples = Vec::with_capacity(runs);
    let mut first_result = None;

    for run_idx in 0..runs {
        let (result, timings) = match &conditioning {
            LoadedConditioning::VoiceClonePrompt(prompt) => {
                engine.synthesize_with_voice_clone_prompt_profile(&request, prompt)?
            }
            LoadedConditioning::SpeakerEmbedding(embedding) => {
                engine.synthesize_with_speaker_embedding_profile(&request, embedding)?
            }
            LoadedConditioning::None => engine.synthesize_with_profile(&request)?,
        };
        if run_idx == 0 {
            first_result = Some(result);
        }
        samples.push(timings);
    }

    if let Some(path) = common.out_path {
        let result = first_result.context("internal: missing first synthesis result")?;
        write_wav_f32(&path, result.sample_rate_hz, &result.pcm_f32)?;
        eprintln!(
            "wrote profile run #1 WAV: path={} sample_rate={} frames={}",
            path.display(),
            result.sample_rate_hz,
            result.generated_frames
        );
    }

    let summary = if runs == 1 {
        samples[0].format_table()
    } else {
        let avg = SynthesisStageTimings::average(&samples).expect("non-empty samples");
        format!(
            "averaged over {runs} runs\n{}",
            avg.format_table()
        )
    };
    eprint!("{summary}");

    Ok(())
}

enum LoadedConditioning {
    None,
    SpeakerEmbedding(Vec<f32>),
    VoiceClonePrompt(VoiceClonePromptV2),
}

fn load_synthesis_conditioning(
    engine: &Qwen3TtsEngine,
    args: &CommonSynthesisArgs,
) -> Result<LoadedConditioning> {
    if let Some(path) = &args.voice_clone_prompt {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let prompt = engine.decode_voice_clone_prompt(&bytes)?;
        return Ok(LoadedConditioning::VoiceClonePrompt(prompt));
    }
    if let Some(path) = &args.speaker_bin {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let embedding = engine.decode_speaker_embedding_bin(&bytes)?;
        return Ok(LoadedConditioning::SpeakerEmbedding(embedding));
    }
    Ok(LoadedConditioning::None)
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

fn print_usage() {
    eprintln!(
        "usage:\n  cargo run -p qwen3-tts-cli -- synthesize --text TEXT --out OUT.wav [--model-dir DIR] [--reference-wav REF.wav | --speaker-bin speaker.bin | --voice-clone-prompt prompt.pb] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language-id N] [--vocoder-threads N] [--chunk-size N] [--backend auto|cpu|metal|vulkan] [--backend-fallback LIST] [--vocoder-ep auto|cpu|coreml] [--vocoder-ep-fallback LIST]\n  cargo run -p qwen3-tts-cli -- profile --text TEXT [--model-dir DIR] [--runs N] [--out OUT.wav] [--reference-wav | --speaker-bin | --voice-clone-prompt] (same tuning flags as synthesize plus backend flags)\n  cargo run -p qwen3-tts-cli -- speaker-bin --voice-clone-prompt prompt.pb --out speaker.bin [--model-dir DIR] [--backend ...] [--vocoder-ep ...]\n  cargo run -p qwen3-tts-cli -- tui [--model-dir DIR] [--reference-wav REF.wav | --speaker-bin speaker.bin | --voice-clone-prompt prompt.pb] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language-id N] [--vocoder-threads N] [--chunk-size N] [--backend auto|cpu|metal|vulkan] [--backend-fallback LIST] [--vocoder-ep auto|cpu|coreml] [--vocoder-ep-fallback LIST]\n\nCLI flags override environment variables.\nDefault transformer auto chain: Apple = metal,vulkan,cpu ; others = vulkan,cpu.\nDefault vocoder auto chain: Apple = coreml,cpu ; others = cpu.\n\nEnv fallback remains available: QWEN3_TTS_BACKEND / QWEN3_TTS_BACKEND_FALLBACK / QWEN3_TTS_VOCODER_EP / QWEN3_TTS_VOCODER_EP_FALLBACK\n\nIf --frames is omitted, synthesize/profile derive a text-length-based max frame budget.\n\nOr from the repo root (see .cargo/config.toml): cargo xtask bench … / cargo xtask profile …"
    );
}
