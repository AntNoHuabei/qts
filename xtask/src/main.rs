use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("bench") => run_bench(args.collect()),
        Some("profile") => run_profile(args.collect()),
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn run_bench(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = workspace_root()?;
    let mut backend = String::from("cpu");
    let mut model_dir = default_model_dir(&workspace_root);
    let mut text = String::from("hello");
    let mut threads = String::from("4");
    let mut frames = String::from("16");
    let mut temperature = String::from("0.0");
    let mut top_k = String::from("0");
    let mut top_p = String::from("1.0");
    let mut criterion_args = Vec::new();

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "cpu" | "metal" | "vulkan" | "all" | "both" => {
                backend = args[idx].clone();
                idx += 1;
            }
            "--model-dir" => {
                model_dir = PathBuf::from(value_arg(&args, &mut idx, "--model-dir")?);
            }
            "--text" => {
                text = value_arg(&args, &mut idx, "--text")?;
            }
            "--threads" => {
                threads = value_arg(&args, &mut idx, "--threads")?;
            }
            "--frames" => {
                frames = value_arg(&args, &mut idx, "--frames")?;
            }
            "--temperature" => {
                temperature = value_arg(&args, &mut idx, "--temperature")?;
            }
            "--top-k" => {
                top_k = value_arg(&args, &mut idx, "--top-k")?;
            }
            "--top-p" => {
                top_p = value_arg(&args, &mut idx, "--top-p")?;
            }
            "--" => {
                criterion_args.extend_from_slice(&args[idx + 1..]);
                break;
            }
            other => {
                return Err(format!("unknown xtask bench argument: {other}").into());
            }
        }
    }

    let backends = match backend.as_str() {
        "cpu" => vec!["cpu"],
        "metal" => vec!["metal"],
        "vulkan" => vec!["vulkan"],
        "all" => vec!["cpu", "metal", "vulkan"],
        "both" => vec!["cpu", "metal"],
        _ => return Err(format!("unsupported backend: {backend}").into()),
    };

    for backend in backends {
        run_single_bench(
            &workspace_root,
            backend,
            &model_dir,
            &text,
            &threads,
            &frames,
            &temperature,
            &top_k,
            &top_p,
            &criterion_args,
        )?;
    }

    Ok(())
}

fn run_profile(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    if args
        .iter()
        .any(|a| matches!(a.as_str(), "--help" | "-h"))
    {
        print_profile_help();
        return Ok(());
    }

    let workspace_root = workspace_root()?;
    let mut backend = String::from("cpu");
    let mut forwarded = args;
    if let Some(first) = forwarded.first() {
        if matches!(
            first.as_str(),
            "cpu" | "metal" | "vulkan" | "all" | "both"
        ) {
            backend = forwarded.remove(0);
        }
    }

    let backends = match backend.as_str() {
        "cpu" => vec!["cpu"],
        "metal" => vec!["metal"],
        "vulkan" => vec!["vulkan"],
        "all" => vec!["cpu", "metal", "vulkan"],
        "both" => vec!["cpu", "metal"],
        _ => return Err(format!("unsupported backend: {backend}").into()),
    };

    for backend in backends {
        let mut command = Command::new("cargo");
        command.current_dir(&workspace_root);
        command.args(["run", "-p", "qwen3-tts-cli"]);
        if backend != "cpu" {
            command.args(["--features", backend]);
        }
        // Decouple Cargo features from GGML backend selection: on macOS, Vulkan is only used when
        // explicitly requested (see `QWEN3_TTS_BACKEND` in qwen3-tts `backend.rs`).
        command.env("QWEN3_TTS_BACKEND", backend);
        command.arg("--");
        command.arg("profile");
        command.args(&forwarded);

        eprintln!(
            "running synthesis profile: cargo_features={backend} QWEN3_TTS_BACKEND={backend}"
        );

        let status = command.status()?;
        if !status.success() {
            return Err(format!("profile failed for backend={backend}").into());
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_single_bench(
    workspace_root: &Path,
    backend: &str,
    model_dir: &Path,
    text: &str,
    threads: &str,
    frames: &str,
    temperature: &str,
    top_k: &str,
    top_p: &str,
    criterion_args: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut command = Command::new("cargo");
    command.current_dir(workspace_root);
    command.arg("bench");
    command.arg("-p").arg("qwen3-tts");
    command.arg("--bench").arg("synthesize");
    match backend {
        "metal" => {
            command.arg("--features").arg("metal");
        }
        "vulkan" => {
            command.arg("--features").arg("vulkan");
        }
        _ => {}
    }
    command.arg("--");
    for arg in criterion_args {
        command.arg(arg);
    }

    command.env(
        "CARGO_TARGET_DIR",
        workspace_root.join(format!("target/bench-{backend}")),
    );
    command.env("QWEN3_TTS_BENCH_BACKEND", backend);
    command.env("QWEN3_TTS_BENCH_MODEL_DIR", model_dir);
    command.env("QWEN3_TTS_BENCH_TEXT", text);
    command.env("QWEN3_TTS_BENCH_THREADS", threads);
    command.env("QWEN3_TTS_BENCH_MAX_AUDIO_FRAMES", frames);
    command.env("QWEN3_TTS_BENCH_TEMPERATURE", temperature);
    command.env("QWEN3_TTS_BENCH_TOP_K", top_k);
    command.env("QWEN3_TTS_BENCH_TOP_P", top_p);

    eprintln!(
        "running criterion bench: backend={backend} model_dir={} threads={threads} frames={frames}",
        model_dir.display()
    );

    let status = command.status()?;
    if !status.success() {
        return Err(format!("criterion benchmark failed for backend={backend}").into());
    }
    Ok(())
}

fn default_model_dir(workspace_root: &Path) -> PathBuf {
    workspace_root.join("models/volko76-q4k-q8")
}

fn workspace_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("xtask manifest has no workspace parent")?
        .to_path_buf())
}

fn value_arg(
    args: &[String],
    idx: &mut usize,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    *idx += 1;
    let value = args
        .get(*idx)
        .ok_or_else(|| format!("missing value for {flag}"))?
        .clone();
    *idx += 1;
    Ok(value)
}

fn print_usage() {
    eprintln!(
        "usage:\n  cargo xtask bench [cpu|metal|vulkan|all|both] [--model-dir PATH] [--text TEXT] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [-- <criterion args>]\n  cargo xtask profile [cpu|metal|vulkan|all|both] [--model-dir PATH] [--text TEXT] [--runs N] [--out OUT.wav] [--reference-wav | --speaker-bin | --voice-clone-prompt] [... same flags as synthesize ...]\n\nTry: cargo xtask profile --help"
    );
}

fn print_profile_help() {
    eprintln!(
        "cargo xtask profile — run qwen3-tts-cli profile with optional backend feature\n\n\
         usage:\n  cargo xtask profile [cpu|metal|vulkan|all|both] [-- ARGS_FOR_CLI...]\n\n\
         The first token selects both `cargo --features` and sets QWEN3_TTS_BACKEND for the child \
         (so macOS + vulkan actually uses the Vulkan GGML backend, not only links it).\n\n\
         Examples:\n  cargo xtask profile cpu --model-dir models/volko76-q4k-q8 --text hello --frames 32 --runs 3\n  cargo xtask profile metal --model-dir \"$QWEN3_TTS_MODEL_DIR\" --text hello --frames 64 --out /tmp/p.wav\n  cargo xtask profile vulkan --model-dir \"$QWEN3_TTS_MODEL_DIR\" --text hello --frames 64\n\n\
         All flags after the optional backend token are passed to `qwen3-tts-cli profile`."
    );
}
