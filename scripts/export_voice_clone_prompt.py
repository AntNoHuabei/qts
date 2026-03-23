#!/usr/bin/env python3
"""Export a stage-2 protobuf voice clone prompt using upstream qwen-tts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts import voice_clone_prompt_pb2

SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Upstream Qwen3-TTS Base model id or local path."
    )
    parser.add_argument(
        "--ref-audio",
        required=True,
        help="Reference audio path/URL/base64/path-like accepted by qwen-tts.",
    )
    parser.add_argument("--out", required=True, help="Output protobuf .pb path.")
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Reference transcript. Required unless --x-vector-only-mode is set.",
    )
    parser.add_argument(
        "--x-vector-only-mode",
        action="store_true",
        help="Export only the speaker embedding path from upstream prompt creation.",
    )
    parser.add_argument("--device", default="cpu", help="Device map passed to qwen-tts.")
    parser.add_argument(
        "--dtype",
        default="auto",
        help="dtype passed to qwen-tts (for example: auto, float16, bfloat16).",
    )
    return parser.parse_args()


def tensor_i32_payload(tensor: Any) -> voice_clone_prompt_pb2.TensorI32:
    cpu = tensor.detach().cpu()
    return voice_clone_prompt_pb2.TensorI32(
        shape=list(cpu.shape),
        values=[int(value) for value in cpu.reshape(-1).tolist()],
    )


def tensor_f32_payload(tensor: Any) -> voice_clone_prompt_pb2.TensorF32:
    cpu = tensor.detach().cpu().reshape(-1)
    return voice_clone_prompt_pb2.TensorF32(
        shape=list(cpu.shape),
        values=[float(value) for value in cpu.tolist()],
    )


def build_prompt_payload(
    *,
    model_name_or_path: str,
    ref_audio: str,
    ref_text: str | None,
    x_vector_only_mode: bool,
    device: str,
    dtype: str,
) -> voice_clone_prompt_pb2.VoiceClonePromptV2:
    if not x_vector_only_mode and not ref_text:
        raise SystemExit("--ref-text is required unless --x-vector-only-mode is set")

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise SystemExit(
            "qwen-tts is required. Install the upstream package first, for example: uv sync"
        ) from exc

    model = Qwen3TTSModel.from_pretrained(
        model_name_or_path,
        device_map=device,
        dtype=resolve_dtype(dtype),
    )
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
    )
    if len(prompt_items) != 1:
        raise SystemExit(f"expected exactly one prompt item, got {len(prompt_items)}")

    item = prompt_items[0]
    prompt = voice_clone_prompt_pb2.VoiceClonePromptV2(
        schema_version=SCHEMA_VERSION,
        source="QwenLM/Qwen3-TTS create_voice_clone_prompt",
        model_id=model_name_or_path,
        speaker_encoder_sample_rate_hz=int(
            getattr(model.model, "speaker_encoder_sample_rate", 0) or 0
        ),
        x_vector_only_mode=bool(item.x_vector_only_mode),
        icl_mode=bool(item.icl_mode),
        ref_text=item.ref_text or "",
        ref_spk_embedding=tensor_f32_payload(item.ref_spk_embedding),
    )
    if item.ref_code is not None:
        prompt.ref_code.CopyFrom(tensor_i32_payload(item.ref_code))
    return prompt


def resolve_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise SystemExit(f"unsupported --dtype value: {name}") from exc


def main() -> None:
    args = parse_args()
    payload = build_prompt_payload(
        model_name_or_path=args.model,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only_mode,
        device=args.device,
        dtype=args.dtype,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload.SerializeToString())
    print(
        f"wrote voice clone prompt: path={out_path} "
        f"embedding_dim={len(payload.ref_spk_embedding.values)} "
        f"has_ref_code={payload.HasField('ref_code')}"
    )


if __name__ == "__main__":
    main()
