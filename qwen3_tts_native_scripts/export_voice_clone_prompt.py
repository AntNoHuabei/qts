"""Export a stage-1 voice clone prompt using upstream qwen-tts.

This module intentionally preserves more fields than qwen3-tts-native uses today.
Stage 1 only consumes `ref_spk_embedding`, but `ref_code`, `ref_text`, and ICL flags
are written too so stage 2 can reuse the same artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "qwen3_tts.voice_clone_prompt.v1"


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
    parser.add_argument("--out", required=True, help="Output JSON path.")
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


def tensor_payload(tensor: Any) -> dict[str, Any]:
    cpu = tensor.detach().cpu()
    return {
        "shape": list(cpu.shape),
        "values": [int(value) for value in cpu.reshape(-1).tolist()],
    }


def tensor_f32_list(tensor: Any) -> list[float]:
    cpu = tensor.detach().cpu().reshape(-1)
    return [float(value) for value in cpu.tolist()]


def build_prompt_payload(
    *,
    model_name_or_path: str,
    ref_audio: str,
    ref_text: str | None,
    x_vector_only_mode: bool,
    device: str,
    dtype: str,
) -> dict[str, Any]:
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
    return {
        "schema": SCHEMA,
        "source": "QwenLM/Qwen3-TTS create_voice_clone_prompt",
        "model_id": model_name_or_path,
        "speaker_encoder_sample_rate_hz": getattr(
            model.model, "speaker_encoder_sample_rate", None
        ),
        "x_vector_only_mode": bool(item.x_vector_only_mode),
        "icl_mode": bool(item.icl_mode),
        "ref_text": item.ref_text,
        "ref_code": None if item.ref_code is None else tensor_payload(item.ref_code),
        "ref_spk_embedding": tensor_f32_list(item.ref_spk_embedding),
    }


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
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote voice clone prompt: path={out_path} "
        f"embedding_dim={len(payload['ref_spk_embedding'])} "
        f"has_ref_code={payload['ref_code'] is not None}"
    )


if __name__ == "__main__":
    main()
