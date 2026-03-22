"""Export a raw speaker.bin from upstream qwen-tts voice clone prompt extraction."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

from qwen3_tts_native_scripts.export_voice_clone_prompt import build_prompt_payload


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
    parser.add_argument("--out", required=True, help="Output speaker.bin path.")
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
    with out_path.open("wb") as handle:
        for value in payload.ref_spk_embedding.values:
            handle.write(struct.pack("<f", float(value)))
    print(
        f"wrote speaker.bin: path={out_path} "
        f"embedding_dim={len(payload.ref_spk_embedding.values)}"
    )


if __name__ == "__main__":
    main()
