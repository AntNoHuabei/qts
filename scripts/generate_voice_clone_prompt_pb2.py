#!/usr/bin/env python3
"""Regenerate Python protobuf bindings from the canonical qts schema."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    proto_root = repo_root / "crates" / "qts" / "proto"
    proto_file = proto_root / "voice_clone_prompt.proto"
    out_dir = repo_root / "scripts"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{proto_root}",
            f"--python_out={out_dir}",
            str(proto_file),
        ],
        check=True,
    )

    print(f"generated {out_dir / 'voice_clone_prompt_pb2.py'} from {proto_file}")


if __name__ == "__main__":
    main()
