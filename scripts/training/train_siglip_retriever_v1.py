from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts" / "training" / "train_dual_encoder.py"
TRAIN_FILE = ROOT / "data" / "train" / "retriever_train_pairs.csv"


def main() -> int:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train-file",
        str(TRAIN_FILE),
        "--checkpoint-name",
        "siglip_retriever_v1_best.pt",
        "--summary-name",
        "siglip_retriever_v1_train_summary.json",
        "--pooling",
        "cls",
        "--epochs",
        "8",
        "--batch-size",
        "4",
        "--min-val-queries",
        "2",
        "--val-ratio",
        "0.2",
    ]

    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    os.execv(sys.executable, cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
