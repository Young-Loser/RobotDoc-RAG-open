from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts" / "training" / "train_dual_encoder.py"
TRAIN_FILE = ROOT / "data" / "train" / "retriever_train_pairs_large_v4.csv"


def main() -> int:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train-file",
        str(TRAIN_FILE),
        "--checkpoint-name",
        "siglip_retriever_v2_best.pt",
        "--summary-name",
        "siglip_retriever_v2_train_summary.json",
        "--pooling",
        "position_weighted",
        "--text-pooling",
        "position_weighted",
        "--image-pooling",
        "pretrained",
        "--epochs",
        "6",
        "--batch-size",
        "8",
        "--learning-rate",
        "5e-5",
        "--margin-weight",
        "0.25",
        "--max-length",
        "128",
        "--val-ratio",
        "0.12",
        "--min-val-queries",
        "20",
        "--use-weighted-sampler",
    ]

    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    os.execv(sys.executable, cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
