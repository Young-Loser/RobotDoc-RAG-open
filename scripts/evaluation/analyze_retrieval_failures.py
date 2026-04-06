from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs" / "siglip_retriever_v2_eval_details.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "reports" / "siglip_v2_failures.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export retrieval failure cases for manual analysis.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT, help="Evaluation detail CSV.")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT, help="Failure CSV output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_file)
    failures = df[(~df["top1_exact_hit"]) | (~df["top5_exact_hit"])].copy()
    failures = failures[
        [
            "query_id",
            "query_type",
            "query_en",
            "gold_doc_id",
            "gold_page_idx",
            "top1_doc_id",
            "top1_page_idx",
            "top1_doc_hit",
            "top1_exact_hit",
            "top5_doc_hit",
            "top5_exact_hit",
            "retrieved_doc_ids",
            "retrieved_page_idxs",
            "retrieved_scores",
        ]
    ]
    failures.to_csv(args.output_file, index=False, encoding="utf-8")

    print(f"Saved failure analysis to: {args.output_file}")
    print(f"num_failures = {len(failures)}")


if __name__ == "__main__":
    main()
