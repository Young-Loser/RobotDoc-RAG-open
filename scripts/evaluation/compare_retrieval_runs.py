from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"
REPORT_DIR = OUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "bm25": OUT_DIR / "bm25_eval_details.csv",
    "siglip_v1": OUT_DIR / "siglip_retriever_v1_eval_details.csv",
    "siglip_v2": OUT_DIR / "siglip_retriever_v2_eval_details.csv",
    "two_stage": OUT_DIR / "two_stage_rerank_eval_details.csv",
}

COMPARE_CSV = REPORT_DIR / "retrieval_comparison.csv"
BEST_BY_QUERY_CSV = REPORT_DIR / "retrieval_best_by_query.csv"
SUMMARY_JSON = REPORT_DIR / "retrieval_comparison_summary.json"


def main() -> None:
    frames = []
    available_runs = []

    for run_name, detail_file in RUNS.items():
        if not detail_file.exists():
            continue
        df = pd.read_csv(detail_file).copy()
        df["run_name"] = run_name
        frames.append(df)
        available_runs.append(run_name)

    if not frames:
        raise FileNotFoundError("No evaluation detail files found to compare.")

    df_all = pd.concat(frames, ignore_index=True)
    compare_cols = [
        "run_name",
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
    ]
    df_compare = df_all[compare_cols].sort_values(["query_id", "run_name"]).reset_index(drop=True)
    df_compare.to_csv(COMPARE_CSV, index=False, encoding="utf-8")

    priority = df_all["top1_exact_hit"].astype(int) * 4 + df_all["top5_exact_hit"].astype(int) * 2 + df_all["top1_doc_hit"].astype(int)
    df_all = df_all.assign(priority=priority)
    df_best = (
        df_all.sort_values(["query_id", "priority", "run_name"], ascending=[True, False, True])
        .groupby("query_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    df_best[compare_cols].to_csv(BEST_BY_QUERY_CSV, index=False, encoding="utf-8")

    summary = {
        "available_runs": available_runs,
        "num_queries": int(df_all["query_id"].nunique()),
        "run_metrics": {},
    }
    for run_name, group in df_all.groupby("run_name"):
        summary["run_metrics"][run_name] = {
            "top1_exact_acc": float(group["top1_exact_hit"].mean()),
            "top5_exact_recall": float(group["top5_exact_hit"].mean()),
            "top1_doc_acc": float(group["top1_doc_hit"].mean()),
            "top5_doc_recall": float(group["top5_doc_hit"].mean()),
        }

    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:")
    print(" -", COMPARE_CSV)
    print(" -", BEST_BY_QUERY_CSV)
    print(" -", SUMMARY_JSON)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
