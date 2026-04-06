from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_FILE = ROOT / "data" / "manifest.csv"
OCR_CLEAN_FILE = ROOT / "data" / "ocr" / "page_texts_clean.csv"
DOC_MANIFEST_FILE = ROOT / "data" / "documents.csv"
PAGE_INDEX_FILE = ROOT / "data" / "page_index.csv"
SUMMARY_FILE = ROOT / "outputs" / "data_preparation" / "data_rebuild_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a unified page index from render and OCR outputs.")
    parser.add_argument("--manifest-file", type=Path, default=MANIFEST_FILE, help="Rendered page manifest CSV.")
    parser.add_argument("--ocr-clean-file", type=Path, default=OCR_CLEAN_FILE, help="Clean OCR CSV.")
    parser.add_argument("--doc-manifest-file", type=Path, default=DOC_MANIFEST_FILE, help="Document manifest CSV.")
    parser.add_argument("--page-index-file", type=Path, default=PAGE_INDEX_FILE, help="Output unified page index CSV.")
    parser.add_argument("--summary-file", type=Path, default=SUMMARY_FILE, help="Output rebuild summary JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    args.page_index_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)

    df_manifest = pd.read_csv(args.manifest_file)
    df_ocr = pd.read_csv(args.ocr_clean_file)
    df_docs = pd.read_csv(args.doc_manifest_file) if args.doc_manifest_file.exists() else pd.DataFrame()

    join_keys = ["page_id", "doc_id", "page_idx", "image_path"]
    use_keys = [key for key in join_keys if key in df_manifest.columns and key in df_ocr.columns]
    df_page_index = df_manifest.merge(df_ocr, on=use_keys, how="left", suffixes=("", "_ocr"))

    if not df_docs.empty and "doc_id" in df_docs.columns:
        df_page_index = df_page_index.merge(df_docs, on="doc_id", how="left", suffixes=("", "_doc"))

    df_page_index = df_page_index.sort_values(["doc_id", "page_idx"]).reset_index(drop=True)
    df_page_index["has_ocr_text"] = df_page_index["ocr_text"].fillna("").str.len() > 0
    df_page_index["has_clean_text"] = df_page_index["ocr_text_clean"].fillna("").str.len() > 0
    df_page_index["has_ocr_error"] = df_page_index["error"].fillna("").str.len() > 0

    df_page_index.to_csv(args.page_index_file, index=False, encoding="utf-8")

    summary = {
        "documents": int(df_page_index["doc_id"].nunique()) if len(df_page_index) else 0,
        "pages": int(len(df_page_index)),
        "pages_with_ocr_text": int(df_page_index["has_ocr_text"].sum()) if len(df_page_index) else 0,
        "pages_with_clean_text": int(df_page_index["has_clean_text"].sum()) if len(df_page_index) else 0,
        "pages_with_ocr_error": int(df_page_index["has_ocr_error"].sum()) if len(df_page_index) else 0,
        "doc_ids": sorted(df_page_index["doc_id"].dropna().unique().tolist()) if len(df_page_index) else [],
    }

    with args.summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved page index to: {args.page_index_file}")
    print(f"Saved rebuild summary to: {args.summary_file}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
