from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "ocr" / "page_texts.csv"
OUT_FILE = ROOT / "data" / "ocr" / "page_texts_clean.csv"
SUMMARY_FILE = ROOT / "outputs" / "data_preparation" / "ocr_clean_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean OCR texts and summarize the rebuild output.")
    parser.add_argument("--input-file", type=Path, default=IN_FILE, help="Input OCR CSV.")
    parser.add_argument("--output-file", type=Path, default=OUT_FILE, help="Output cleaned OCR CSV.")
    parser.add_argument("--summary-file", type=Path, default=SUMMARY_FILE, help="Output summary JSON.")
    return parser


def normalize_line(line: str) -> str:
    if not isinstance(line, str):
        return ""
    line = line.strip()
    line = re.sub(r"\s+", " ", line)

    if len(line) <= 1:
        return ""
    if re.fullmatch(r"[^\wA-Za-z]+", line):
        return ""
    if re.fullmatch(r"[-–—]?\d+[-–—]?", line):
        return ""

    return line


def split_lines(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    lines = text.splitlines()
    return [normalize_line(x) for x in lines if normalize_line(x)]


def build_doc_line_stats(df: pd.DataFrame):
    doc_line_pages = defaultdict(lambda: defaultdict(set))

    for row in df.itertuples(index=False):
        lines = split_lines(row.ocr_text)
        unique_lines = set(x.lower() for x in lines)
        for line in unique_lines:
            doc_line_pages[row.doc_id][line].add(int(row.page_idx))

    doc_line_freq = {}
    for doc_id, mapping in doc_line_pages.items():
        doc_line_freq[doc_id] = {key: len(value) for key, value in mapping.items()}
    return doc_line_freq


def is_probable_header_footer(line: str, doc_id: str, doc_line_freq: dict, doc_num_pages: int) -> bool:
    key = line.lower()
    freq = doc_line_freq.get(doc_id, {}).get(key, 0)
    return freq >= 8 and freq / max(doc_num_pages, 1) >= 0.08


def is_low_value_line(line: str) -> bool:
    low = line.lower()
    bad_patterns = [
        r"^lr$",
        r"^rrca$",
        r"^user manual$",
        r"^datasheet$",
    ]
    return any(re.fullmatch(pattern, low) for pattern in bad_patterns)


def clean_page_text(text: str, doc_id: str, doc_line_freq: dict, doc_num_pages: int) -> str:
    lines = split_lines(text)
    cleaned = []
    seen = set()

    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        if is_low_value_line(line):
            continue
        if is_probable_header_footer(line, doc_id, doc_line_freq, doc_num_pages):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def main() -> None:
    args = build_parser().parse_args()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_file)
    doc_num_pages = df.groupby("doc_id").size().to_dict()
    doc_line_freq = build_doc_line_stats(df)

    df["ocr_text_clean"] = df.apply(
        lambda row: clean_page_text(
            row["ocr_text"] if isinstance(row["ocr_text"], str) else "",
            row["doc_id"],
            doc_line_freq,
            doc_num_pages[row["doc_id"]],
        ),
        axis=1,
    )
    df["ocr_text_len"] = df["ocr_text"].fillna("").str.len()
    df["ocr_text_clean_len"] = df["ocr_text_clean"].fillna("").str.len()
    df["ocr_reduction"] = df["ocr_text_len"] - df["ocr_text_clean_len"]

    df.to_csv(args.output_file, index=False, encoding="utf-8")

    summary = {
        "rows": int(len(df)),
        "documents": int(df["doc_id"].nunique()) if len(df) else 0,
        "non_empty_raw_pages": int((df["ocr_text"].fillna("").str.len() > 0).sum()),
        "non_empty_clean_pages": int((df["ocr_text_clean"].fillna("").str.len() > 0).sum()),
        "error_pages": int((df["error"].fillna("").str.len() > 0).sum()) if "error" in df.columns else 0,
        "avg_clean_length": float(df["ocr_text_clean_len"].mean()) if len(df) else 0.0,
        "avg_reduction": float(df["ocr_reduction"].mean()) if len(df) else 0.0,
    }

    with args.summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved cleaned OCR table to: {args.output_file}")
    print(f"Saved OCR clean summary to: {args.summary_file}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
