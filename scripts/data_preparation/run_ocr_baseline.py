from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from paddlex import create_pipeline


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest.csv"
RAW_JSON_DIR = ROOT / "data" / "ocr" / "raw_json"
OCR_TABLE = ROOT / "data" / "ocr" / "page_texts.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OCR over rendered page images.")
    parser.add_argument("--manifest-file", type=Path, default=MANIFEST, help="Input page manifest CSV.")
    parser.add_argument("--raw-json-dir", type=Path, default=RAW_JSON_DIR, help="Directory for raw OCR json files.")
    parser.add_argument("--output-file", type=Path, default=OCR_TABLE, help="Output OCR CSV.")
    parser.add_argument("--device", default="gpu:0", help="PaddleX OCR device, for example gpu:0 or cpu.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pages.")
    parser.add_argument("--doc-id", default=None, help="Only process pages from one doc_id.")
    return parser


def extract_text_from_result(res_json):
    info = res_json.get("res", res_json)

    rec_texts = info.get("rec_texts", [])
    rec_scores = info.get("rec_scores", [])
    rec_boxes = info.get("rec_boxes", [])

    kept_texts = []
    kept_scores = []

    for text, score in zip(rec_texts, rec_scores):
        if isinstance(text, str) and text.strip():
            kept_texts.append(text.strip())
            try:
                kept_scores.append(float(score))
            except Exception:
                pass

    merged_text = "\n".join(kept_texts)
    avg_score = sum(kept_scores) / len(kept_scores) if kept_scores else 0.0

    return merged_text, avg_score, len(rec_boxes)


def main() -> None:
    args = build_parser().parse_args()

    args.raw_json_dir.mkdir(parents=True, exist_ok=True)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest_file)
    if args.doc_id:
        df = df[df["doc_id"] == args.doc_id].copy()
    if args.limit is not None:
        df = df.head(args.limit).copy()

    pipeline = create_pipeline(pipeline="OCR", device=args.device)
    rows = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Running OCR on {args.device}"):
        image_path = str(row.image_path)
        doc_id = row.doc_id
        page_idx = int(row.page_idx)
        page_id = getattr(row, "page_id", f"{doc_id}:{page_idx:04d}")

        try:
            output = pipeline.predict(
                input=image_path,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

            res = next(iter(output))
            res_json = res.json
            merged_text, avg_score, num_boxes = extract_text_from_result(res_json)

            json_path = args.raw_json_dir / f"{doc_id}_page_{page_idx:04d}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(res_json, f, ensure_ascii=False, indent=2)

            rows.append(
                {
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "page_idx": page_idx,
                    "image_path": image_path,
                    "ocr_json_path": str(json_path),
                    "num_boxes": num_boxes,
                    "avg_score": avg_score,
                    "ocr_text": merged_text,
                    "error": "",
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "page_idx": page_idx,
                    "image_path": image_path,
                    "ocr_json_path": "",
                    "num_boxes": 0,
                    "avg_score": 0.0,
                    "ocr_text": "",
                    "error": str(exc),
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["doc_id", "page_idx"]).reset_index(drop=True)
    out_df.to_csv(args.output_file, index=False, encoding="utf-8")

    print(f"Saved OCR table to: {args.output_file}")
    print(f"Total rows: {len(out_df)}")
    print(f"Non-empty OCR pages: {(out_df['ocr_text'].fillna('').str.len() > 0).sum()}")
    print(f"Error pages: {(out_df['error'].fillna('').str.len() > 0).sum()}")


if __name__ == "__main__":
    main()
