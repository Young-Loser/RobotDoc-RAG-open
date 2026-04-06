from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_FILE = ROOT / "data" / "train" / "retriever_train_pairs_large_v3.csv"
QUERY_FILE = ROOT / "data" / "eval" / "query_gold_v1.csv"
BM25_FILE = ROOT / "outputs" / "bm25_eval_details.csv"
SIGLIP_V1_FILE = ROOT / "outputs" / "siglip_retriever_v1_eval_details.csv"
TWO_STAGE_FILE = ROOT / "outputs" / "two_stage_rerank_eval_details.csv"
PAGE_INDEX_FILE = ROOT / "data" / "page_index.csv"

OUT_FILE = ROOT / "data" / "train" / "retriever_train_pairs_large_v4.csv"


def page_path(doc_id: str, page_idx: int) -> str:
    return str(ROOT / "data" / "pages" / doc_id / f"page_{page_idx:04d}.png")


def parse_json_list(value):
    if isinstance(value, str) and value.strip():
        return json.loads(value)
    return []


def first_wrong_from_result(row, gold_doc: str, gold_page: int):
    docs = parse_json_list(row["retrieved_doc_ids"])
    pages = parse_json_list(row["retrieved_page_idxs"])
    for doc_id, page_idx in zip(docs, pages):
        if not (doc_id == gold_doc and int(page_idx) == gold_page):
            return doc_id, int(page_idx)
    return None, None


def add_pair(rows: list[dict], *, sample_id: str, query_id: str, query_en: str, query_zh: str, query_type: str, gold_doc: str, gold_page: int, neg_doc: str, neg_page: int, neg_source: str):
    if neg_doc is None:
        return
    rows.append(
        {
            "sample_id": sample_id,
            "query_id": query_id,
            "query_en": query_en,
            "query_zh": query_zh,
            "query_type": query_type,
            "pos_doc_id": gold_doc,
            "pos_page_idx": gold_page,
            "pos_image_path": page_path(gold_doc, gold_page),
            "neg_doc_id": neg_doc,
            "neg_page_idx": neg_page,
            "neg_image_path": page_path(neg_doc, neg_page),
            "neg_source": neg_source,
        }
    )


def main() -> None:
    df_base = pd.read_csv(BASE_FILE)
    df_q = pd.read_csv(QUERY_FILE)
    df_q = df_q[df_q["label_status"] == "confirmed"].reset_index(drop=True)
    df_page_index = pd.read_csv(PAGE_INDEX_FILE)

    df_bm25 = pd.read_csv(BM25_FILE) if BM25_FILE.exists() else pd.DataFrame()
    df_siglip_v1 = pd.read_csv(SIGLIP_V1_FILE) if SIGLIP_V1_FILE.exists() else pd.DataFrame()
    df_two_stage = pd.read_csv(TWO_STAGE_FILE) if TWO_STAGE_FILE.exists() else pd.DataFrame()

    bm25_map = {row["query_id"]: row for _, row in df_bm25.iterrows()}
    siglip_v1_map = {row["query_id"]: row for _, row in df_siglip_v1.iterrows()}
    two_stage_map = {row["query_id"]: row for _, row in df_two_stage.iterrows()}

    doc_page_set = set(zip(df_page_index["doc_id"], df_page_index["page_idx"].astype(int)))
    extra_rows = []

    for row in df_q.itertuples(index=False):
        gold_doc = row.final_doc_id
        gold_page = int(row.final_page_idx)

        if row.query_id in bm25_map:
            neg_doc, neg_page = first_wrong_from_result(bm25_map[row.query_id], gold_doc, gold_page)
            add_pair(
                extra_rows,
                sample_id=f"{row.query_id}_gold_bm25",
                query_id=row.query_id,
                query_en=row.query_en,
                query_zh=row.query_zh,
                query_type=row.query_type,
                gold_doc=gold_doc,
                gold_page=gold_page,
                neg_doc=neg_doc,
                neg_page=neg_page,
                neg_source="gold_bm25_hard",
            )

        if row.query_id in siglip_v1_map:
            neg_doc, neg_page = first_wrong_from_result(siglip_v1_map[row.query_id], gold_doc, gold_page)
            add_pair(
                extra_rows,
                sample_id=f"{row.query_id}_gold_siglip_v1",
                query_id=row.query_id,
                query_en=row.query_en,
                query_zh=row.query_zh,
                query_type=row.query_type,
                gold_doc=gold_doc,
                gold_page=gold_page,
                neg_doc=neg_doc,
                neg_page=neg_page,
                neg_source="gold_siglip_v1_hard",
            )

        if row.query_id in two_stage_map:
            neg_doc, neg_page = first_wrong_from_result(two_stage_map[row.query_id], gold_doc, gold_page)
            add_pair(
                extra_rows,
                sample_id=f"{row.query_id}_gold_two_stage",
                query_id=row.query_id,
                query_en=row.query_en,
                query_zh=row.query_zh,
                query_type=row.query_type,
                gold_doc=gold_doc,
                gold_page=gold_page,
                neg_doc=neg_doc,
                neg_page=neg_page,
                neg_source="gold_two_stage_hard",
            )

        if gold_page > 0 and (gold_doc, gold_page - 1) in doc_page_set:
            add_pair(
                extra_rows,
                sample_id=f"{row.query_id}_gold_adjacent_prev",
                query_id=row.query_id,
                query_en=row.query_en,
                query_zh=row.query_zh,
                query_type=row.query_type,
                gold_doc=gold_doc,
                gold_page=gold_page,
                neg_doc=gold_doc,
                neg_page=gold_page - 1,
                neg_source="gold_adjacent_prev",
            )
        if (gold_doc, gold_page + 1) in doc_page_set:
            add_pair(
                extra_rows,
                sample_id=f"{row.query_id}_gold_adjacent_next",
                query_id=row.query_id,
                query_en=row.query_en,
                query_zh=row.query_zh,
                query_type=row.query_type,
                gold_doc=gold_doc,
                gold_page=gold_page,
                neg_doc=gold_doc,
                neg_page=gold_page + 1,
                neg_source="gold_adjacent_next",
            )

    df_extra = pd.DataFrame(extra_rows)
    df_out = pd.concat([df_base, df_extra], ignore_index=True)
    df_out = df_out.drop_duplicates(
        subset=["query_id", "pos_doc_id", "pos_page_idx", "neg_doc_id", "neg_page_idx"]
    ).reset_index(drop=True)

    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"Saved optimized trainset to: {OUT_FILE}")
    print("base_rows =", len(df_base))
    print("extra_rows =", len(df_extra))
    print("total_rows =", len(df_out))
    print("\nPairs by neg_source:")
    print(df_out["neg_source"].value_counts())
    print("\nPairs by query_type:")
    print(df_out["query_type"].value_counts())


if __name__ == "__main__":
    main()
