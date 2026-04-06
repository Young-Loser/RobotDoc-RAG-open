from pathlib import Path
import json
import re
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]

POS_FILE = ROOT / "data" / "train" / "page_query_candidates_filtered.csv"
DOC_FILE = ROOT / "data" / "ocr" / "page_texts_clean.csv"

OUT_FILE = ROOT / "data" / "train" / "retriever_train_pairs_large.csv"

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def build_bm25(df_docs: pd.DataFrame):
    corpus = df_docs["ocr_text_clean"].fillna("").tolist()
    tokenized = [tokenize(x) for x in corpus]
    return BM25Okapi(tokenized)

def page_path(doc_id: str, page_idx: int) -> str:
    return str(ROOT / "data" / "pages" / doc_id / f"page_{page_idx:04d}.png")

def first_wrong_bm25(df_docs, bm25, query_en, gold_doc, gold_page, topk=10):
    q_tokens = tokenize(query_en)
    scores = bm25.get_scores(q_tokens)

    tmp = df_docs.copy()
    tmp["score"] = scores
    tmp = tmp.sort_values("score", ascending=False).head(topk)

    for row in tmp.itertuples(index=False):
        if not (row.doc_id == gold_doc and int(row.page_idx) == gold_page):
            return row.doc_id, int(row.page_idx), "bm25_hard"
    return None, None, None

def build_adjacent_negative(gold_doc, gold_page, doc_page_set):
    if (gold_doc, gold_page - 1) in doc_page_set and gold_page > 0:
        return gold_doc, gold_page - 1, "adjacent_prev"
    if (gold_doc, gold_page + 1) in doc_page_set:
        return gold_doc, gold_page + 1, "adjacent_next"
    return None, None, None

def build_same_doc_far_negative(gold_doc, gold_page, df_docs):
    sub = df_docs[df_docs["doc_id"] == gold_doc].copy()
    sub["dist"] = (sub["page_idx"] - gold_page).abs()
    sub = sub[sub["page_idx"] != gold_page].sort_values("dist", ascending=False)
    if len(sub) == 0:
        return None, None, None
    row = sub.iloc[0]
    return row["doc_id"], int(row["page_idx"]), "same_doc_far"

def main():
    df_pos = pd.read_csv(POS_FILE)
    df_docs = pd.read_csv(DOC_FILE)
    df_docs = df_docs[df_docs["ocr_text_clean"].fillna("").str.len() > 0].copy().reset_index(drop=True)

    bm25 = build_bm25(df_docs)
    doc_page_set = set(zip(df_docs["doc_id"], df_docs["page_idx"].astype(int)))

    rows = []

    for row in df_pos.itertuples(index=False):
        query_id = row.query_id
        query_zh = row.query_zh
        query_en = row.query_en
        query_type = row.page_type

        gold_doc = row.doc_id
        gold_page = int(row.page_idx)
        pos_path = page_path(gold_doc, gold_page)

        # 1) BM25 hard negative
        neg_doc, neg_page, neg_source = first_wrong_bm25(
            df_docs, bm25, query_en, gold_doc, gold_page, topk=10
        )
        if neg_doc is not None:
            rows.append({
                "sample_id": f"{query_id}_bm25",
                "query_id": query_id,
                "query_en": query_en,
                "query_zh": query_zh,
                "query_type": query_type,
                "pos_doc_id": gold_doc,
                "pos_page_idx": gold_page,
                "pos_image_path": pos_path,
                "neg_doc_id": neg_doc,
                "neg_page_idx": neg_page,
                "neg_image_path": page_path(neg_doc, neg_page),
                "neg_source": neg_source,
            })

        # 2) 相邻页 hard negative
        neg_doc, neg_page, neg_source = build_adjacent_negative(
            gold_doc, gold_page, doc_page_set
        )
        if neg_doc is not None:
            rows.append({
                "sample_id": f"{query_id}_{neg_source}",
                "query_id": query_id,
                "query_en": query_en,
                "query_zh": query_zh,
                "query_type": query_type,
                "pos_doc_id": gold_doc,
                "pos_page_idx": gold_page,
                "pos_image_path": pos_path,
                "neg_doc_id": neg_doc,
                "neg_page_idx": neg_page,
                "neg_image_path": page_path(neg_doc, neg_page),
                "neg_source": neg_source,
            })

        # 3) 同文档远页 negative
        neg_doc, neg_page, neg_source = build_same_doc_far_negative(
            gold_doc, gold_page, df_docs
        )
        if neg_doc is not None:
            rows.append({
                "sample_id": f"{query_id}_{neg_source}",
                "query_id": query_id,
                "query_en": query_en,
                "query_zh": query_zh,
                "query_type": query_type,
                "pos_doc_id": gold_doc,
                "pos_page_idx": gold_page,
                "pos_image_path": pos_path,
                "neg_doc_id": neg_doc,
                "neg_page_idx": neg_page,
                "neg_image_path": page_path(neg_doc, neg_page),
                "neg_source": neg_source,
            })

    df_out = pd.DataFrame(rows).drop_duplicates(
        subset=["query_id", "pos_doc_id", "pos_page_idx", "neg_doc_id", "neg_page_idx"]
    ).reset_index(drop=True)

    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"Saved large training pairs to: {OUT_FILE}")
    print("num_positive_queries:", df_pos['query_id'].nunique())
    print("num_pairs:", len(df_out))

    print("\nPairs by neg_source:")
    print(df_out["neg_source"].value_counts())

    print("\nPairs by query_type:")
    print(df_out["query_type"].value_counts())

    print("\nSample rows:")
    print(df_out.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
