from pathlib import Path
import re
import json
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
DOC_FILE = ROOT / "data" / "ocr" / "page_texts_clean.csv"
QUERY_FILE = ROOT / "data" / "eval" / "query_gold_v1.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_CSV = OUT_DIR / "bm25_eval_details.csv"
SUMMARY_JSON = OUT_DIR / "bm25_eval_summary.json"

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def build_bm25(df_docs: pd.DataFrame):
    corpus = df_docs["ocr_text_clean"].fillna("").tolist()
    tokenized = [tokenize(x) for x in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25

def retrieve_topk(df_docs: pd.DataFrame, bm25: BM25Okapi, query: str, topk: int = 5):
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)

    tmp = df_docs.copy()
    tmp["score"] = scores
    tmp = tmp.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    return tmp

def evaluate():
    df_docs = pd.read_csv(DOC_FILE)
    df_docs = df_docs[df_docs["ocr_text_clean"].fillna("").str.len() > 0].reset_index(drop=True)

    df_q = pd.read_csv(QUERY_FILE)
    df_q = df_q[df_q["label_status"] == "confirmed"].reset_index(drop=True)

    bm25 = build_bm25(df_docs)

    rows = []

    for q in df_q.itertuples(index=False):
        topk = retrieve_topk(df_docs, bm25, q.query_en, topk=5)

        gold_doc = q.final_doc_id
        gold_page = int(q.final_page_idx)

        top1 = topk.iloc[0]
        top1_doc_hit = (top1["doc_id"] == gold_doc)
        top1_exact_hit = (top1["doc_id"] == gold_doc and int(top1["page_idx"]) == gold_page)

        top5_doc_hit = False
        top5_exact_hit = False

        retrieved_docs = []
        retrieved_pages = []
        retrieved_scores = []

        for row in topk.itertuples(index=False):
            retrieved_docs.append(row.doc_id)
            retrieved_pages.append(int(row.page_idx))
            retrieved_scores.append(float(row.score))

            if row.doc_id == gold_doc:
                top5_doc_hit = True
            if row.doc_id == gold_doc and int(row.page_idx) == gold_page:
                top5_exact_hit = True

        rows.append({
            "query_id": q.query_id,
            "query_zh": q.query_zh,
            "query_en": q.query_en,
            "query_type": q.query_type,
            "gold_doc_id": gold_doc,
            "gold_page_idx": gold_page,

            "top1_doc_id": top1["doc_id"],
            "top1_page_idx": int(top1["page_idx"]),
            "top1_score": float(top1["score"]),

            "top1_doc_hit": top1_doc_hit,
            "top1_exact_hit": top1_exact_hit,
            "top5_doc_hit": top5_doc_hit,
            "top5_exact_hit": top5_exact_hit,

            "retrieved_doc_ids": json.dumps(retrieved_docs, ensure_ascii=False),
            "retrieved_page_idxs": json.dumps(retrieved_pages, ensure_ascii=False),
            "retrieved_scores": json.dumps(retrieved_scores, ensure_ascii=False),
        })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(DETAIL_CSV, index=False, encoding="utf-8")

    summary = {
        "num_queries": int(len(df_res)),
        "top1_exact_acc": float(df_res["top1_exact_hit"].mean()) if len(df_res) else 0.0,
        "top5_exact_recall": float(df_res["top5_exact_hit"].mean()) if len(df_res) else 0.0,
        "top1_doc_acc": float(df_res["top1_doc_hit"].mean()) if len(df_res) else 0.0,
        "top5_doc_recall": float(df_res["top5_doc_hit"].mean()) if len(df_res) else 0.0,
    }

    by_type = []
    for qtype, g in df_res.groupby("query_type"):
        by_type.append({
            "query_type": qtype,
            "num_queries": int(len(g)),
            "top1_exact_acc": float(g["top1_exact_hit"].mean()),
            "top5_exact_recall": float(g["top5_exact_hit"].mean()),
            "top1_doc_acc": float(g["top1_doc_hit"].mean()),
            "top5_doc_recall": float(g["top5_doc_hit"].mean()),
        })

    summary["by_query_type"] = by_type

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(" -", DETAIL_CSV)
    print(" -", SUMMARY_JSON)

    print("\n=== Overall ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n=== Per-query ===")
    print(df_res[[
        "query_id", "query_type", "query_en",
        "gold_doc_id", "gold_page_idx",
        "top1_doc_id", "top1_page_idx",
        "top1_doc_hit", "top1_exact_hit", "top5_doc_hit", "top5_exact_hit"
    ]].to_string(index=False))

if __name__ == "__main__":
    evaluate()
