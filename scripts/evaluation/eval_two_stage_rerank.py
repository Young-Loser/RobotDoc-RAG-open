from pathlib import Path
import json
import re
import time
import random
import sys
import argparse

import torch
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from robotdoc_rag.retriever import load_retriever_from_checkpoint
from robotdoc_rag.evaluation import (
    build_failure_table,
    cache_is_compatible,
    summarize_retrieval_results,
)

DOC_TEXT_FILE = ROOT / "data" / "ocr" / "page_texts_clean.csv"
QUERY_FILE = ROOT / "data" / "eval" / "query_gold_v1.csv"
CKPT_FILE = ROOT / "outputs" / "checkpoints" / "siglip_retriever_v1_best.pt"

SIGLIP_CACHE_DIR = ROOT / "outputs" / "siglip_cache"
DOC_EMB_FILE = SIGLIP_CACHE_DIR / "siglip_retriever_v1_doc_embeddings.pt"
DOC_META_FILE = SIGLIP_CACHE_DIR / "siglip_retriever_v1_doc_metadata.csv"
DOC_CACHE_INFO_FILE = SIGLIP_CACHE_DIR / "siglip_retriever_v1_doc_cache_info.json"

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_CSV = OUT_DIR / "two_stage_rerank_eval_details.csv"
SUMMARY_JSON = OUT_DIR / "two_stage_rerank_eval_summary.json"
FAILURE_CSV = OUT_DIR / "reports" / "two_stage_rerank_failures.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BM25 + SigLIP two-stage reranking.")
    parser.add_argument("--topk-bm25", type=int, default=20, help="BM25 candidate pool size.")
    parser.add_argument("--topk-final", type=int, default=5, help="Final reranked top-k size.")
    return parser


def load_siglip_model():
    ckpt = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=False)
    model = load_retriever_from_checkpoint(CKPT_FILE, device=DEVICE)
    print("Loaded checkpoint:", CKPT_FILE)
    print("Best val loss:", ckpt.get("best_val_loss", None))
    print("Pooling:", ckpt.get("pooling", "cls"))
    return model


def build_bm25(df_docs: pd.DataFrame):
    corpus = df_docs["ocr_text_clean"].fillna("").tolist()
    tokenized = [tokenize(x) for x in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25


def retrieve_bm25_topk(df_docs: pd.DataFrame, bm25: BM25Okapi, query: str, topk: int = 20):
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)

    tmp = df_docs.copy()
    tmp["bm25_score"] = scores
    tmp = tmp.sort_values("bm25_score", ascending=False).head(topk).reset_index(drop=True)
    return tmp


def main():
    args = build_parser().parse_args()
    set_seed(SEED)
    t0 = time.time()

    # 1) 文本库（BM25 用）
    df_docs_text = pd.read_csv(DOC_TEXT_FILE)
    df_docs_text = df_docs_text[df_docs_text["ocr_text_clean"].fillna("").str.len() > 0].copy()
    df_docs_text = df_docs_text.sort_values(["doc_id", "page_idx"]).reset_index(drop=True)

    # 2) query gold
    df_q = pd.read_csv(QUERY_FILE)
    df_q = df_q[df_q["label_status"] == "confirmed"].reset_index(drop=True)

    # 3) 训练后视觉 embedding cache
    assert DOC_EMB_FILE.exists(), f"Missing: {DOC_EMB_FILE}"
    assert DOC_META_FILE.exists(), f"Missing: {DOC_META_FILE}"
    assert DOC_CACHE_INFO_FILE.exists(), f"Missing: {DOC_CACHE_INFO_FILE}. Please rerun eval_siglip_retriever_v1.py first."

    ckpt = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=False)
    pooling = ckpt.get("pooling", "cls")
    text_pooling = ckpt.get("text_pooling")
    image_pooling = ckpt.get("image_pooling")
    if not cache_is_compatible(
        DOC_CACHE_INFO_FILE,
        checkpoint_path=CKPT_FILE,
        model_pooling=pooling,
        num_docs=len(df_docs_text),
        text_pooling=text_pooling,
        image_pooling=image_pooling,
    ):
        raise RuntimeError("SigLIP v1 cache is stale for the current checkpoint. Re-run eval_siglip_retriever_v1.py with --force-reencode.")

    doc_embs = torch.load(DOC_EMB_FILE, map_location="cpu", weights_only=False)
    df_docs_vis = pd.read_csv(DOC_META_FILE)

    # 构造 (doc_id, page_idx) -> embedding row index
    vis_index = {}
    for i, row in df_docs_vis.iterrows():
        vis_index[(row["doc_id"], int(row["page_idx"]))] = i

    # BM25
    bm25 = build_bm25(df_docs_text)

    # SigLIP v1 query encoder
    model = load_siglip_model()

    print("num_docs_text =", len(df_docs_text))
    print("num_docs_vis  =", len(df_docs_vis))
    print("num_queries   =", len(df_q))

    rows = []

    for q in df_q.itertuples(index=False):
        # ===== stage 1: BM25 recall =====
        topk_bm25 = retrieve_bm25_topk(df_docs_text, bm25, q.query_en, topk=args.topk_bm25)

        # ===== stage 2: SigLIP rerank =====
        with torch.no_grad():
            query_emb = model.encode_text([q.query_en]).cpu()[0]  # [dim]

        candidate_indices = []
        candidate_doc_ids = []
        candidate_page_idxs = []
        candidate_bm25_scores = []

        for row in topk_bm25.itertuples(index=False):
            key = (row.doc_id, int(row.page_idx))
            if key not in vis_index:
                continue
            candidate_indices.append(vis_index[key])
            candidate_doc_ids.append(row.doc_id)
            candidate_page_idxs.append(int(row.page_idx))
            candidate_bm25_scores.append(float(row.bm25_score))

        cand_embs = doc_embs[candidate_indices]  # [k, dim]
        rerank_scores = (cand_embs @ query_emb).detach()    # cosine，因为已经 normalize

        # 取 rerank 后前 5
        rerank_order = torch.argsort(rerank_scores, descending=True)
        top5_order = rerank_order[: args.topk_final].tolist()

        gold_doc = q.final_doc_id
        gold_page = int(q.final_page_idx)

        top1_idx_local = top5_order[0]
        top1_doc = candidate_doc_ids[top1_idx_local]
        top1_page = candidate_page_idxs[top1_idx_local]

        top1_doc_hit = (top1_doc == gold_doc)
        top1_exact_hit = (top1_doc == gold_doc and top1_page == gold_page)

        top5_doc_hit = False
        top5_exact_hit = False

        retrieved_doc_ids = []
        retrieved_page_idxs = []
        retrieved_scores = []

        for idx_local in top5_order:
            doc_id = candidate_doc_ids[idx_local]
            page_idx = candidate_page_idxs[idx_local]
            score = float(rerank_scores[idx_local])

            retrieved_doc_ids.append(doc_id)
            retrieved_page_idxs.append(page_idx)
            retrieved_scores.append(score)

            if doc_id == gold_doc:
                top5_doc_hit = True
            if doc_id == gold_doc and page_idx == gold_page:
                top5_exact_hit = True

        rows.append({
            "query_id": q.query_id,
            "query_zh": q.query_zh,
            "query_en": q.query_en,
            "query_type": q.query_type,
            "gold_doc_id": gold_doc,
            "gold_page_idx": gold_page,

            "top1_doc_id": top1_doc,
            "top1_page_idx": top1_page,
            "top1_score": float(rerank_scores[top1_idx_local]),

            "top1_doc_hit": top1_doc_hit,
            "top1_exact_hit": top1_exact_hit,
            "top5_doc_hit": top5_doc_hit,
            "top5_exact_hit": top5_exact_hit,

            "retrieved_doc_ids": json.dumps(retrieved_doc_ids, ensure_ascii=False),
            "retrieved_page_idxs": json.dumps(retrieved_page_idxs, ensure_ascii=False),
            "retrieved_scores": json.dumps(retrieved_scores, ensure_ascii=False),
            "candidate_pool_size": len(candidate_indices),
        })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(DETAIL_CSV, index=False, encoding="utf-8")
    FAILURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    build_failure_table(df_res).to_csv(FAILURE_CSV, index=False, encoding="utf-8")

    summary = summarize_retrieval_results(
        df_res,
        extra={
            "avg_candidate_pool_size": float(df_res["candidate_pool_size"].mean()) if len(df_res) else 0.0,
            "elapsed_seconds": time.time() - t0,
            "pooling": pooling,
            "text_pooling": text_pooling or pooling,
            "image_pooling": image_pooling or pooling,
            "topk_bm25": args.topk_bm25,
            "topk_final": args.topk_final,
        },
    )

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(" -", DETAIL_CSV)
    print(" -", SUMMARY_JSON)
    print(" -", FAILURE_CSV)

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
    main()
