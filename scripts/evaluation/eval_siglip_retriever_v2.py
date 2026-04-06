from pathlib import Path
import json
import time
import random
import sys
import argparse

import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from robotdoc_rag.retriever import load_retriever_from_checkpoint
from robotdoc_rag.evaluation import (
    build_failure_table,
    cache_is_compatible,
    encode_all_pages,
    encode_queries,
    summarize_retrieval_results,
    write_cache_metadata,
)

MANIFEST_FILE = ROOT / "data" / "manifest.csv"
QUERY_FILE = ROOT / "data" / "eval" / "query_gold_v1.csv"
CKPT_FILE = ROOT / "outputs" / "checkpoints" / "siglip_retriever_v2_best.pt"

OUT_DIR = ROOT / "outputs"
CACHE_DIR = OUT_DIR / "siglip_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DOC_EMB_FILE = CACHE_DIR / "siglip_retriever_v2_doc_embeddings.pt"
DOC_META_FILE = CACHE_DIR / "siglip_retriever_v2_doc_metadata.csv"
DOC_CACHE_INFO_FILE = CACHE_DIR / "siglip_retriever_v2_doc_cache_info.json"

DETAIL_CSV = OUT_DIR / "siglip_retriever_v2_eval_details.csv"
SUMMARY_JSON = OUT_DIR / "siglip_retriever_v2_eval_summary.json"
FAILURE_CSV = OUT_DIR / "reports" / "siglip_retriever_v2_failures.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the SigLIP retriever v2.")
    parser.add_argument("--force-reencode", action="store_true", help="Ignore cached document embeddings.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k retrieval size.")
    parser.add_argument("--checkpoint-file", type=Path, default=CKPT_FILE, help="Checkpoint to evaluate.")
    return parser


def page_path(doc_id: str, page_idx: int) -> str:
    return str(ROOT / "data" / "pages" / doc_id / f"page_{page_idx:04d}.png")

def load_model(checkpoint_file: Path):
    ckpt = torch.load(checkpoint_file, map_location=DEVICE, weights_only=False)
    model = load_retriever_from_checkpoint(checkpoint_file, device=DEVICE)
    print("Loaded checkpoint:", checkpoint_file)
    print("Best val loss:", ckpt.get("best_val_loss", None))
    print("Pooling:", ckpt.get("pooling", "cls"))
    return model

def main():
    args = build_parser().parse_args()
    set_seed(SEED)
    t0 = time.time()

    df_docs = pd.read_csv(MANIFEST_FILE).copy()
    df_docs = df_docs.sort_values(["doc_id", "page_idx"]).reset_index(drop=True)
    df_docs["image_path"] = df_docs.apply(
        lambda r: page_path(r["doc_id"], int(r["page_idx"])), axis=1
    )

    df_q = pd.read_csv(QUERY_FILE)
    df_q = df_q[df_q["label_status"] == "confirmed"].reset_index(drop=True)

    print("num_docs =", len(df_docs))
    print("num_queries =", len(df_q))

    model = load_model(args.checkpoint_file)
    ckpt = torch.load(args.checkpoint_file, map_location=DEVICE, weights_only=False)
    pooling = ckpt.get("pooling", "cls")
    text_pooling = ckpt.get("text_pooling")
    image_pooling = ckpt.get("image_pooling")

    use_cache = (
        not args.force_reencode
        and DOC_EMB_FILE.exists()
        and DOC_META_FILE.exists()
        and cache_is_compatible(
            DOC_CACHE_INFO_FILE,
            checkpoint_path=args.checkpoint_file,
            model_pooling=pooling,
            num_docs=len(df_docs),
            text_pooling=text_pooling,
            image_pooling=image_pooling,
        )
    )

    if use_cache:
        print("\nFound cached doc embeddings, loading...")
        doc_embs = torch.load(DOC_EMB_FILE, map_location="cpu", weights_only=False)
        df_docs = pd.read_csv(DOC_META_FILE)
        print("Loaded cached embeddings:", doc_embs.shape)
    else:
        print("\nEncoding full corpus for current checkpoint...")
        doc_embs = encode_all_pages(df_docs, model, image_bs=16)
        torch.save(doc_embs, DOC_EMB_FILE)
        df_docs.to_csv(DOC_META_FILE, index=False, encoding="utf-8")
        write_cache_metadata(
            DOC_CACHE_INFO_FILE,
            checkpoint_path=args.checkpoint_file,
            model_pooling=pooling,
            num_docs=len(df_docs),
            text_pooling=text_pooling,
            image_pooling=image_pooling,
        )
        print("Saved doc embeddings to:", DOC_EMB_FILE)
        print("Saved doc metadata to:", DOC_META_FILE)
        print("Saved cache metadata to:", DOC_CACHE_INFO_FILE)

    query_embs = encode_queries(df_q["query_en"].tolist(), model, query_bs=16).detach()

    print("\nScoring query-page similarities...")
    scores = (query_embs @ doc_embs.t()).detach()

    rows = []

    for qi, qrow in enumerate(df_q.itertuples(index=False)):
        q_scores = scores[qi]
        topk_scores, topk_idx = torch.topk(q_scores, k=min(args.topk, len(df_docs)))

        gold_doc = qrow.final_doc_id
        gold_page = int(qrow.final_page_idx)

        top1_idx = int(topk_idx[0])
        top1_doc = df_docs.iloc[top1_idx]["doc_id"]
        top1_page = int(df_docs.iloc[top1_idx]["page_idx"])

        top1_doc_hit = (top1_doc == gold_doc)
        top1_exact_hit = (top1_doc == gold_doc and top1_page == gold_page)

        top5_doc_hit = False
        top5_exact_hit = False

        retrieved_doc_ids = []
        retrieved_page_idxs = []
        retrieved_scores = []

        for idx, sc in zip(topk_idx.tolist(), topk_scores.tolist()):
            doc_id = df_docs.iloc[int(idx)]["doc_id"]
            page_idx = int(df_docs.iloc[int(idx)]["page_idx"])

            retrieved_doc_ids.append(doc_id)
            retrieved_page_idxs.append(page_idx)
            retrieved_scores.append(float(sc))

            if doc_id == gold_doc:
                top5_doc_hit = True
            if doc_id == gold_doc and page_idx == gold_page:
                top5_exact_hit = True

        rows.append({
            "query_id": qrow.query_id,
            "query_zh": qrow.query_zh,
            "query_en": qrow.query_en,
            "query_type": qrow.query_type,
            "gold_doc_id": gold_doc,
            "gold_page_idx": gold_page,

            "top1_doc_id": top1_doc,
            "top1_page_idx": top1_page,
            "top1_score": float(topk_scores[0].detach()),

            "top1_doc_hit": top1_doc_hit,
            "top1_exact_hit": top1_exact_hit,
            "top5_doc_hit": top5_doc_hit,
            "top5_exact_hit": top5_exact_hit,

            "retrieved_doc_ids": json.dumps(retrieved_doc_ids, ensure_ascii=False),
            "retrieved_page_idxs": json.dumps(retrieved_page_idxs, ensure_ascii=False),
            "retrieved_scores": json.dumps(retrieved_scores, ensure_ascii=False),
        })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(DETAIL_CSV, index=False, encoding="utf-8")
    FAILURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    build_failure_table(df_res).to_csv(FAILURE_CSV, index=False, encoding="utf-8")

    summary = summarize_retrieval_results(
        df_res,
        extra={
            "num_docs": int(len(df_docs)),
            "elapsed_seconds": time.time() - t0,
            "cache_used": use_cache,
            "pooling": pooling,
            "text_pooling": text_pooling or pooling,
            "image_pooling": image_pooling or pooling,
            "checkpoint_file": str(args.checkpoint_file),
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
