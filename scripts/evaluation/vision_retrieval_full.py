from pathlib import Path
import json
import time
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_FILE = ROOT / "data" / "manifest.csv"
QUERY_FILE = ROOT / "data" / "eval" / "query_gold_v1.csv"

OUT_DIR = ROOT / "outputs"
CACHE_DIR = OUT_DIR / "vision_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DOC_EMB_FILE = CACHE_DIR / "colqwen2_doc_embeddings.pt"
DOC_META_FILE = CACHE_DIR / "colqwen2_doc_metadata.csv"

DETAIL_CSV = OUT_DIR / "vision_full_eval_details.csv"
SUMMARY_JSON = OUT_DIR / "vision_full_eval_summary.json"

MODEL_NAME = "vidore/colqwen2-v1.0"


def page_path(doc_id: str, page_idx: int) -> Path:
    return ROOT / "data" / "pages" / doc_id / f"page_{page_idx:04d}.png"


def batch_list(xs, bs):
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


def load_model():
    model = ColQwen2.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)
    return model, processor


def encode_all_pages(df_docs: pd.DataFrame, model, processor, image_bs: int = 1):
    doc_embeddings = []

    image_paths = df_docs["image_path"].tolist()

    for batch_paths in tqdm(list(batch_list(image_paths, image_bs)), desc="Encoding full corpus page images"):
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        batch = processor.process_images(batch_imgs)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            emb = model(**batch)

        doc_embeddings.extend(list(torch.unbind(emb.to("cpu"))))

        # 尽量及时释放
        del batch_imgs, batch, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return doc_embeddings


def encode_queries(queries, model, processor, query_bs: int = 4):
    query_embeddings = []

    for batch_q in tqdm(list(batch_list(queries, query_bs)), desc="Encoding queries"):
        batch = processor.process_queries(batch_q)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            emb = model(**batch)

        query_embeddings.extend(list(torch.unbind(emb.to("cpu"))))

        del batch, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return query_embeddings


def main():
    t0 = time.time()

    df_docs = pd.read_csv(MANIFEST_FILE).copy()
    df_docs = df_docs.sort_values(["doc_id", "page_idx"]).reset_index(drop=True)

    # 用 manifest 重新生成 image_path，避免后面路径列不一致
    df_docs["image_path"] = df_docs.apply(
        lambda r: str(page_path(r["doc_id"], int(r["page_idx"]))), axis=1
    )

    df_q = pd.read_csv(QUERY_FILE)
    df_q = df_q[df_q["label_status"] == "confirmed"].reset_index(drop=True)
    queries = df_q["query_en"].tolist()

    print(f"num_docs = {len(df_docs)}")
    print(f"num_queries = {len(df_q)}")

    model, processor = load_model()

    # ===== A. 全库页图编码（优先读缓存） =====
    if DOC_EMB_FILE.exists() and DOC_META_FILE.exists():
        print("\nFound cached doc embeddings, loading...")
        doc_embeddings = torch.load(DOC_EMB_FILE)
        df_docs = pd.read_csv(DOC_META_FILE)
        print(f"Loaded {len(doc_embeddings)} cached page embeddings.")
    else:
        print("\nNo cache found, encoding full corpus...")
        doc_embeddings = encode_all_pages(df_docs, model, processor, image_bs=1)

        torch.save(doc_embeddings, DOC_EMB_FILE)
        df_docs.to_csv(DOC_META_FILE, index=False, encoding="utf-8")
        print(f"Saved doc embeddings to: {DOC_EMB_FILE}")
        print(f"Saved doc metadata to: {DOC_META_FILE}")

    # ===== B. query 编码 =====
    query_embeddings = encode_queries(queries, model, processor, query_bs=4)

    # ===== C. 评测 =====
    print("\nScoring query-page pairs...")
    scores = processor.score_multi_vector(query_embeddings, doc_embeddings)  # [num_queries, num_docs]
    scores = scores.cpu()

    rows = []

    for qi, qrow in enumerate(df_q.itertuples(index=False)):
        q_scores = scores[qi]
        k = min(5, len(df_docs))
        topk_scores, topk_idx = torch.topk(q_scores, k=k)

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
            "top1_score": float(topk_scores[0]),

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

    summary = {
        "num_queries": int(len(df_res)),
        "num_docs": int(len(df_docs)),
        "top1_exact_acc": float(df_res["top1_exact_hit"].mean()) if len(df_res) else 0.0,
        "top5_exact_recall": float(df_res["top5_exact_hit"].mean()) if len(df_res) else 0.0,
        "top1_doc_acc": float(df_res["top1_doc_hit"].mean()) if len(df_res) else 0.0,
        "top5_doc_recall": float(df_res["top5_doc_hit"].mean()) if len(df_res) else 0.0,
        "elapsed_seconds": time.time() - t0,
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

    print("\nSaved:")
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
    main()
