from pathlib import Path
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor

ROOT = Path(__file__).resolve().parents[2]
QUERY_FILE = ROOT / "data" / "eval" / "query_gold_v1.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_CSV = OUT_DIR / "vision_smoke_details.csv"
SUMMARY_JSON = OUT_DIR / "vision_smoke_summary.json"


def page_path(doc_id: str, page_idx: int) -> Path:
    return ROOT / "data" / "pages" / doc_id / f"page_{page_idx:04d}.png"


def batch_list(xs, bs):
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


def main():
    df_q = pd.read_csv(QUERY_FILE)
    df_q = df_q[df_q["label_status"] == "confirmed"].reset_index(drop=True)

    # 只用已确认 gold 页构成一个小型视觉检索语料库（去重）
    df_corpus = (
        df_q[["final_doc_id", "final_page_idx"]]
        .drop_duplicates()
        .rename(columns={"final_doc_id": "doc_id", "final_page_idx": "page_idx"})
        .reset_index(drop=True)
    )
    df_corpus["image_path"] = df_corpus.apply(
        lambda r: str(page_path(r["doc_id"], int(r["page_idx"]))), axis=1
    )

    queries = df_q["query_en"].tolist()
    images = [Image.open(p).convert("RGB") for p in df_corpus["image_path"].tolist()]

    print(f"num_queries = {len(queries)}")
    print(f"num_unique_pages_in_smoke_corpus = {len(images)}")

    model_name = "vidore/colqwen2-v1.0"
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()

    processor = ColQwen2Processor.from_pretrained(model_name)

    # 1) encode page images
    doc_embeddings = []
    image_bs = 2

    for batch_imgs in tqdm(list(batch_list(images, image_bs)), desc="Encoding page images"):
        batch = processor.process_images(batch_imgs)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            emb = model(**batch)
        doc_embeddings.extend(list(torch.unbind(emb.to("cpu"))))

    # 2) encode queries
    query_embeddings = []
    query_bs = 4

    for batch_q in tqdm(list(batch_list(queries, query_bs)), desc="Encoding queries"):
        batch = processor.process_queries(batch_q)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            emb = model(**batch)
        query_embeddings.extend(list(torch.unbind(emb.to("cpu"))))

    # 3) score query-page pairs
    scores = processor.score_multi_vector(query_embeddings, doc_embeddings)  # [num_queries, num_pages]
    scores = scores.cpu()

    rows = []

    for qi, qrow in enumerate(df_q.itertuples(index=False)):
        q_scores = scores[qi]
        k = min(5, len(df_corpus))
        topk_scores, topk_idx = torch.topk(q_scores, k=k)

        gold_doc = qrow.final_doc_id
        gold_page = int(qrow.final_page_idx)

        top1_idx = int(topk_idx[0])
        top1_doc = df_corpus.iloc[top1_idx]["doc_id"]
        top1_page = int(df_corpus.iloc[top1_idx]["page_idx"])

        top1_doc_hit = (top1_doc == gold_doc)
        top1_exact_hit = (top1_doc == gold_doc and top1_page == gold_page)

        top5_doc_hit = False
        top5_exact_hit = False

        retrieved_doc_ids = []
        retrieved_page_idxs = []
        retrieved_scores = []

        for idx, sc in zip(topk_idx.tolist(), topk_scores.tolist()):
            doc_id = df_corpus.iloc[int(idx)]["doc_id"]
            page_idx = int(df_corpus.iloc[int(idx)]["page_idx"])

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
        "num_pages_in_smoke_corpus": int(len(df_corpus)),
        "top1_exact_acc": float(df_res["top1_exact_hit"].mean()) if len(df_res) else 0.0,
        "top5_exact_recall": float(df_res["top5_exact_hit"].mean()) if len(df_res) else 0.0,
        "top1_doc_acc": float(df_res["top1_doc_hit"].mean()) if len(df_res) else 0.0,
        "top5_doc_recall": float(df_res["top5_doc_hit"].mean()) if len(df_res) else 0.0,
    }

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
