from pathlib import Path
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor

ROOT = Path(__file__).resolve().parents[2]

POS_FILE = ROOT / "data" / "train" / "page_query_candidates_filtered.csv"
DOC_EMB_FILE = ROOT / "outputs" / "vision_cache" / "colqwen2_doc_embeddings.pt"
DOC_META_FILE = ROOT / "outputs" / "vision_cache" / "colqwen2_doc_metadata.csv"

OUT_FILE = ROOT / "data" / "train" / "vision_hard_negatives.csv"

MODEL_NAME = "vidore/colqwen2-v1.0"


def batch_list(xs, bs):
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


def main():
    df_pos = pd.read_csv(POS_FILE)
    df_docs = pd.read_csv(DOC_META_FILE)
    doc_embeddings = torch.load(DOC_EMB_FILE)

    print("num_positive_queries =", len(df_pos))
    print("num_doc_pages =", len(df_docs))

    model = ColQwen2.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)

    queries = df_pos["query_en"].tolist()

    query_embeddings = []
    for batch_q in tqdm(list(batch_list(queries, 8)), desc="Encoding queries for vision_hard"):
        batch = processor.process_queries(batch_q)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            emb = model(**batch)
        query_embeddings.extend(list(torch.unbind(emb.to("cpu"))))

    scores = processor.score_multi_vector(query_embeddings, doc_embeddings).cpu()

    rows = []

    for qi, row in enumerate(df_pos.itertuples(index=False)):
        gold_doc = row.doc_id
        gold_page = int(row.page_idx)

        q_scores = scores[qi]
        topk_scores, topk_idx = torch.topk(q_scores, k=10)

        neg_doc = None
        neg_page = None
        neg_score = None

        for idx, sc in zip(topk_idx.tolist(), topk_scores.tolist()):
            doc_id = df_docs.iloc[int(idx)]["doc_id"]
            page_idx = int(df_docs.iloc[int(idx)]["page_idx"])
            if not (doc_id == gold_doc and page_idx == gold_page):
                neg_doc = doc_id
                neg_page = page_idx
                neg_score = float(sc)
                break

        if neg_doc is not None:
            rows.append({
                "query_id": row.query_id,
                "query_en": row.query_en,
                "query_zh": row.query_zh,
                "query_type": row.page_type,
                "pos_doc_id": gold_doc,
                "pos_page_idx": gold_page,
                "neg_doc_id": neg_doc,
                "neg_page_idx": neg_page,
                "neg_source": "vision_hard",
                "neg_score": neg_score,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"Saved vision hard negatives to: {OUT_FILE}")
    print("num_rows =", len(df_out))
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
