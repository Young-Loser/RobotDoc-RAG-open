from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

QUERY_FILE = ROOT / "data" / "eval" / "query_gold_v1.csv"
BM25_FILE = ROOT / "outputs" / "bm25_eval_details.csv"
VISION_FILE = ROOT / "outputs" / "vision_full_eval_details.csv"

OUT_FILE = ROOT / "data" / "train" / "retriever_train_pairs.csv"


def page_path(doc_id: str, page_idx: int) -> str:
    return str(ROOT / "data" / "pages" / doc_id / f"page_{page_idx:04d}.png")


def parse_list_field(x):
    if isinstance(x, str) and x.strip():
        return json.loads(x)
    return []


def first_wrong_from_retrieval(row, gold_doc, gold_page):
    docs = parse_list_field(row["retrieved_doc_ids"])
    pages = parse_list_field(row["retrieved_page_idxs"])

    for d, p in zip(docs, pages):
        if not (d == gold_doc and int(p) == gold_page):
            return d, int(p)
    return None, None


def build_same_doc_hard_negative(gold_doc, gold_page):
    # 相邻页常常最容易混淆，是很好的 hard negative
    if gold_page > 0:
        return gold_doc, gold_page - 1, "adjacent_prev"
    return gold_doc, gold_page + 1, "adjacent_next"


def main():
    df_q = pd.read_csv(QUERY_FILE)
    df_q = df_q[df_q["label_status"] == "confirmed"].reset_index(drop=True)

    df_bm25 = pd.read_csv(BM25_FILE)
    df_vis = pd.read_csv(VISION_FILE)

    bm25_map = {r["query_id"]: r for _, r in df_bm25.iterrows()}
    vis_map = {r["query_id"]: r for _, r in df_vis.iterrows()}

    rows = []

    for _, q in df_q.iterrows():
        query_id = q["query_id"]
        query_en = q["query_en"]
        query_zh = q["query_zh"]
        query_type = q["query_type"]

        gold_doc = q["final_doc_id"]
        gold_page = int(q["final_page_idx"])

        pos_path = page_path(gold_doc, gold_page)

        # 1) BM25 hard negative
        if query_id in bm25_map:
            neg_doc, neg_page = first_wrong_from_retrieval(
                bm25_map[query_id], gold_doc, gold_page
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
                    "neg_source": "bm25_hard",
                })

        # 2) Vision hard negative
        if query_id in vis_map:
            neg_doc, neg_page = first_wrong_from_retrieval(
                vis_map[query_id], gold_doc, gold_page
            )
            if neg_doc is not None:
                rows.append({
                    "sample_id": f"{query_id}_vision",
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
                    "neg_source": "vision_hard",
                })

        # 3) Same-document adjacent hard negative
        neg_doc, neg_page, neg_source = build_same_doc_hard_negative(gold_doc, gold_page)
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

    print(f"Saved training pairs to: {OUT_FILE}")
    print(f"num_queries = {df_q['query_id'].nunique()}")
    print(f"num_pairs = {len(df_out)}")
    print("\nPairs by neg_source:")
    print(df_out["neg_source"].value_counts())

    print("\nSample rows:")
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
