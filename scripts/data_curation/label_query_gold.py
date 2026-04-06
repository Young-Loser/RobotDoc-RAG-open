from pathlib import Path
import re
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
DOC_FILE = ROOT / "data" / "ocr" / "page_texts_clean.csv"
QUERY_FILE = ROOT / "data" / "eval" / "query_starter.csv"
OUT_FILE = ROOT / "data" / "eval" / "query_gold.csv"

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def build_bm25(df: pd.DataFrame):
    corpus = df["ocr_text_clean"].fillna("").tolist()
    tokenized = [tokenize(x) for x in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25

def main():
    df_docs = pd.read_csv(DOC_FILE)
    df_docs = df_docs[df_docs["ocr_text_clean"].fillna("").str.len() > 0].reset_index(drop=True)

    df_queries = pd.read_csv(QUERY_FILE)

    bm25 = build_bm25(df_docs)

    rows = []

    print(f"Loaded {len(df_docs)} pages and {len(df_queries)} starter queries.\n")
    print("For each query:")
    print("  - inspect the top candidates")
    print("  - open the suggested page image if needed")
    print("  - type the gold doc_id and gold page_idx")
    print("  - if top candidate rank is correct, you can directly type: rank=1 / rank=2 / ...\n")

    for q in df_queries.itertuples(index=False):
        query = q.query
        q_tokens = tokenize(query)
        scores = bm25.get_scores(q_tokens)

        tmp = df_docs.copy()
        tmp["score"] = scores
        tmp = tmp.sort_values("score", ascending=False).head(5).reset_index(drop=True)

        print("\n" + "=" * 100)
        print(f"Query ID   : {q.query_id}")
        print(f"Query      : {q.query}")
        print(f"Type       : {q.query_type}")
        print(f"Doc hint   : {q.target_doc_hint}")
        print(f"Note       : {q.note}")
        print("-" * 100)

        for i, row in enumerate(tmp.itertuples(index=False), start=1):
            preview = str(row.ocr_text_clean)[:260].replace("\n", " | ")
            print(f"[{i}] doc={row.doc_id} page={row.page_idx} score={row.score:.4f}")
            print(f"    img={row.image_path}")
            print(f"    txt={preview}")
            print()

        print("How to label:")
        print("  1) If candidate rank 1 is correct -> input: rank=1")
        print("  2) If candidate rank 3 is correct -> input: rank=3")
        print("  3) If none is correct -> input like: doc_id,page_idx")
        print("  4) If you want to skip temporarily -> input: skip")

        user_in = input("Your label: ").strip()

        gold_doc_id = ""
        gold_page_idx = -1
        label_source = ""
        note = str(q.note)

        if user_in.lower() == "skip":
            label_source = "skip"
        elif user_in.startswith("rank="):
            rank = int(user_in.split("=")[1])
            chosen = tmp.iloc[rank - 1]
            gold_doc_id = chosen["doc_id"]
            gold_page_idx = int(chosen["page_idx"])
            label_source = f"bm25_top{rank}"
        else:
            try:
                doc_id, page_idx = [x.strip() for x in user_in.split(",")]
                gold_doc_id = doc_id
                gold_page_idx = int(page_idx)
                label_source = "manual"
            except Exception:
                print("Invalid input, mark as skip.")
                label_source = "skip"

        rows.append(
            {
                "query_id": q.query_id,
                "query": q.query,
                "query_type": q.query_type,
                "target_doc_hint": q.target_doc_hint,
                "gold_doc_id": gold_doc_id,
                "gold_page_idx": gold_page_idx,
                "label_source": label_source,
                "note": note,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved labels to: {OUT_FILE}")
    print(out_df)

if __name__ == "__main__":
    main()
