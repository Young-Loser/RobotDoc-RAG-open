from pathlib import Path
import re
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "ocr" / "page_texts_clean.csv"

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    text = text.lower()
    # 简单英文分词：保留字母数字
    return re.findall(r"[a-zA-Z0-9]+", text)

def build_bm25(df: pd.DataFrame):
    corpus = df["ocr_text_clean"].fillna("").tolist()
    tokenized_corpus = [tokenize(x) for x in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

def search(df: pd.DataFrame, bm25: BM25Okapi, query: str, topk: int = 5):
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)

    tmp = df.copy()
    tmp["score"] = scores
    tmp = tmp.sort_values("score", ascending=False).head(topk)

    return tmp[["doc_id", "page_idx", "score", "ocr_text_clean"]]

def main():
    df = pd.read_csv(DATA_FILE)

    # 只保留有清洗文本的页面
    df = df[df["ocr_text_clean"].fillna("").str.len() > 0].reset_index(drop=True)

    bm25, _ = build_bm25(df)

    print("BM25 ready.")
    print("num_pages =", len(df))
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Query: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        topk = search(df, bm25, query, topk=5)

        print("\nTop-5 results:")
        for i, row in enumerate(topk.itertuples(index=False), start=1):
            preview = str(row.ocr_text_clean)[:300].replace("\n", " | ")
            print(f"\n[{i}] doc={row.doc_id} page={row.page_idx} score={row.score:.4f}")
            print(preview)
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
