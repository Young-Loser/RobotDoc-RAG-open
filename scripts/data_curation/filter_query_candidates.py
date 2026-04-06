from pathlib import Path
import re
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "train" / "page_query_candidates.csv"
DOC_FILE = ROOT / "data" / "ocr" / "page_texts_clean.csv"
OUT_FILE = ROOT / "data" / "train" / "page_query_candidates_filtered.csv"

BAD_HEADING_TERMS = [
    "table of contents",
    "contents",
    "about this manual",
    "getting started",
    "index",
    "copyright",
    "franka robotics",
    "datasheet",
]

BAD_QUERY_PATTERNS_ZH = [
    "哪一页包含这一主题的说明内容",
]

BAD_QUERY_PATTERNS_EN = [
    "which page contains information about this topic",
]

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def build_bm25(df_docs: pd.DataFrame):
    corpus = df_docs["ocr_text_clean"].fillna("").tolist()
    tokenized = [tokenize(x) for x in corpus]
    return BM25Okapi(tokenized)

def keep_row(row):
    qzh = str(row["query_zh"]).strip().lower()
    qen = str(row["query_en"]).strip().lower()
    source = str(row["query_source"]).strip().lower()
    heading = str(row["anchor_heading"]).strip().lower()
    page_type = str(row["page_type"]).strip().lower()

    # 1. 删明显低价值 heading 型 query
    if any(x in heading for x in BAD_HEADING_TERMS):
        if source in {"layout_heading", "fact_heading", "generic_heading"}:
            return False

    # 2. generic 弱 query 直接去掉
    if source in {"generic_topic", "generic_heading"}:
        return False

    # 3. 过泛 query 过滤
    if any(x in qzh for x in BAD_QUERY_PATTERNS_ZH):
        return False
    if any(x in qen for x in BAD_QUERY_PATTERNS_EN):
        return False

    # 4. navigation 只保留明确的
    if page_type == "navigation" and source not in {"navigation_emstop", "navigation_heading"}:
        return False

    # 5. semantic 页如果 heading 太弱，过滤
    weak_heading_terms = ["contents", "about this manual", "getting started"]
    if source in {"semantic_keyword", "semantic_topic"} and any(x in heading for x in weak_heading_terms):
        return False

    return True

def retrieval_filter(df_queries: pd.DataFrame, df_docs: pd.DataFrame, bm25: BM25Okapi):
    kept_rows = []

    # 建一个 (doc_id, page_idx) -> row index 的映射
    idx_map = {
        (row.doc_id, int(row.page_idx)): i
        for i, row in df_docs.reset_index(drop=True).iterrows()
    }

    for row in df_queries.itertuples(index=False):
        key = (row.doc_id, int(row.page_idx))
        if key not in idx_map:
            continue
        gold_idx = idx_map[key]

        q_tokens = tokenize(row.query_en)
        scores = bm25.get_scores(q_tokens)

        tmp = df_docs.copy()
        tmp["score"] = scores
        tmp = tmp.sort_values("score", ascending=False).head(20).reset_index(drop=True)

        top20_pairs = set(zip(tmp["doc_id"], tmp["page_idx"].astype(int)))
        top5_docs = set(tmp.head(5)["doc_id"].tolist())

        # 保留条件：
        # 1) gold 页在 BM25 top20 中
        # 或 2) gold 文档在 BM25 top5 中，说明至少对文档是可检索的
        if key in top20_pairs or row.doc_id in top5_docs:
            kept_rows.append(row._asdict())

    return pd.DataFrame(kept_rows)

def main():
    df = pd.read_csv(IN_FILE)
    print("original candidates:", len(df))

    # 规则过滤
    df = df[df.apply(keep_row, axis=1)].copy().reset_index(drop=True)
    print("after rule filter:", len(df))

    # 每页最多保留 3 条，优先保留更“具体”的 source
    source_priority = {
        "fact_number_unit": 1,
        "fact_keyword": 2,
        "fact_page_lookup": 3,
        "layout_workspace": 1,
        "layout_dimensions": 2,
        "layout_schematic": 3,
        "layout_7dof": 4,
        "semantic_keyword": 1,
        "semantic_topic": 2,
        "navigation_emstop": 1,
        "navigation_heading": 2,
        "fact_heading": 5,
        "layout_heading": 5,
    }
    df["source_priority"] = df["query_source"].map(lambda x: source_priority.get(x, 99))
    df = df.sort_values(
        ["doc_id", "page_idx", "source_priority", "page_score", "ocr_text_clean_len"],
        ascending=[True, True, True, False, False]
    ).reset_index(drop=True)

    df["rank_in_page"] = df.groupby(["doc_id", "page_idx"]).cumcount() + 1
    df = df[df["rank_in_page"] <= 3].copy().reset_index(drop=True)
    print("after per-page cap:", len(df))

    # 检索性过滤
    df_docs = pd.read_csv(DOC_FILE)
    df_docs = df_docs[df_docs["ocr_text_clean"].fillna("").str.len() > 0].copy().reset_index(drop=True)
    bm25 = build_bm25(df_docs)

    df = retrieval_filter(df, df_docs, bm25)
    print("after retrieval filter:", len(df))

    df = df.drop(columns=["source_priority", "rank_in_page"], errors="ignore")
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"\nSaved filtered query candidates to: {OUT_FILE}")

    print("\nQueries by page_type:")
    print(df["page_type"].value_counts())

    print("\nQueries by source:")
    print(df["query_source"].value_counts())

    print("\nSample filtered queries:")
    print(df.head(25).to_string(index=False))

if __name__ == "__main__":
    main()
