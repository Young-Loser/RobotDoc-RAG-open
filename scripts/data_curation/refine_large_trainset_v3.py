from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "train" / "retriever_train_pairs_large_v2.csv"
OUT_FILE = ROOT / "data" / "train" / "retriever_train_pairs_large_v3.csv"

BAD_QUERY_PATTERNS_EN = [
    "which page discusses '",
    "which page visually presents '",
]

BAD_QUERY_PATTERNS_ZH = [
    "哪一页介绍了“",
    "哪一页展示了“",
]

BAD_HEADING_HINTS = [
    "getting started",
    "table of contents",
    "about this manual",
    "guiding mode",
    "pose teaching",
    "working with",
]

def keep_row(row):
    qen = str(row["query_en"]).strip().lower()
    qzh = str(row["query_zh"]).strip().lower()

    # 过滤明显 heading-copy query
    if any(p in qen for p in BAD_QUERY_PATTERNS_EN):
        return False
    if any(p in qzh for p in BAD_QUERY_PATTERNS_ZH):
        return False

    # 再过滤一层典型章节名/弱标题
    if any(h in qen for h in BAD_HEADING_HINTS):
        return False
    if any(h in qzh for h in BAD_HEADING_HINTS):
        return False

    return True

def main():
    df = pd.read_csv(IN_FILE)
    print("original pairs:", len(df))

    df_out = df[df.apply(keep_row, axis=1)].copy().reset_index(drop=True)

    print("refined pairs:", len(df_out))

    print("\nPairs by neg_source:")
    print(df_out["neg_source"].value_counts())

    print("\nPairs by query_type:")
    print(df_out["query_type"].value_counts())

    print("\nAvg pairs per query:")
    print(round(len(df_out) / df_out["query_id"].nunique(), 2))

    print("\nSample rows:")
    print(df_out.head(12).to_string(index=False))

    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved refined large trainset to: {OUT_FILE}")

if __name__ == "__main__":
    main()
