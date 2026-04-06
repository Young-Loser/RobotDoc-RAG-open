from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

LARGE_FILE = ROOT / "data" / "train" / "retriever_train_pairs_large.csv"
VISION_FILE = ROOT / "data" / "train" / "vision_hard_negatives.csv"

OUT_FILE = ROOT / "data" / "train" / "retriever_train_pairs_large_v2.csv"

def page_path(doc_id: str, page_idx: int) -> str:
    return str(ROOT / "data" / "pages" / doc_id / f"page_{page_idx:04d}.png")

def main():
    df_large = pd.read_csv(LARGE_FILE)
    df_vis = pd.read_csv(VISION_FILE)

    # 把 vision_hard 转成与 large_train 一致的列格式
    df_vis_fmt = pd.DataFrame({
        "sample_id": df_vis["query_id"].astype(str) + "_vision",
        "query_id": df_vis["query_id"],
        "query_en": df_vis["query_en"],
        "query_zh": df_vis["query_zh"],
        "query_type": df_vis["query_type"],
        "pos_doc_id": df_vis["pos_doc_id"],
        "pos_page_idx": df_vis["pos_page_idx"].astype(int),
        "pos_image_path": [
            page_path(d, int(p))
            for d, p in zip(df_vis["pos_doc_id"], df_vis["pos_page_idx"])
        ],
        "neg_doc_id": df_vis["neg_doc_id"],
        "neg_page_idx": df_vis["neg_page_idx"].astype(int),
        "neg_image_path": [
            page_path(d, int(p))
            for d, p in zip(df_vis["neg_doc_id"], df_vis["neg_page_idx"])
        ],
        "neg_source": "vision_hard",
    })

    # 合并
    df_out = pd.concat([df_large, df_vis_fmt], axis=0, ignore_index=True)

    # 去重：同一 query / 正页 / 负页 只保留一次
    df_out = df_out.drop_duplicates(
        subset=["query_id", "pos_doc_id", "pos_page_idx", "neg_doc_id", "neg_page_idx"]
    ).reset_index(drop=True)

    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"Saved merged training pairs to: {OUT_FILE}")
    print("large_v1_pairs:", len(df_large))
    print("vision_hard_pairs:", len(df_vis_fmt))
    print("large_v2_pairs:", len(df_out))

    print("\nPairs by neg_source:")
    print(df_out["neg_source"].value_counts())

    print("\nPairs by query_type:")
    print(df_out["query_type"].value_counts())

    print("\nAvg pairs per query:")
    print(round(len(df_out) / df_out["query_id"].nunique(), 2))

    print("\nSample rows:")
    print(df_out.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
