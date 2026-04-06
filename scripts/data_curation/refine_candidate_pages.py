# second pages candidate

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "train" / "page_candidates_top500.csv"
OUT_FILE = ROOT / "data" / "train" / "page_candidates_final.csv"

def keep_row(row):
    page_type = str(row["page_type"])
    score = float(row["page_score"])
    text = str(row["ocr_text_clean"]) if not pd.isna(row["ocr_text_clean"]) else ""
    text_low = text.lower()
    text_len = int(row["ocr_text_clean_len"]) if not pd.isna(row["ocr_text_clean_len"]) else len(text)

    # 过滤极低信息页
    if text_len < 120:
        return False

    # 强过滤：法律协议
    if "license agreement" in text_low or "end-user license agreement" in text_low:
        return False

    # 强过滤：证书/声明/注册/目录类导航页
    noisy_nav_terms = [
        "certificate", "certificates", "declaration", "declaration of incorporation",
        "registration", "contents", "table of contents"
    ]
    if page_type == "navigation" and any(k in text_low for k in noisy_nav_terms):
        return False

    # 短证书页直接过滤
    if "certificate" in text_low and text_len < 1200:
        return False

    # 高价值类型保留，但导航要分数够
    if page_type in {"fact", "semantic", "layout"}:
        return True

    if page_type == "navigation":
        return score >= 10

    # generic 只保留高分页
    if page_type == "generic":
        return score >= 10

    return False

def main():
    df = pd.read_csv(IN_FILE)
    df_out = df[df.apply(keep_row, axis=1)].copy().reset_index(drop=True)

    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"Saved refined candidates to: {OUT_FILE}")
    print("original pages:", len(df))
    print("refined pages:", len(df_out))

    print("\nRefined pages by doc:")
    print(df_out["doc_id"].value_counts())

    print("\nRefined pages by type:")
    print(df_out["page_type"].value_counts())

    print("\nLowest-score kept pages:")
    print(df_out.sort_values("page_score", ascending=True)[
        ["doc_id", "page_idx", "page_type", "page_score", "ocr_text_clean_len", "score_reasons"]
    ].head(15).to_string(index=False))

if __name__ == "__main__":
    main()
