from pathlib import Path
import re
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "ocr" / "page_texts_clean.csv"

OUT_ALL = ROOT / "data" / "train" / "page_candidates_scored.csv"
OUT_TOP = ROOT / "data" / "train" / "page_candidates_top500.csv"

TARGET_TOTAL = 500

HIGH_VALUE_KEYWORDS = [
    # 参数/规格/尺寸/接口
    "specification", "specifications", "technical specifications",
    "dimension", "dimensions", "workspace", "schematic", "diagram",
    "payload", "reach", "joint", "torque", "force", "sensor", "sensing",
    "interface", "connector", "ethernet", "tcp/ip", "usb", "io",
    "voltage", "current", "power", "weight", "mass", "speed",
    "accuracy", "repeatability", "range", "resolution",
    "installation", "mounting", "wiring", "cable", "pin", "port",
    "component", "components", "overview", "module",
    "safety", "emergency stop", "protection", "operating", "operation",
    "maintenance", "calibration", "troubleshooting", "error", "status",
    # 机器人/机械臂常见术语
    "degrees of freedom", "dof", "end effector", "gripper",
    "flange", "actuator", "base", "vision module", "control connector",
]

LOW_VALUE_KEYWORDS = [
    # 低价值/法律/附录/目录
    "license agreement", "end-user license agreement", "eula",
    "certificate", "certificates", "registration",
    "declaration of incorporation", "declaration", "compliance",
    "table of contents", "contents", "index", "glossary",
    "copyright", "trademark", "warranty", "legal", "licensor",
    "agreement", "terms and conditions",
]

NAVIGATION_KEYWORDS = [
    "contents", "emergency stop", "declaration of incorporation",
    "certificate", "table of contents"
]

UNITS_PATTERN = re.compile(
    r"\b(\d+(\.\d+)?\s?(mm|cm|m|kg|g|nm|n·m|nm|v|a|w|hz|ms|s|deg|°|mbps|gbps))\b",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(r"\b\d+(\.\d+)?\b")

BULLET_PATTERN = re.compile(r"[•·▪◦]")

UPPER_TITLE_PATTERN = re.compile(r"^[A-Z0-9\s\-\&\(\)\/]+$")

def count_keyword_hits(text: str, keywords):
    text_low = text.lower()
    hits = 0
    matched = []
    for kw in keywords:
        if kw in text_low:
            hits += 1
            matched.append(kw)
    return hits, matched

def classify_page_type(text: str):
    t = text.lower()

    if any(k in t for k in ["workspace", "schematic", "diagram", "dimensions"]):
        return "layout"
    if any(k in t for k in ["specification", "payload", "reach", "degrees of freedom", "torque", "sensor"]):
        return "fact"
    if any(k in t for k in ["components", "overview", "module", "installation", "maintenance", "safety"]):
        return "semantic"
    if any(k in t for k in NAVIGATION_KEYWORDS):
        return "navigation"
    return "generic"

def compute_score(row):
    text = row["ocr_text_clean"] if isinstance(row["ocr_text_clean"], str) else ""
    text_low = text.lower()
    text_len = len(text)
    num_boxes = int(row["num_boxes"]) if not pd.isna(row["num_boxes"]) else 0

    score = 0.0
    reasons = []

    # 1) 文本长度：太短差，适中最好，极长略加分但别太夸张
    if text_len < 50:
        score -= 5
        reasons.append("too_short")
    elif text_len < 120:
        score -= 1
        reasons.append("short")
    elif text_len <= 3500:
        score += 3
        reasons.append("good_length")
    else:
        score += 1
        reasons.append("very_long")

    # 2) OCR 框数量：信息密度线索
    if num_boxes >= 8:
        score += 2
        reasons.append("dense_boxes")
    if num_boxes >= 20:
        score += 1
        reasons.append("very_dense_boxes")

    # 3) 数字/单位：对 fact 页很重要
    unit_hits = len(UNITS_PATTERN.findall(text))
    number_hits = len(NUMBER_PATTERN.findall(text))

    score += min(unit_hits, 6) * 0.6
    if unit_hits > 0:
        reasons.append(f"unit_hits={unit_hits}")

    score += min(number_hits, 20) * 0.08
    if number_hits >= 5:
        reasons.append(f"number_hits={number_hits}")

    # 4) 项目符号/多项结构：说明页常见
    bullet_hits = len(BULLET_PATTERN.findall(text))
    score += min(bullet_hits, 8) * 0.25
    if bullet_hits > 0:
        reasons.append(f"bullet_hits={bullet_hits}")

    # 5) 高价值关键词
    hv_hits, hv_matched = count_keyword_hits(text, HIGH_VALUE_KEYWORDS)
    score += min(hv_hits, 8) * 1.2
    if hv_hits > 0:
        reasons.append(f"high_value={','.join(hv_matched[:4])}")

    # 6) 低价值关键词：扣分，但 navigation 少量保留
    lv_hits, lv_matched = count_keyword_hits(text, LOW_VALUE_KEYWORDS)
    score -= min(lv_hits, 6) * 1.5
    if lv_hits > 0:
        reasons.append(f"low_value={','.join(lv_matched[:4])}")

    # 7) 全大写标题/结构化强页，略加分
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    upper_title_hits = 0
    for line in lines[:8]:
        if 6 <= len(line) <= 60 and UPPER_TITLE_PATTERN.match(line):
            upper_title_hits += 1
    score += min(upper_title_hits, 3) * 0.4
    if upper_title_hits > 0:
        reasons.append(f"upper_titles={upper_title_hits}")

    # 8) 特别惩罚：纯法律/证书类
    if "license agreement" in text_low or "end-user license agreement" in text_low:
        score -= 4
        reasons.append("legal_page")
    if "certificate" in text_low and text_len < 1200:
        score -= 2
        reasons.append("certificate_page")

    page_type = classify_page_type(text)

    return score, page_type, ";".join(reasons)

def allocate_quotas(df, target_total=500):
    """
    按文档页数分配配额，但设置上下界，避免被大文档垄断。
    """
    counts = df.groupby("doc_id").size().to_dict()
    total_pages = sum(counts.values())

    raw = {}
    for doc_id, n in counts.items():
        # 基于占比
        q = target_total * (n / total_pages)
        raw[doc_id] = q

    quotas = {}
    for doc_id, q in raw.items():
        quotas[doc_id] = int(round(q))

    # 下界 / 上界
    min_quota = 25
    max_quota = 90

    for doc_id in quotas:
        quotas[doc_id] = max(min_quota, min(max_quota, quotas[doc_id]))

    # 调整总和到 target_total
    current = sum(quotas.values())

    # 如果过多，从大文档减
    while current > target_total:
        doc_sorted = sorted(quotas.keys(), key=lambda x: quotas[x], reverse=True)
        changed = False
        for d in doc_sorted:
            if quotas[d] > min_quota:
                quotas[d] -= 1
                current -= 1
                changed = True
                if current == target_total:
                    break
        if not changed:
            break

    # 如果不足，给大文档加
    while current < target_total:
        doc_sorted = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
        changed = False
        for d in doc_sorted:
            if quotas[d] < max_quota:
                quotas[d] += 1
                current += 1
                changed = True
                if current == target_total:
                    break
        if not changed:
            break

    return quotas

def main():
    df = pd.read_csv(IN_FILE)

    # 只保留有清洗文本的页面
    df = df[df["ocr_text_clean"].fillna("").str.len() > 0].copy().reset_index(drop=True)

    scores = df.apply(compute_score, axis=1, result_type="expand")
    scores.columns = ["page_score", "page_type", "score_reasons"]
    df = pd.concat([df, scores], axis=1)

    # 标记是否建议过滤
    df["is_strong_low_value"] = df["ocr_text_clean"].fillna("").str.lower().apply(
        lambda x: any(k in x for k in ["license agreement", "end-user license agreement"])
    )

    # 文档配额
    quotas = allocate_quotas(df, target_total=TARGET_TOTAL)

    selected_parts = []
    for doc_id, q in quotas.items():
        sub = df[df["doc_id"] == doc_id].copy()

        # navigation 页留少量，不完全禁掉
        nav_mask = (sub["page_type"] == "navigation")
        strong_low_mask = sub["is_strong_low_value"]

        # 先去掉特别差的法律页
        sub = sub[~strong_low_mask].copy()

        # 排序：高分优先
        sub = sub.sort_values(["page_score", "ocr_text_clean_len"], ascending=[False, False])

        # 控制 navigation 不超过文档配额的 10%
        nav_cap = max(2, int(math.ceil(q * 0.10)))
        nav_rows = sub[sub["page_type"] == "navigation"].head(nav_cap)
        non_nav_rows = sub[sub["page_type"] != "navigation"].head(max(q - len(nav_rows), 0))

        picked = pd.concat([non_nav_rows, nav_rows], axis=0).drop_duplicates(
            subset=["doc_id", "page_idx"]
        )

        # 如果还不够，再从剩余高分里补
        if len(picked) < q:
            remain = sub[
                ~sub.set_index(["doc_id", "page_idx"]).index.isin(
                    picked.set_index(["doc_id", "page_idx"]).index
                )
            ].head(q - len(picked))
            picked = pd.concat([picked, remain], axis=0)

        picked["selected_quota"] = q
        selected_parts.append(picked.head(q))

    df_top = pd.concat(selected_parts, axis=0).drop_duplicates(
        subset=["doc_id", "page_idx"]
    ).reset_index(drop=True)

    # 如果最终不是 500，做一次全局修正
    if len(df_top) > TARGET_TOTAL:
        df_top = df_top.sort_values(["page_score", "ocr_text_clean_len"], ascending=[False, False]).head(TARGET_TOTAL)
    elif len(df_top) < TARGET_TOTAL:
        already = set(zip(df_top["doc_id"], df_top["page_idx"]))
        extra = df[
            ~df.set_index(["doc_id", "page_idx"]).index.isin(already)
        ].sort_values(["page_score", "ocr_text_clean_len"], ascending=[False, False]).head(TARGET_TOTAL - len(df_top))
        df_top = pd.concat([df_top, extra], axis=0).drop_duplicates(subset=["doc_id", "page_idx"]).head(TARGET_TOTAL)

    # 输出
    df = df.sort_values(["page_score", "ocr_text_clean_len"], ascending=[False, False]).reset_index(drop=True)
    df_top = df_top.sort_values(["doc_id", "page_idx"]).reset_index(drop=True)

    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ALL, index=False, encoding="utf-8")
    df_top.to_csv(OUT_TOP, index=False, encoding="utf-8")

    print(f"Saved scored pages to: {OUT_ALL}")
    print(f"Saved top candidate pages to: {OUT_TOP}")
    print("\nTotal scored pages:", len(df))
    print("Selected pages:", len(df_top))

    print("\nSelected pages by doc:")
    print(df_top["doc_id"].value_counts())

    print("\nSelected pages by type:")
    print(df_top["page_type"].value_counts())

    print("\nTop 15 selected pages:")
    print(df_top[[
        "doc_id", "page_idx", "page_type", "page_score", "ocr_text_clean_len", "num_boxes", "score_reasons"
    ]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
