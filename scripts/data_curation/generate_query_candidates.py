from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "train" / "page_candidates_final.csv"
OUT_FILE = ROOT / "data" / "train" / "page_query_candidates.csv"

NUMBER_UNIT_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s?(mm|cm|m|kg|g|nm|n·m|nm|v|a|w|hz|ms|s|deg|°|mbps|gbps)\b",
    re.IGNORECASE,
)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()

def split_lines(text: str):
    if not isinstance(text, str):
        return []
    lines = []
    for x in text.splitlines():
        x = clean_text(x)
        if len(x) >= 3:
            lines.append(x)
    return lines

def get_heading_candidates(lines):
    headings = []
    for line in lines[:12]:
        if 4 <= len(line) <= 80:
            headings.append(line)
    return headings[:5]

def first_match(text_low, keywords):
    for kw in keywords:
        if kw in text_low:
            return kw
    return None

def extract_number_unit(text):
    ms = NUMBER_UNIT_PATTERN.findall(text)
    if not ms:
        return None
    # find original matched substring
    m = NUMBER_UNIT_PATTERN.search(text)
    return m.group(0) if m else None

def zh_query_from_keyword(kw):
    mapping = {
        "payload": "额定负载",
        "reach": "最大工作半径",
        "degrees of freedom": "自由度",
        "dof": "自由度",
        "torque": "力矩",
        "force": "力/力矩",
        "sensor": "传感器",
        "sensing": "感知能力",
        "interface": "接口",
        "connector": "连接器",
        "ethernet": "以太网接口",
        "tcp/ip": "TCP/IP 接口",
        "installation": "安装说明",
        "maintenance": "维护说明",
        "safety": "安全说明",
        "emergency stop": "紧急停止",
        "workspace": "工作空间示意图",
        "dimensions": "尺寸图",
        "schematic": "原理/结构示意图",
        "components": "组成部件",
        "operation": "操作说明",
        "error": "错误说明",
        "status": "状态说明",
        "gripper": "夹爪",
        "end effector": "末端执行器",
    }
    return mapping.get(kw, kw)

def generate_fact_queries(text, lines):
    text_low = text.lower()
    queries = []

    keyword_groups = [
        "payload", "reach", "degrees of freedom", "dof", "torque",
        "force", "sensor", "sensing", "interface", "connector",
        "ethernet", "tcp/ip", "gripper", "end effector"
    ]

    kw = first_match(text_low, keyword_groups)
    num_unit = extract_number_unit(text)
    heading = lines[0] if lines else ""

    if kw:
        zh_kw = zh_query_from_keyword(kw)
        queries.append((
            f"这页中关于{zh_kw}的参数是什么？",
            f"What specification about {kw} is given on this page?",
            "fact_keyword"
        ))
        queries.append((
            f"哪一页提到了{zh_kw}相关参数？",
            f"Which page mentions {kw} related specifications?",
            "fact_page_lookup"
        ))

    if num_unit:
        queries.append((
            f"哪一页包含参数值 {num_unit}？",
            f"Which page contains the specification value {num_unit}?",
            "fact_number_unit"
        ))

    if heading:
        queries.append((
            f"哪一页介绍了“{heading[:40]}”相关内容？",
            f"Which page discusses '{heading[:40]}'?",
            "fact_heading"
        ))

    if not queries:
        queries.append((
            "这页包含哪些技术规格参数？",
            "What technical specifications are shown on this page?",
            "fact_fallback"
        ))
        queries.append((
            "哪一页是技术规格相关页面？",
            "Which page is about technical specifications?",
            "fact_fallback2"
        ))

    return queries[:3]

def generate_layout_queries(text, lines):
    text_low = text.lower()
    queries = []

    if "workspace" in text_low:
        queries.append((
            "哪一页是工作空间示意图页面？",
            "Which page shows the workspace diagram?",
            "layout_workspace"
        ))
    if "dimensions" in text_low or "dimension" in text_low:
        queries.append((
            "哪一页包含尺寸图或尺寸示意？",
            "Which page contains the dimensions diagram?",
            "layout_dimensions"
        ))
    if "schematic" in text_low:
        queries.append((
            "哪一页包含结构/原理示意图？",
            "Which page contains the schematic diagram?",
            "layout_schematic"
        ))
    if "7 dof" in text_low or "degrees of freedom" in text_low:
        queries.append((
            "哪一页包含 7 自由度相关尺寸图？",
            "Which page contains the 7 DoF dimensions figure?",
            "layout_7dof"
        ))

    heading = lines[0] if lines else ""
    if heading:
        queries.append((
            f"哪一页展示了“{heading[:40]}”的图示页面？",
            f"Which page visually presents '{heading[:40]}'?",
            "layout_heading"
        ))

    if not queries:
        queries.append((
            "哪一页包含图示或布局类信息？",
            "Which page contains diagram or layout information?",
            "layout_fallback"
        ))
        queries.append((
            "哪一页更像示意图页面？",
            "Which page looks like a schematic or layout page?",
            "layout_fallback2"
        ))

    return queries[:3]

def generate_semantic_queries(text, lines):
    text_low = text.lower()
    queries = []

    semantic_keywords = [
        "components", "installation", "maintenance", "safety", "operation",
        "error", "status", "module", "overview", "connector", "interface"
    ]

    kw = first_match(text_low, semantic_keywords)
    heading = lines[0] if lines else ""

    if kw:
        zh_kw = zh_query_from_keyword(kw)
        queries.append((
            f"哪一页介绍了{zh_kw}相关内容？",
            f"Which page describes {kw} related information?",
            "semantic_keyword"
        ))
        queries.append((
            f"这页主要讲的是哪方面的{zh_kw}内容？",
            f"What {kw} topic is mainly explained on this page?",
            "semantic_topic"
        ))

    if heading:
        queries.append((
            f"哪一页讲解了“{heading[:40]}”这一主题？",
            f"Which page explains the topic '{heading[:40]}'?",
            "semantic_heading"
        ))

    if not queries:
        queries.append((
            "哪一页是功能说明或操作说明页面？",
            "Which page is about operation or functional description?",
            "semantic_fallback"
        ))
        queries.append((
            "哪一页包含模块/组件说明？",
            "Which page contains module or component descriptions?",
            "semantic_fallback2"
        ))

    return queries[:2]

def generate_navigation_queries(text, lines):
    text_low = text.lower()
    queries = []

    if "emergency stop" in text_low:
        queries.append((
            "哪一页提到了紧急停止相关条目？",
            "Which page mentions the emergency stop entry?",
            "navigation_emstop"
        ))
    if "contents" in text_low or "table of contents" in text_low:
        queries.append((
            "哪一页是目录相关页面？",
            "Which page is a table of contents page?",
            "navigation_contents"
        ))

    heading = lines[0] if lines else ""
    if heading:
        queries.append((
            f"哪一页包含“{heading[:40]}”这类导航信息？",
            f"Which page contains navigation information about '{heading[:40]}'?",
            "navigation_heading"
        ))

    if not queries:
        queries.append((
            "哪一页更像目录或导航页？",
            "Which page looks like a navigation or contents page?",
            "navigation_fallback"
        ))
        queries.append((
            "哪一页列出了章节或条目？",
            "Which page lists sections or entries?",
            "navigation_fallback2"
        ))

    return queries[:2]

def generate_generic_queries(text, lines):
    heading = lines[0] if lines else ""
    queries = []

    if heading:
        queries.append((
            f"哪一页讨论了“{heading[:40]}”？",
            f"Which page discusses '{heading[:40]}'?",
            "generic_heading"
        ))

    queries.append((
        "哪一页包含这一主题的说明内容？",
        "Which page contains information about this topic?",
        "generic_topic"
    ))

    return queries[:2]

def main():
    df = pd.read_csv(IN_FILE)

    rows = []
    qid = 1

    for row in df.itertuples(index=False):
        doc_id = row.doc_id
        page_idx = int(row.page_idx)
        page_type = row.page_type
        text = row.ocr_text_clean if isinstance(row.ocr_text_clean, str) else ""
        lines = get_heading_candidates(split_lines(text))

        if page_type == "fact":
            queries = generate_fact_queries(text, lines)
        elif page_type == "layout":
            queries = generate_layout_queries(text, lines)
        elif page_type == "semantic":
            queries = generate_semantic_queries(text, lines)
        elif page_type == "navigation":
            queries = generate_navigation_queries(text, lines)
        else:
            queries = generate_generic_queries(text, lines)

        for query_zh, query_en, source in queries:
            rows.append({
                "query_id": f"auto_{qid:05d}",
                "doc_id": doc_id,
                "page_idx": page_idx,
                "page_type": page_type,
                "query_zh": clean_text(query_zh),
                "query_en": clean_text(query_en),
                "query_source": source,
                "anchor_heading": lines[0] if lines else "",
                "ocr_text_clean_len": int(row.ocr_text_clean_len),
                "page_score": float(row.page_score),
            })
            qid += 1

    df_out = pd.DataFrame(rows)

    # 去重：中英文 query 重复去掉
    df_out = df_out.drop_duplicates(
        subset=["doc_id", "page_idx", "query_zh", "query_en"]
    ).reset_index(drop=True)

    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"Saved query candidates to: {OUT_FILE}")
    print("num_pages_used:", df[['doc_id','page_idx']].drop_duplicates().shape[0])
    print("num_query_candidates:", len(df_out))

    print("\nQueries by page_type:")
    print(df_out["page_type"].value_counts())

    print("\nQueries by source:")
    print(df_out["query_source"].value_counts().head(20))

    print("\nSample queries:")
    print(df_out.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
