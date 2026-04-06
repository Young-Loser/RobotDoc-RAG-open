from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


ROOT = Path(__file__).resolve().parents[2]
RERANK_FILE = ROOT / "outputs" / "two_stage_rerank_eval_details.csv"
OUT_DIR = ROOT / "outputs" / "generator_cases"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"

FACT_HINTS = [
    "maximum",
    "rated",
    "payload",
    "degrees of freedom",
    "force",
    "torque",
    "reach",
    "dimension",
    "weight",
    "sensor",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple multimodal generation strategies over retrieved pages.")
    parser.add_argument("--retrieval-file", type=Path, default=RERANK_FILE, help="Retrieval detail CSV to consume.")
    parser.add_argument("--output-file", type=Path, default=OUT_DIR / "multistrategy_generator_results.json", help="Output JSON file.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Vision-language model name.")
    parser.add_argument("--topk-pages", type=int, default=3, help="How many retrieved pages to use.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of queries.")
    parser.add_argument("--query-ids", nargs="*", default=None, help="Only run selected query ids.")
    parser.add_argument("--allow-online-model-load", action="store_true", help="Allow downloading model files if local cache is missing.")
    return parser


def page_path(doc_id: str, page_idx: int) -> Path:
    return ROOT / "data" / "pages" / doc_id / f"page_{page_idx:04d}.png"


def stitch_horizontally(image_paths: list[Path], out_path: Path) -> Path:
    imgs = [Image.open(path).convert("RGB") for path in image_paths]
    heights = [img.height for img in imgs]
    max_h = max(heights)

    resized = []
    for img in imgs:
        if img.height != max_h:
            new_w = int(img.width * (max_h / img.height))
            img = img.resize((new_w, max_h))
        resized.append(img)

    total_w = sum(img.width for img in resized)
    canvas = Image.new("RGB", (total_w, max_h), color=(255, 255, 255))

    x = 0
    for img in resized:
        canvas.paste(img, (x, 0))
        x += img.width

    canvas.save(out_path)
    return out_path


def load_model(model_name: str, allow_online_model_load: bool = False):
    load_path = resolve_model_path(model_name) if not allow_online_model_load else model_name
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        local_files_only=not allow_online_model_load,
    )
    processor = AutoProcessor.from_pretrained(
        load_path,
        local_files_only=not allow_online_model_load,
    )
    return model, processor


def resolve_model_path(model_name: str) -> str:
    repo_dir = HF_CACHE_ROOT / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if not repo_dir.exists():
        return model_name

    snapshots = sorted([path for path in repo_dir.iterdir() if path.is_dir()])
    if not snapshots:
        return model_name
    return str(snapshots[-1])


def run_qwen(model, processor, content_blocks: list[dict], max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": content_blocks}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def is_fact_like_question(question: str) -> bool:
    low = question.lower()
    return any(hint in low for hint in FACT_HINTS)


def parse_score_from_text(text: str, default: float = 0.2) -> float:
    match = re.search(r"(?i)confidence[^0-9]{0,10}([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return default
    score = float(match.group(1))
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def extract_answer_text(text: str) -> str:
    cleaned = text.strip()

    answer_tag_match = re.search(
        r"(?is)<one concise answer>\s*(.*?)\s*(?:</one(?:\s+concise\s+answer)?>|<confidence>|$)",
        cleaned,
    )
    if answer_tag_match:
        cleaned = answer_tag_match.group(1).strip()

    answer_line_match = re.search(r"(?is)answer\s*:\s*(.*?)(?:\n\s*confidence\s*:|$)", cleaned)
    if answer_line_match:
        cleaned = answer_line_match.group(1).strip()

    cleaned = re.sub(r"(?is)</?one\s+concise\s+answer>", "", cleaned)
    cleaned = re.sub(r"(?is)</?confidence>", "", cleaned)
    cleaned = re.sub(r"(?im)^answer\s*:\s*", "", cleaned)
    cleaned = re.sub(r"(?im)^confidence\s*:\s*[0-9]+(?:\.[0-9]+)?\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^[0-9]+(?:\.[0-9]+)?\s*$", "", cleaned)

    lines = [line.strip() for line in cleaned.splitlines()]
    deduped = []
    seen = set()
    for line in lines:
        if not line:
            continue
        if re.fullmatch(r"(?i)confidence\s*:?\s*[0-9]+(?:\.[0-9]+)?", line):
            continue
        if line.lower() in seen:
            continue
        seen.add(line.lower())
        deduped.append(line)

    return "\n".join(deduped).strip()


def strip_confidence_line(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    kept = [line for line in lines if not re.fullmatch(r"(?i)confidence\s*:\s*[0-9]+(?:\.[0-9]+)?", line.strip())]
    return "\n".join(line for line in kept if line.strip()).strip()


def answer_quality_score(answer: str, question: str) -> float:
    answer_clean = extract_answer_text(strip_confidence_line(answer))
    low = answer_clean.lower()
    score = 0.0

    if answer_clean:
        score += 0.25
    if len(answer_clean) >= 20:
        score += 0.2
    if len(answer_clean) >= 60:
        score += 0.15
    if low not in {"yes", "no", "unknown"}:
        score += 0.1
    if not re.fullmatch(r"(?i)confidence\s*:\s*[0-9]+(?:\.[0-9]+)?", answer.strip()):
        score += 0.2
    if "<one concise answer>" not in low and "<confidence>" not in low:
        score += 0.05
    if "cannot confidently answer" not in low and "answer is uncertain" not in low:
        score += 0.05
    if is_fact_like_question(question) and re.search(r"\d", answer_clean):
        score += 0.1

    bad_patterns = [
        "common knowledge",
        "cannot provide a specific answer",
        "the document does not provide specific information",
        "the image does not provide specific information",
    ]
    if any(pattern in low for pattern in bad_patterns):
        score -= 0.4

    return max(0.0, min(1.0, score))


def format_prompt(question: str, mode: str) -> str:
    fact_instruction = (
        "If the question asks for a numeric value, specification, component list, or exact fact, "
        "quote only what is directly supported by the page image. "
        "Do not invent values or borrow facts from another product."
    )
    base = (
        "You are answering from robot manuals and datasheets.\n"
        "Use only the visible evidence in the provided page image(s).\n"
        "If the answer is not clearly supported, say 'Not enough evidence on this page.'\n"
        f"{fact_instruction}\n"
        "Keep the answer short and factual."
    )
    if mode == "per_page":
        return (
            f"{base}\n"
            "Return exactly two sections:\n"
            "Answer: <one concise answer>\n"
            "Confidence: <number between 0 and 1>\n"
            f"Question: {question}"
        )
    if mode == "joint":
        return (
            f"{base}\n"
            "When multiple pages are provided, synthesize only if the pages support the same answer.\n"
            f"Question: {question}"
        )
    if mode == "stitched":
        return (
            f"{base}\n"
            "The input image is a horizontal stitch of several retrieved pages.\n"
            f"Question: {question}"
        )
    return f"{base}\nQuestion: {question}"


def single_page_strategy(model, processor, image_path: Path, question: str) -> dict:
    prompt = format_prompt(question, mode="single")
    answer = run_qwen(
        model,
        processor,
        [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ],
    )
    answer = answer.strip()
    final_answer = extract_answer_text(answer)
    return {
        "answer": final_answer,
        "raw_answer": answer,
        "image_path": str(image_path),
        "quality_score": answer_quality_score(answer, question),
    }


def stitched_strategy(model, processor, stitched_path: Path, question: str) -> dict:
    prompt = format_prompt(question, mode="stitched")
    answer = run_qwen(
        model,
        processor,
        [
            {"type": "image", "image": str(stitched_path)},
            {"type": "text", "text": prompt},
        ],
    )
    answer = answer.strip()
    final_answer = extract_answer_text(answer)
    return {
        "answer": final_answer,
        "raw_answer": answer,
        "stitched_image_path": str(stitched_path),
        "quality_score": answer_quality_score(answer, question),
    }


def per_page_weighted_strategy(model, processor, image_paths: list[Path], question: str, retrieval_scores: list[float]) -> dict:
    page_results = []
    for rank, (image_path, retrieval_score) in enumerate(zip(image_paths, retrieval_scores), start=1):
        prompt = format_prompt(question, mode="per_page")
        answer = run_qwen(
            model,
            processor,
            [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        ).strip()
        answer_only = extract_answer_text(answer)
        generation_confidence = parse_score_from_text(answer)
        quality_score = answer_quality_score(answer, question)
        blended_score = (
            0.45 * generation_confidence
            + 0.25 * max(retrieval_score, 0.0)
            + 0.30 * quality_score
        )
        page_results.append(
            {
                "rank": rank,
                "image_path": str(image_path),
                "retrieval_score": retrieval_score,
                "generation_confidence": generation_confidence,
                "quality_score": quality_score,
                "blended_score": blended_score,
                "answer": answer_only,
                "raw_answer": answer,
            }
        )

    best_page = max(page_results, key=lambda item: item["blended_score"])
    return {
        "selected_answer": best_page["answer"],
        "selected_image_path": best_page["image_path"],
        "selected_blended_score": best_page["blended_score"],
        "per_page_results": page_results,
    }


def multi_image_joint_strategy(model, processor, image_paths: list[Path], question: str) -> dict:
    content = []
    for idx, image_path in enumerate(image_paths, start=1):
        content.append({"type": "image", "image": str(image_path)})
        content.append({"type": "text", "text": f"Page {idx}."})

    content.append(
        {
            "type": "text",
            "text": (
                format_prompt(question, mode="joint")
            ),
        }
    )
    answer = run_qwen(model, processor, content).strip()
    final_answer = extract_answer_text(answer)
    return {
        "answer": final_answer,
        "raw_answer": answer,
        "image_paths": [str(path) for path in image_paths],
        "quality_score": answer_quality_score(answer, question),
    }


def select_final_answer(question: str, single_page: dict, stitched: dict, per_page_weighted: dict, multi_image_joint: dict) -> dict:
    per_page_quality = answer_quality_score(per_page_weighted["selected_answer"], question)
    candidates = [
        ("single_page", single_page["answer"], 0.30 + 0.70 * single_page.get("quality_score", 0.0)),
        ("stitched", stitched["answer"], 0.35 + 0.70 * stitched.get("quality_score", 0.0)),
        ("per_page_weighted", per_page_weighted["selected_answer"], 0.10 + 0.60 * float(per_page_weighted["selected_blended_score"]) + 0.30 * per_page_quality),
        ("multi_image_joint", multi_image_joint["answer"], 0.35 + 0.75 * multi_image_joint.get("quality_score", 0.0)),
    ]
    best_name, best_answer, best_score = max(candidates, key=lambda item: item[2])
    return {
        "selected_strategy": best_name,
        "selected_answer": best_answer,
        "selection_score": best_score,
    }


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.retrieval_file)

    if args.query_ids:
        df = df[df["query_id"].isin(args.query_ids)].copy()
    if args.limit is not None:
        df = df.head(args.limit).copy()

    model, processor = load_model(args.model_name, allow_online_model_load=args.allow_online_model_load)
    all_results = []

    for row in df.itertuples(index=False):
        query_id = row.query_id
        question = row.query_en

        retrieved_doc_ids = json.loads(row.retrieved_doc_ids)
        retrieved_page_idxs = json.loads(row.retrieved_page_idxs)
        retrieved_scores = json.loads(row.retrieved_scores)

        top_imgs = [
            page_path(doc_id, int(page_idx))
            for doc_id, page_idx in zip(retrieved_doc_ids[: args.topk_pages], retrieved_page_idxs[: args.topk_pages])
        ]
        top_scores = [float(score) for score in retrieved_scores[: args.topk_pages]]

        top1_img = top_imgs[0]
        stitched_path = OUT_DIR / f"{query_id}_top{args.topk_pages}_stitched.png"
        stitch_horizontally(top_imgs, stitched_path)

        single_page = single_page_strategy(model, processor, top1_img, question)
        stitched = stitched_strategy(model, processor, stitched_path, question)
        per_page_weighted = per_page_weighted_strategy(model, processor, top_imgs, question, top_scores)
        multi_image_joint = multi_image_joint_strategy(model, processor, top_imgs, question)
        final_choice = select_final_answer(question, single_page, stitched, per_page_weighted, multi_image_joint)

        result = {
            "query_id": query_id,
            "query_type": row.query_type,
            "question": question,
            "retrieved_doc_ids": retrieved_doc_ids[: args.topk_pages],
            "retrieved_page_idxs": retrieved_page_idxs[: args.topk_pages],
            "retrieved_scores": top_scores,
            "single_page": single_page,
            "stitched": stitched,
            "per_page_weighted": per_page_weighted,
            "multi_image_joint": multi_image_joint,
            "final_choice": final_choice,
        }
        all_results.append(result)

        print("\n" + "=" * 100)
        print("query_id:", query_id)
        print("question:", question)
        print("selected_strategy:", final_choice["selected_strategy"])
        print("\n[single_page]")
        print(single_page["answer"])
        print("\n[stitched]")
        print(stitched["answer"])
        print("\n[per_page_weighted]")
        print(per_page_weighted["selected_answer"])
        print("\n[multi_image_joint]")
        print(multi_image_joint["answer"])
        print("\n[final]")
        print(final_choice["selected_answer"])

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to: {args.output_file}")


if __name__ == "__main__":
    main()
