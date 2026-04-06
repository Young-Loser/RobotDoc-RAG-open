from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def batch_list(xs, bs: int):
    for i in range(0, len(xs), bs):
        yield xs[i : i + bs]


def checkpoint_fingerprint(checkpoint_path: Path) -> str:
    stat = checkpoint_path.stat()
    payload = f"{checkpoint_path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_is_compatible(
    meta_file: Path,
    *,
    checkpoint_path: Path,
    model_pooling: str,
    num_docs: int,
    text_pooling: str | None = None,
    image_pooling: str | None = None,
) -> bool:
    if not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        meta.get("checkpoint_fingerprint") == checkpoint_fingerprint(checkpoint_path)
        and meta.get("pooling") == model_pooling
        and meta.get("text_pooling") == text_pooling
        and meta.get("image_pooling") == image_pooling
        and int(meta.get("num_docs", -1)) == int(num_docs)
    )


def write_cache_metadata(
    meta_file: Path,
    *,
    checkpoint_path: Path,
    model_pooling: str,
    num_docs: int,
    text_pooling: str | None = None,
    image_pooling: str | None = None,
) -> None:
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_fingerprint": checkpoint_fingerprint(checkpoint_path),
        "pooling": model_pooling,
        "text_pooling": text_pooling,
        "image_pooling": image_pooling,
        "num_docs": int(num_docs),
    }
    meta_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.no_grad()
def encode_all_pages(df_docs: pd.DataFrame, model, image_bs: int = 16):
    doc_embs = []
    image_paths = df_docs["image_path"].tolist()

    for batch_paths in tqdm(batch_list(image_paths, image_bs), total=(len(image_paths) + image_bs - 1) // image_bs, desc="Encoding full corpus images"):
        batch_imgs = [Image.open(path).convert("RGB") for path in batch_paths]
        feats = model.encode_images(batch_imgs)
        doc_embs.append(feats.cpu())

        del batch_imgs, feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(doc_embs, dim=0)


@torch.no_grad()
def encode_queries(queries: Iterable[str], model, query_bs: int = 16):
    query_embs = []
    queries = list(queries)

    for batch_q in tqdm(batch_list(queries, query_bs), total=(len(queries) + query_bs - 1) // query_bs, desc="Encoding queries"):
        feats = model.encode_text(batch_q)
        query_embs.append(feats.cpu())

        del feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(query_embs, dim=0)


def summarize_retrieval_results(df_res: pd.DataFrame, extra: dict | None = None) -> dict:
    summary = {
        "num_queries": int(len(df_res)),
        "top1_exact_acc": float(df_res["top1_exact_hit"].mean()) if len(df_res) else 0.0,
        "top5_exact_recall": float(df_res["top5_exact_hit"].mean()) if len(df_res) else 0.0,
        "top1_doc_acc": float(df_res["top1_doc_hit"].mean()) if len(df_res) else 0.0,
        "top5_doc_recall": float(df_res["top5_doc_hit"].mean()) if len(df_res) else 0.0,
    }

    by_type = []
    for qtype, group in df_res.groupby("query_type"):
        by_type.append(
            {
                "query_type": qtype,
                "num_queries": int(len(group)),
                "top1_exact_acc": float(group["top1_exact_hit"].mean()),
                "top5_exact_recall": float(group["top5_exact_hit"].mean()),
                "top1_doc_acc": float(group["top1_doc_hit"].mean()),
                "top5_doc_recall": float(group["top5_doc_hit"].mean()),
            }
        )
    summary["by_query_type"] = by_type

    if extra:
        summary.update(extra)
    return summary


def build_failure_table(df_res: pd.DataFrame) -> pd.DataFrame:
    return df_res[
        (~df_res["top1_exact_hit"]) | (~df_res["top5_exact_hit"])
    ][
        [
            "query_id",
            "query_type",
            "query_en",
            "gold_doc_id",
            "gold_page_idx",
            "top1_doc_id",
            "top1_page_idx",
            "top1_doc_hit",
            "top1_exact_hit",
            "top5_doc_hit",
            "top5_exact_hit",
        ]
    ].reset_index(drop=True)
