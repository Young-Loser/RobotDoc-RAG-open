from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PairExample:
    query_id: str
    query_en: str
    query_type: str
    pos_image_path: str
    neg_image_path: str


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> PairExample:
        row = self.df.iloc[idx]
        return PairExample(
            query_id=row["query_id"],
            query_en=row["query_en"],
            query_type=row["query_type"],
            pos_image_path=row["pos_image_path"],
            neg_image_path=row["neg_image_path"],
        )


def collate_pair_batch(batch: list[PairExample]):
    queries = [item.query_en for item in batch]
    pos_imgs = [Image.open(item.pos_image_path).convert("RGB") for item in batch]
    neg_imgs = [Image.open(item.neg_image_path).convert("RGB") for item in batch]
    qtypes = [item.query_type for item in batch]
    return queries, pos_imgs, neg_imgs, qtypes


def split_pairs_by_query(df: pd.DataFrame, val_ratio: float, min_val_queries: int, seed: int):
    qids = sorted(df["query_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(qids)

    n_val = min(len(qids) - 1, max(min_val_queries, int(len(qids) * val_ratio)))
    n_val = max(1, n_val)

    val_qids = set(qids[-n_val:])
    train_qids = set(qids[:-n_val])

    df_train = df[df["query_id"].isin(train_qids)].reset_index(drop=True)
    df_val = df[df["query_id"].isin(val_qids)].reset_index(drop=True)
    return df_train, df_val, sorted(train_qids), sorted(val_qids)


def build_weighted_sampler(df_train: pd.DataFrame) -> WeightedRandomSampler:
    counts = Counter(df_train["query_type"].tolist())
    weights = [1.0 / counts[qtype] for qtype in df_train["query_type"].tolist()]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


def load_training_pairs(train_file: Path) -> pd.DataFrame:
    df = pd.read_csv(train_file)
    required_columns = {
        "query_id",
        "query_en",
        "query_type",
        "pos_image_path",
        "neg_image_path",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {train_file}: {sorted(missing)}")
    return df


def summarize_training_pairs(df: pd.DataFrame) -> dict:
    missing_pos = 0
    missing_neg = 0
    for row in df.itertuples(index=False):
        if not Path(row.pos_image_path).exists():
            missing_pos += 1
        if not Path(row.neg_image_path).exists():
            missing_neg += 1

    return {
        "rows": int(len(df)),
        "query_count": int(df["query_id"].nunique()),
        "query_type_counts": {k: int(v) for k, v in df["query_type"].value_counts().to_dict().items()},
        "neg_source_counts": {k: int(v) for k, v in df["neg_source"].value_counts().to_dict().items()} if "neg_source" in df.columns else {},
        "missing_pos_images": missing_pos,
        "missing_neg_images": missing_neg,
    }
