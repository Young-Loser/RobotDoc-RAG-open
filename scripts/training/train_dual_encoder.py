from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from robotdoc_rag.retriever import (
    PageDualEncoder,
    RetrieverConfig,
    get_default_device,
    hard_negative_margin_loss,
    info_nce_loss,
    save_retriever_checkpoint,
)
from robotdoc_rag.training_data import (
    PairDataset,
    build_weighted_sampler,
    collate_pair_batch,
    load_training_pairs,
    set_seed,
    split_pairs_by_query,
    summarize_training_pairs,
)


DEFAULT_TRAIN_FILE = ROOT / "data" / "train" / "retriever_train_pairs_large_v3.csv"
DEFAULT_CKPT_DIR = ROOT / "outputs" / "checkpoints"
DEFAULT_SUMMARY_DIR = ROOT / "outputs" / "training"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a page-level dual-encoder retriever.")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE, help="Training pairs CSV.")
    parser.add_argument("--checkpoint-name", default="siglip_retriever_stage3_best.pt", help="Best checkpoint filename.")
    parser.add_argument("--summary-name", default="siglip_retriever_stage3_train_summary.json", help="Training summary filename.")
    parser.add_argument("--model-name", default="google/siglip-base-patch16-224", help="Backbone model name.")
    parser.add_argument("--device", default=get_default_device(), help="Training device.")
    parser.add_argument("--allow-online-model-load", action="store_true", help="Allow Hugging Face downloads when local cache is missing.")
    parser.add_argument("--pooling", choices=["cls", "position_weighted"], default="position_weighted", help="Feature pooling strategy.")
    parser.add_argument("--text-pooling", choices=["cls", "position_weighted", "pretrained"], default=None, help="Optional override for text-side pooling.")
    parser.add_argument("--image-pooling", choices=["cls", "position_weighted", "pretrained"], default=None, help="Optional override for image-side pooling.")
    parser.add_argument("--out-dim", type=int, default=256, help="Projection output dimension.")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum query token length.")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature.")
    parser.add_argument("--margin", type=float, default=0.2, help="Hard negative margin.")
    parser.add_argument("--margin-weight", type=float, default=0.25, help="Weight for hard negative margin loss.")
    parser.add_argument("--val-ratio", type=float, default=0.12, help="Validation query ratio.")
    parser.add_argument("--min-val-queries", type=int, default=2, help="Minimum number of validation queries.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Balance training by query_type.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches per epoch for smoke testing.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit validation batches for smoke testing.")
    parser.add_argument("--describe-only", action="store_true", help="Only print dataset summary and split info.")
    return parser


@torch.no_grad()
def evaluate(model, loader, temperature: float, margin: float, margin_weight: float, max_batches: int | None = None):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for queries, pos_imgs, neg_imgs, _ in loader:
        q = model.encode_text(queries)
        p = model.encode_images(pos_imgs)
        n = model.encode_images(neg_imgs)

        loss_nce = info_nce_loss(q, p, temperature=temperature)
        loss_margin = hard_negative_margin_loss(q, p, n, margin=margin)
        loss = loss_nce + margin_weight * loss_margin

        total_loss += loss.item()
        total_batches += 1
        if max_batches is not None and total_batches >= max_batches:
            break

    return total_loss / max(total_batches, 1)


def resolve_progress_total(loader, max_batches: int | None) -> int:
    if max_batches is None:
        return len(loader)
    return min(len(loader), max_batches)


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)

    df = load_training_pairs(args.train_file)
    summary = summarize_training_pairs(df)

    print("Training pair summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    df_train, df_val, train_qids, val_qids = split_pairs_by_query(
        df,
        val_ratio=args.val_ratio,
        min_val_queries=args.min_val_queries,
        seed=args.seed,
    )

    print("\nSplit summary:")
    print(f"train_qids: {len(train_qids)}")
    print(f"val_qids: {len(val_qids)}")
    print(f"train_pairs: {len(df_train)}")
    print(f"val_pairs: {len(df_val)}")

    if args.describe_only:
        return 0

    sampler = build_weighted_sampler(df_train) if args.use_weighted_sampler else None

    train_loader = DataLoader(
        PairDataset(df_train),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_pair_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        PairDataset(df_val),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pair_batch,
        num_workers=args.num_workers,
    )

    model_config = RetrieverConfig(
        model_name=args.model_name,
        out_dim=args.out_dim,
        pooling=args.pooling,
        text_pooling=args.text_pooling,
        image_pooling=args.image_pooling,
        max_length=args.max_length,
        device=args.device,
        local_files_only=not args.allow_online_model_load,
    )
    model = PageDualEncoder(model_config).to(args.device)
    print(f"\nLoaded dual encoder on device: {args.device}")
    print(f"Pooling strategy: base={args.pooling}, text={args.text_pooling or args.pooling}, image={args.image_pooling or args.pooling}")
    print(f"Train batches per epoch: {resolve_progress_total(train_loader, args.max_train_batches)}")
    print(f"Val batches per evaluation: {resolve_progress_total(val_loader, args.max_val_batches)}")

    optimizer = torch.optim.AdamW(
        list(model.text_proj.parameters()) + list(model.image_proj.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    DEFAULT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    best_path = DEFAULT_CKPT_DIR / args.checkpoint_name
    summary_path = DEFAULT_SUMMARY_DIR / args.summary_name

    history = []
    best_val = float("inf")
    serializable_args = {}
    for key, value in vars(args).items():
        serializable_args[key] = str(value) if isinstance(value, Path) else value

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            total=resolve_progress_total(train_loader, args.max_train_batches),
        )
        for queries, pos_imgs, neg_imgs, _ in pbar:
            q = model.encode_text(queries)
            p = model.encode_images(pos_imgs)
            n = model.encode_images(neg_imgs)

            loss_nce = info_nce_loss(q, p, temperature=args.temperature)
            loss_margin = hard_negative_margin_loss(q, p, n, margin=args.margin)
            loss = loss_nce + args.margin_weight * loss_margin

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "nce": f"{loss_nce.item():.4f}",
                    "margin": f"{loss_margin.item():.4f}",
                }
            )

            if args.max_train_batches is not None and total_batches >= args.max_train_batches:
                break

        pbar.close()

        train_loss = total_loss / max(total_batches, 1)
        val_loss = evaluate(
            model,
            val_loader,
            temperature=args.temperature,
            margin=args.margin,
            margin_weight=args.margin_weight,
            max_batches=args.max_val_batches,
        )

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_retriever_checkpoint(
                best_path,
                model,
                extra_state={
                    "best_val_loss": best_val,
                    "train_qids": train_qids,
                    "val_qids": val_qids,
                    "training_args": serializable_args,
                },
            )
            print(f"Saved best checkpoint to: {best_path}")

    summary_payload = {
        "train_file": str(args.train_file),
        "checkpoint_path": str(best_path),
        "best_val_loss": best_val,
        "pooling": args.pooling,
        "used_weighted_sampler": args.use_weighted_sampler,
        "dataset_summary": summary,
        "history": history,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print("\nTraining done.")
    print(f"Best val loss: {best_val}")
    print(f"Checkpoint: {best_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
