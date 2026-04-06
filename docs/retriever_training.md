# Retriever Training

This stage upgrades the project from separate experiment scripts to a shared page-level dual-encoder training stack.

## Main Components

- `src/robotdoc_rag/retriever.py`
  Shared dual-encoder model, pooling logic, losses, and checkpoint helpers.

- `src/robotdoc_rag/training_data.py`
  Shared training-pair loading, splitting, batching, and dataset summary helpers.

- `scripts/training/train_dual_encoder.py`
  Unified training entrypoint with configurable train file, pooling strategy, weighted sampling, and smoke-test controls.

## Pooling

Supported pooling strategies:

- `cls`
- `position_weighted`

The `position_weighted` strategy is intended to better match the resume goal of improving page-level representation quality beyond a plain CLS token.

## Recommended Checks

Dataset summary only:

```bash
python3 scripts/training/train_dual_encoder.py --describe-only
```

Short smoke run:

```bash
python3 scripts/training/train_dual_encoder.py \
  --epochs 1 \
  --batch-size 2 \
  --max-train-batches 1 \
  --max-val-batches 1 \
  --use-weighted-sampler
```

Legacy-compatible presets:

```bash
python3 scripts/training/train_siglip_retriever_v1.py
python3 scripts/training/train_siglip_retriever_v2.py
```
