# Pipeline Overview

This project is being upgraded from a loose collection of scripts into a reusable pipeline.

## Phase 1 Goals

- Add a single configuration source
- Add a single entrypoint for running multi-step jobs
- Standardize project directories
- Make later refactors easier without changing every script call manually

## Current Entrypoint

Run from the repository root:

```bash
python3 run_pipeline.py --list
python3 run_pipeline.py data_rebuild
```

## Current Config

The default config lives at `configs/default.json`.

It currently defines:

- `data_rebuild`: render PDFs, run OCR, clean OCR text, build unified page index
- `retrieval_baseline`: run BM25 evaluation

## Why This Matters

This is the engineering base for later stages:

- custom document rebuild
- unified retriever training
- comparable evaluation runs
- multi-page generation experiments

## Rebuild Artifacts

The rebuild stage now produces both document-level and page-level metadata:

- `data/documents.csv`
- `data/manifest.csv`
- `data/page_index.csv`
- `outputs/data_preparation/data_rebuild_summary.json`
