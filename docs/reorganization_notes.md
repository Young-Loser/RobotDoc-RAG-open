# Reorganization Notes

This note records how the repository was cleaned up so future changes can stay consistent.

## What Changed

- Split the original flat `scripts/` folder into task-oriented subfolders.
- Removed the empty `artifacts/generation/` directory.
- Added a root `README.md` and this note for orientation.
- Updated all scripts to derive `ROOT` from the current file location.

## Current Script Mapping

### `scripts/data_preparation/`

- `render_pdfs.py`
- `run_ocr_baseline.py`
- `clean_ocr_texts.py`

### `scripts/data_curation/`

- `generate_query_candidates.py`
- `filter_candidate_pages.py`
- `refine_candidate_pages.py`
- `filter_query_candidates.py`
- `label_query_gold.py`
- `build_retriever_trainset.py`
- `build_large_retriever_trainset.py`
- `build_vision_hard_negatives.py`
- `merge_large_trainset_v2.py`
- `refine_large_trainset_v3.py`

### `scripts/retrieval/`

- `bm25_retrieval.py`

### `scripts/training/`

- `train_siglip_retriever_v1.py`
- `train_siglip_retriever_v2.py`

### `scripts/evaluation/`

- `eval_bm25.py`
- `eval_siglip_retriever_v1.py`
- `eval_siglip_retriever_v2.py`
- `eval_two_stage_rerank.py`
- `vision_retrieval_full.py`
- `vision_retrieval_smoke.py`

### `scripts/generation/`

- `minimal_multimodal_generator.py`

## Directory Intent

- `data/raw_pdfs/`: original manuals and datasheets
- `data/pages/`: page images rendered from PDFs
- `data/ocr/`: OCR raw json and cleaned text tables
- `data/eval/`: gold evaluation queries
- `data/train/`: candidate tables and retriever training pairs
- `outputs/checkpoints/`: trained model weights
- `outputs/siglip_cache/`: cached embeddings for SigLIP experiments
- `outputs/vision_cache/`: cached embeddings for vision retrieval experiments
- `outputs/generator_cases/`: generated demo images and outputs

## Naming Guidance

- Put new runnable scripts under the matching `scripts/<category>/` folder.
- Keep filenames action-oriented, for example `build_*`, `train_*`, `eval_*`, `run_*`.
- Keep version suffixes only when the repository must preserve multiple experiment variants.
