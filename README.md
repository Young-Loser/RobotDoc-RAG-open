# robotdoc-rag

`robotdoc-rag` is a page-level multimodal RAG pipeline for robot manuals, datasheets, and other mixed-layout technical documents.

This open-source version focuses on a practical question: how do we retrieve and answer from technical pages where the answer depends on layout, figures, tables, and page-level visual evidence rather than OCR text alone?

The repository includes:

- document rebuild from raw PDFs
- OCR extraction and cleaned page indexing
- OCR-based and page-image retrieval baselines
- SigLIP-based page retriever training
- two-stage retrieval with BM25 recall and visual reranking
- multi-page visual answer generation

## Why This Project Exists

Pure OCR-first retrieval is often weak on:

- tables and spec sheets
- diagrams and layout-heavy pages
- visual references such as component maps or workspace figures
- cases where OCR noise corrupts retrieval quality

`robotdoc-rag` keeps page images as first-class retrieval units and compares text-first, image-first, and two-stage retrieval pipelines.

## Current Best Result

On the current 10-query evaluation split stored in `outputs/reports/retrieval_comparison_summary.json`, the strongest overall retrieval path is the two-stage setup:

| Method | top1 exact acc | top5 exact recall | top1 doc acc | top5 doc recall |
|---|---:|---:|---:|---:|
| BM25 | 0.2 | 0.6 | 0.4 | 0.9 |
| SigLIP v1 | 0.3 | 0.5 | 0.5 | 0.7 |
| SigLIP v2 | 0.2 | 0.2 | 0.5 | 0.6 |
| Two-stage | **0.5** | **0.6** | **0.5** | **0.7** |

## Good Multimodal Two-Stage Examples

The repository already contains strong examples from the two-stage retrieval output in `outputs/two_stage_rerank_eval_details.csv`.

### Example 1: `q005` `workspace diagram`

- gold page: `fr3_product_manual / page 51`
- two-stage top-1 retrieval: exact hit
- downstream multi-image generation can describe the workspace side view and top view

### Example 2: `q009` `7 dof dimensions`

- gold page: `kinova_gen3_user_guide / page 66`
- two-stage top-1 retrieval: exact hit
- this is a good layout-heavy visual retrieval case

### Example 3: `q010` `robot components`

- gold page: `kinova_gen3_user_guide / page 18`
- two-stage top-1 retrieval: exact hit
- downstream generation gives a clean component list including base, actuators, interface module, and vision module

### Example 4: `q003` `degrees of freedom 7`

- gold page: `fr3_datasheet / page 1`
- two-stage top-1 retrieval: exact hit
- this is a compact fact-style datasheet retrieval case

## Repository Structure

- `configs/`
  pipeline configs
- `data/`
  raw PDFs, rendered pages, OCR outputs, training data, evaluation data
- `docs/`
  workflow notes and project showcase docs
- `outputs/`
  checkpoints, retrieval caches, reports, generated outputs
- `scripts/`
  runnable scripts by stage
- `src/`
  shared project modules

## Environment Setup

Recommended Python version:

- Python `3.10`

Recommended setup with conda:

```bash
conda create -n robotdoc-rag python=3.10 -y
conda activate robotdoc-rag
pip install -r requirements-open-source.txt
```

Notes:

- `torch` installation may need to match your CUDA version
- `paddlex` is needed for the OCR rebuild stage
- `colpali-engine` is needed for the vision retrieval baseline scripts
- `qwen-vl-utils` is needed for the generation script
- this open-source repository does not bundle raw manuals, rendered page images, or large embedding caches by default
- to rebuild the full pipeline on your own data, place PDFs under `data/raw_pdfs/` and rerun the rebuild step

If you only want to inspect the current outputs and reports, you do not need to rerun the full pipeline.

## Quick Start

### 1. Inspect available pipelines

```bash
python run_pipeline.py --list
```

### 2. Rebuild page data from raw PDFs

```bash
python run_pipeline.py data_rebuild
```

This will generate:

- `data/documents.csv`
- `data/manifest.csv`
- `data/ocr/page_texts.csv`
- `data/ocr/page_texts_clean.csv`
- `data/page_index.csv`

### 3. Build optimized training pairs

```bash
python scripts/data_curation/build_retriever_trainset_v4.py
```

### 4. Train the page retriever

Dataset sanity check:

```bash
python scripts/training/train_dual_encoder.py --describe-only
```

Main training entry:

```bash
python scripts/training/train_siglip_retriever_v2.py
```

### 5. Evaluate retrieval

```bash
python scripts/evaluation/eval_bm25.py
python scripts/evaluation/eval_siglip_retriever_v1.py --force-reencode
python scripts/evaluation/eval_siglip_retriever_v2.py --force-reencode
python scripts/evaluation/eval_two_stage_rerank.py
python scripts/evaluation/compare_retrieval_runs.py
python scripts/evaluation/analyze_retrieval_failures.py
```

### 6. Run multi-page generation

Small example:

```bash
python scripts/generation/minimal_multimodal_generator.py --query-ids q005 q010 --topk-pages 3
```

Or:

```bash
python scripts/generation/minimal_multimodal_generator.py --limit 3
```

## Main Workflows

### Data Preparation

- `scripts/data_preparation/render_pdfs.py`
- `scripts/data_preparation/run_ocr_baseline.py`
- `scripts/data_preparation/clean_ocr_texts.py`
- `scripts/data_preparation/build_page_index.py`

### Training Data

- `scripts/data_curation/build_retriever_trainset.py`
- `scripts/data_curation/build_retriever_trainset_v4.py`
- `scripts/data_curation/build_large_retriever_trainset.py`
- `scripts/data_curation/build_vision_hard_negatives.py`

### Retrieval

- `scripts/evaluation/eval_bm25.py`
- `scripts/evaluation/eval_siglip_retriever_v1.py`
- `scripts/evaluation/eval_siglip_retriever_v2.py`
- `scripts/evaluation/eval_two_stage_rerank.py`

### Analysis

- `scripts/evaluation/compare_retrieval_runs.py`
- `scripts/evaluation/analyze_retrieval_failures.py`

### Generation

- `scripts/generation/minimal_multimodal_generator.py`

## Key Output Files

Retrieval:

- `outputs/reports/retrieval_comparison_summary.json`
- `outputs/reports/retrieval_comparison.csv`
- `outputs/reports/retrieval_best_by_query.csv`
- `outputs/reports/siglip_retriever_v2_failures.csv`

Generation:

- `outputs/generator_cases/multistrategy_generator_results.json`

Rebuild:

- `data/documents.csv`
- `data/page_index.csv`

## Suggested Demo Path

If you want to demo the repository quickly:

1. open `data/page_index.csv`
2. open `outputs/reports/retrieval_comparison_summary.json`
3. inspect `outputs/two_stage_rerank_eval_details.csv`
4. run `python scripts/generation/minimal_multimodal_generator.py --query-ids q005 q010 --topk-pages 3`
5. inspect `outputs/generator_cases/multistrategy_generator_results.json`

## Documentation

- `docs/pipeline_overview.md`
- `docs/retriever_training.md`
- `docs/evaluation_workflow.md`
- `docs/generation_workflow.md`
- `docs/project_showcase.md`

## Current Scope

This repository is a strong experimental and reproducible project baseline for:

- robot document page retrieval
- OCR vs image retrieval comparison
- two-stage multimodal retrieval
- multi-page visual answer generation

If you want to extend it further, the most natural next steps are:

- improve fact-style retrieval quality
- add a user-facing demo or service wrapper
- expand the benchmark set
