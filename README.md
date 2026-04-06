# RobotDoc-RAG-open

RobotDoc-RAG-open is a page-level multimodal RAG project for robot manuals, datasheets, and other mixed-layout technical documents.

It focuses on a practical problem: pure OCR-first retrieval often misses answers that depend on page layout, figures, tables, and visual grounding. This repository keeps document pages as first-class multimodal units and compares OCR retrieval, visual retrieval, and a stronger two-stage multimodal pipeline.

## What This Repository Includes

- document rebuild from raw PDFs
- OCR extraction and cleaned page indexing
- BM25 and page-image retrieval baselines
- SigLIP-based page retriever training
- two-stage retrieval with BM25 recall and visual reranking
- multi-page visual answer generation
- reproducible outputs, reports, and project docs

## Current Best Retrieval Result

On the current 10-query evaluation split stored in `outputs/reports/retrieval_comparison_summary.json`, the strongest overall retrieval setup is the two-stage pipeline:

| Method | top1 exact acc | top5 exact recall | top1 doc acc | top5 doc recall |
|---|---:|---:|---:|---:|
| BM25 | 0.2 | 0.6 | 0.4 | 0.9 |
| SigLIP v1 | 0.3 | 0.5 | 0.5 | 0.7 |
| SigLIP v2 | 0.2 | 0.2 | 0.5 | 0.6 |
| Two-stage | **0.5** | **0.6** | **0.5** | **0.7** |

## Good Multimodal Two-Stage Cases

The repository already contains several strong two-stage retrieval examples in `outputs/two_stage_rerank_eval_details.csv`.

### `q005` `workspace diagram`

- gold page: `fr3_product_manual / page 51`
- two-stage top-1 retrieval: exact hit
- downstream generation can describe both the side view and top view of the workspace figure

### `q009` `7 dof dimensions`

- gold page: `kinova_gen3_user_guide / page 66`
- two-stage top-1 retrieval: exact hit
- this is a good layout-heavy visual retrieval example

### `q010` `robot components`

- gold page: `kinova_gen3_user_guide / page 18`
- two-stage top-1 retrieval: exact hit
- downstream generation gives a clean component list including base, actuators, interface module, and vision module

### `q003` `degrees of freedom 7`

- gold page: `fr3_datasheet / page 1`
- two-stage top-1 retrieval: exact hit
- this is a compact datasheet-style fact retrieval case

## Repository Layout

- `configs/`: pipeline configs
- `data/`: manifests, OCR tables, train/eval CSVs
- `docs/`: workflow notes and project showcase docs
- `outputs/`: reports, summaries, and sample generation results
- `scripts/`: runnable scripts grouped by stage
- `src/`: shared project modules

## Environment Setup

Recommended Python version:

- Python `3.10`

Recommended setup:

```bash
conda create -n robotdoc-rag python=3.10 -y
conda activate robotdoc-rag
pip install -r requirements-open-source.txt
```

Notes:

- `torch` may need to match your CUDA version
- `paddlex` is needed for OCR rebuild
- `colpali-engine` is needed for the vision retrieval baseline
- `qwen-vl-utils` is needed for the generation script
- this repository does not bundle raw manuals, rendered page images, or large embedding caches by default
- to rebuild the pipeline on your own documents, place PDFs under `data/raw_pdfs/` and rerun the rebuild step

If you only want to inspect the current reports and showcase outputs, you do not need to rerun the full pipeline.

## Quick Start

### 1. Inspect available pipelines

```bash
python run_pipeline.py --list
```

### 2. Rebuild page data from raw PDFs

```bash
python run_pipeline.py data_rebuild
```

This generates:

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

Sanity check:

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

```bash
python scripts/generation/minimal_multimodal_generator.py --query-ids q005 q010 --topk-pages 3
```

Or:

```bash
python scripts/generation/minimal_multimodal_generator.py --limit 3
```

## Suggested Demo Path

If you want to understand the project quickly:

1. open `outputs/reports/retrieval_comparison_summary.json`
2. inspect `outputs/two_stage_rerank_eval_details.csv`
3. inspect `outputs/generator_cases/multistrategy_generator_results.json`
4. open `docs/project_showcase.md`
5. run `python scripts/generation/minimal_multimodal_generator.py --query-ids q005 q010 --topk-pages 3`

## Documentation

- `docs/README.md`
- `docs/pipeline_overview.md`
- `docs/retriever_training.md`
- `docs/evaluation_workflow.md`
- `docs/generation_workflow.md`
- `docs/project_showcase.md`
- `RELEASE_NOTES.md`

## Current Scope

This repository is a strong experimental baseline for:

- robot document page retrieval
- OCR vs image retrieval comparison
- two-stage multimodal retrieval
- multi-page visual answer generation

Natural next steps:

- improve fact-style retrieval quality
- add a user-facing demo or service wrapper
- expand the benchmark set
