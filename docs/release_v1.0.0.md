# RobotDoc-RAG-open v1.0.0

## Public Release Summary

RobotDoc-RAG-open v1.0.0 is the first public snapshot of a page-level multimodal RAG project for robot manuals, datasheets, and mixed-layout technical documents.

This release packages the project as a reproducible open-source baseline with:

- document rebuild from raw PDFs to page-level structured data
- OCR extraction and cleaned page indexing
- BM25, visual retrieval, and two-stage multimodal retrieval
- SigLIP-based dual-encoder training and evaluation
- multi-page visual generation strategies
- reports, summaries, and showcase examples for direct inspection

## Why This Release Matters

Many technical-document QA pipelines rely too heavily on OCR text alone. That works for some fact-style questions, but it often breaks on:

- tables and dense specification sheets
- page-layout-sensitive questions
- diagrams, component maps, and workspace figures
- pages where OCR noise distorts retrieval quality

RobotDoc-RAG-open is built around the idea that page images should remain first-class retrieval units, not just intermediate OCR sources.

## Current Highlight

On the current 10-query evaluation split, the strongest overall retrieval configuration is the two-stage system:

- BM25 top-1 exact acc: `0.2`
- SigLIP v1 top-1 exact acc: `0.3`
- SigLIP v2 top-1 exact acc: `0.2`
- Two-stage top-1 exact acc: `0.5`

This makes the current repository snapshot a practical baseline for demonstrating multimodal page retrieval on real technical documents.

## Included In v1.0.0

- reorganized repository structure
- unified pipeline entry through `run_pipeline.py`
- page rebuild and OCR cleaning pipeline
- dual-encoder training scripts
- retrieval evaluation and failure analysis tools
- multi-strategy generation script
- polished README, bilingual docs entry, and release notes

## Not Included

To keep the repository lightweight and open-source friendly, this release does not bundle:

- raw PDF manuals
- rendered page-image cache
- heavy vision embedding caches
- training checkpoints
- large temporary runtime artifacts

## Recommended Assets To Inspect

For a quick review of the release, look at:

1. `README.md`
2. `README.zh-CN.md`
3. `outputs/reports/retrieval_comparison_summary.json`
4. `outputs/two_stage_rerank_eval_details.csv`
5. `outputs/generator_cases/multistrategy_generator_results.json`

## Suggested GitHub Release Title

`RobotDoc-RAG-open v1.0.0: First Public Multimodal RAG Baseline`
