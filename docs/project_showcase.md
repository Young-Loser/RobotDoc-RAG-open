# Project Showcase

## Summary

`robotdoc-rag` is a multimodal document question answering project built around robot manuals and datasheets.

The project works at the page level instead of treating the whole document as plain OCR text. This makes it better suited for:

- page diagrams
- visual layout cues
- mixed text-table-figure pages
- cross-page evidence aggregation

## End-to-End Pipeline

1. Input custom PDF manuals or datasheets
2. Render PDFs into page images
3. Run OCR and clean OCR text
4. Build document-level and page-level indices
5. Train and evaluate page retrievers
6. Use retrieved pages for multi-page visual answer generation

## Retrieval Design

The project includes several retrieval paths:

- OCR-first BM25
- single-stage page-image retrieval with SigLIP-based dual encoders
- two-stage retrieval: BM25 recall plus visual reranking
- vision retrieval baselines

The current strongest overall retrieval path is the two-stage setup.

## Generation Design

The generation stage compares multiple answer strategies:

- answer from the top page only
- answer from stitched retrieved pages
- answer each page separately and choose with weighted scoring
- answer jointly from multiple page images

This makes the repository useful not only for retrieval benchmarking, but also for studying how page aggregation affects answer quality.

## What Makes This Project Useful

- supports custom document rebuild
- keeps page images as first-class retrieval units
- exposes evaluation details and comparison reports
- supports failure analysis for retrieval
- supports multi-strategy visual answer generation

## Main Deliverables In The Repository

- standardized data rebuild outputs
- reusable retriever training entrypoints
- retrieval comparison reports
- failure case exports
- multi-strategy generation outputs

## Suggested Demo Path

If you want to present this project to someone else, the cleanest walkthrough is:

1. Show `data/documents.csv` and `data/page_index.csv`
2. Show `outputs/reports/retrieval_comparison_summary.json`
3. Show one or two retrieval failure reports
4. Show `outputs/generator_cases/multistrategy_generator_results.json`
5. Explain why page-level visual evidence matters for robot manuals
