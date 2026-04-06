# Evaluation Workflow

This stage standardizes retrieval evaluation, cache validation, and failure analysis.

## Why It Was Updated

Two issues mattered for result quality:

- stale cached page embeddings could be reused after checkpoint changes
- evaluation outputs were fragmented across scripts and hard to compare

## Current Improvements

- checkpoint-aware cache validation for SigLIP retrieval evaluation
- failure-case CSV export for manual review
- cross-run comparison report generation
- cleaner score handling during evaluation

## Recommended Order

```bash
python scripts/evaluation/eval_bm25.py
python scripts/evaluation/eval_siglip_retriever_v1.py --force-reencode
python scripts/evaluation/eval_siglip_retriever_v2.py --force-reencode
python scripts/evaluation/eval_two_stage_rerank.py
python scripts/evaluation/compare_retrieval_runs.py
python scripts/evaluation/analyze_retrieval_failures.py
```

## Main Outputs

- `outputs/*_eval_details.csv`
- `outputs/*_eval_summary.json`
- `outputs/reports/retrieval_comparison.csv`
- `outputs/reports/retrieval_best_by_query.csv`
- `outputs/reports/*_failures.csv`
