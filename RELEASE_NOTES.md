# Release Notes

## Latest Open-Source Snapshot

This repository snapshot is the cleaned open-source baseline derived from the internal project iteration.

### Included

- repository reorganization into clear pipeline stages
- unified pipeline entry via `run_pipeline.py`
- page rebuild from raw PDFs to OCR-cleaned page index
- SigLIP dual-encoder training and evaluation scripts
- two-stage retrieval evaluation outputs
- multi-strategy visual generation script
- ready-to-read reports and showcase documentation

### Intentionally Excluded

- raw PDF manuals
- rendered page image cache
- heavy vision cache files
- training checkpoints
- large runtime scratch outputs

### Recommended First Look

Start with:

1. `README.md`
2. `outputs/reports/retrieval_comparison_summary.json`
3. `outputs/two_stage_rerank_eval_details.csv`
4. `outputs/generator_cases/multistrategy_generator_results.json`
