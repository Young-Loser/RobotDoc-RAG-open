# Release Notes

## v1.0.0

RobotDoc-RAG-open v1.0.0 is the first public open-source snapshot of this project.

### Highlights

- page-level multimodal RAG baseline for robot manuals and datasheets
- end-to-end rebuild pipeline from raw PDFs to cleaned page index
- BM25, visual retrieval, and two-stage multimodal retrieval comparison
- SigLIP dual-encoder training and evaluation scripts
- multi-page visual generation with multiple answer-selection strategies
- bilingual project entry docs and polished release materials

### Included

- repository reorganization into clear pipeline stages
- unified pipeline entry via `run_pipeline.py`
- rebuild outputs and retrieval reports suitable for inspection
- documentation for pipeline, training, evaluation, and generation

### Intentionally Excluded

- raw PDF manuals
- rendered page image cache
- heavy vision cache files
- training checkpoints
- large runtime scratch outputs

### Recommended First Look

1. `README.md`
2. `README.zh-CN.md`
3. `docs/release_v1.0.0.md`
4. `outputs/reports/retrieval_comparison_summary.json`
5. `outputs/two_stage_rerank_eval_details.csv`
