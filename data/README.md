# `data/` — corpora

Real reference corpora and small sample/fixture datasets used by the studies (Generation-0 real
data, held-out real test sets, task-labeled corpora for Area 3).

- **Large files are gitignored** (see [`../.gitignore`](../.gitignore)) — this README is kept via
  a negation rule. Do not commit weights or large datasets.
- **Version generated corpora and large datasets via the HuggingFace Hub** (per
  [`../CLAUDE.md`](../CLAUDE.md)), not git. Use `hf-transfer` for fast up/download.
- Generated synthetic data from runs lands in [`../runs/`](../runs/); curated/published datasets
  that are worth versioning can be referenced here by Hub ID.
