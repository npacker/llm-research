# `runs/` — experiment outputs (gitignored)

Per-run artifacts: LoRA checkpoints, generated synthetic corpora, metrics, logs, and local
wandb state. **Everything here except this README is gitignored** — outputs are reproducible
from configs + code, and large artifacts should go to the HuggingFace Hub or wandb, not git.

What the built tools write here today:

```
runs/gen<N>_<config>_<ts>/eval/results.json   # scripts/evaluate.py (lm-eval capability battery)
runs/gen<N>_diversity_<ts>/diversity.json     # scripts/diversity.py (collapse-axis metrics)
```

Intended per-run layout once the generation/training pipeline lands (e.g. `runs/area5_edt_gen03_<id>/`):

```
runs/<run-id>/
├── config.yaml        # frozen copy of the config used
├── checkpoints/       # LoRA adapters per generation
├── samples/           # generated synthetic data
├── eval/              # capability battery (results.json)
├── metrics/           # collapse / diversity metrics (diversity.json)
└── logs/
```

Primary experiment tracking is **wandb** (already a dependency); `runs/` is the local mirror.
`runs/*` is gitignored except this README.
