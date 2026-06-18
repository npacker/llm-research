# `runs/` — experiment outputs (gitignored)

Per-run artifacts: LoRA checkpoints, generated synthetic corpora, metrics, logs, and local
wandb state. **Everything here except this README is gitignored** — outputs are reproducible
from configs + code, and large artifacts should go to the HuggingFace Hub or wandb, not git.

Suggested per-run layout (one dir per run, e.g. `runs/area5_edt_gen03_<id>/`):

```
runs/<run-id>/
├── config.yaml        # frozen copy of the config used
├── checkpoints/       # LoRA adapters per generation
├── samples/           # generated synthetic data
├── metrics/           # collapse indicators, diversity/quality metrics
└── logs/
```

Primary experiment tracking is **wandb** (already a dependency); `runs/` is the local mirror.
