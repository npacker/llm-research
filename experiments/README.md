# `experiments/` — per-study runners

One subdirectory per study from
[`research/dynamic-temperature-generative-replay.md`](../research/dynamic-temperature-generative-replay.md),
plus a cross-area folder:

```
experiments/
├── area1_temperature_granularity/   # token vs sequence vs hybrid vs fixed
├── area2_curriculum_scheduling/
├── area3_task_adaptive/
├── area4_prefix_x_temperature/      # 16-cell factorial
├── area5_model_collapse/            # recursive-generation driver
└── cross_area_synthesis/
```

Each runner should stay **thin**: load a config from [`../configs/`](../configs/), call into
[`../src/llm_replay/`](../src/llm_replay/) for generation / training / metrics, and write
artifacts to [`../runs/`](../runs/) (gitignored). Track runs with **wandb** (already a
dependency) so collapse trajectories and metric curves are logged centrally.

Start every study **pilot-first** (1–3B model, few generations, small sample counts) before
scaling — see the single-GPU feasibility note in the research doc.

**Status: nothing built here yet** — these runners depend on the generation / recursive-training
pipeline (`src/llm_replay/generation`, `training`), which doesn't exist. What *does* exist for the
runners to call once that lands: the capability eval ([`../scripts/evaluate.py`](../scripts/evaluate.py))
and the collapse-axis diversity metrics ([`../scripts/diversity.py`](../scripts/diversity.py)).
