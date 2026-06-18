# `notebooks/` — exploration & figures

Jupyter notebooks for interactive exploration, ad-hoc analysis, and paper figures
(collapse-trajectory plots, distribution-matching visualizations). The kernel
**"Python (llm-research)"** is pre-registered by `.devcontainer/post-create.sh`.

Guidelines:

- Notebooks are for exploration and visualization — promote any reusable logic into
  [`../src/llm_replay/`](../src/llm_replay/) so it can be tested and shared.
- Heavy outputs (generated corpora, checkpoints) belong in [`../runs/`](../runs/) /
  [`../data/`](../data/), not committed inside notebooks.
- Consider pairing notebooks with `jupytext` `.py` files for clean diffs if review matters.
