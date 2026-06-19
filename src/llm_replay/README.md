# `src/llm_replay/` — shared research code

Reusable code the research program in
[`research/dynamic-temperature-generative-replay.md`](../../research/dynamic-temperature-generative-replay.md)
shares across study areas. The repo is **not installed as a package** (`[tool.uv] package = false`),
so consumers add `src/` to `sys.path` (the scripts do this) rather than `pip install`-ing it.

Maps to the doc's "Shared Infrastructure" table:

| Module        | Purpose                                                          | Status | Used by |
|---------------|------------------------------------------------------------------|--------|---------|
| `metrics/`    | Distribution/diversity collapse metrics (MAUVE, Vendi, prdc, Self-BLEU, Distinct-n, Fréchet, MMD, KL, tail-mass) | **built** — `metrics/diversity.py` | All (esp. 2, 5) |
| `generation/` | EDT + sampling strategies, prefix-only generation, vLLM pipeline | planned | All |
| `training/`   | LoRA fine-tune loop, recursive-generation driver                 | planned | 2, 5 |
| `data/`       | Corpus load / version / mix helpers                              | planned | All |
| `eval/`       | Capability scoring lives in `scripts/evaluate.py` + `eval_tasks/` (lm-eval) | via lm-eval | All |

`metrics/diversity.py` is driven by [`scripts/diversity.py`](../../scripts/diversity.py); its
functions are split into reference-free (diversity declines under collapse) and reference-based
(distance from real/Gen-0 grows). Validated by smoke runs, not yet a committed `pytest` suite —
see [`tests/`](../../tests/).

When the generation/training modules land and the code is worth installing, this is where to flip
`pyproject.toml` to package mode (add a `[build-system]`) and drop the `sys.path` shims.
