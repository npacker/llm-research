# `src/llm_replay/` — shared research code

Reusable code the research program in
[`research/dynamic-temperature-generative-replay.md`](../../research/dynamic-temperature-generative-replay.md)
shares across study areas. The repo is **not installed as a package** (`[tool.uv] package = false`),
so consumers add `src/` to `sys.path` (the scripts do this) rather than `pip install`-ing it.

Maps to the doc's "Shared Infrastructure" table:

| Module        | Purpose                                                          | Status | Used by |
|---------------|------------------------------------------------------------------|--------|---------|
| `metrics/`    | Distribution/diversity collapse metrics (MAUVE, Vendi, prdc, Self-BLEU, Distinct-n, Fréchet, MMD, KL, tail-mass) | **built** — `metrics/diversity.py` | All (esp. 2, 5) |
| `generation/` | EDT temperature (token/seq/fixed) + prefix-only prompts + vLLM generator + quality validation | **built** — see `generation/README.md` | 1, 4 (→ 2, 5) |
| `corpus.py`   | Shared corpus loader (file / `hf:` spec, optional dataset config) | **built** | All |
| `training/`   | LoRA continued-LM fine-tuning + corpus mixing (forgetting/replay) | **built** — see `training/README.md` | 2, 5 |
| `eval/`       | Capability scoring lives in `scripts/evaluate.py` + `eval_tasks/` (lm-eval) | via lm-eval | All |

`metrics/diversity.py` is driven by [`scripts/diversity.py`](../../scripts/diversity.py); its
functions are split into reference-free (diversity declines under collapse) and reference-based
(distance from real/Gen-0 grows). Validated by smoke runs, not yet a committed `pytest` suite —
see [`tests/`](../../tests/).

When the generation/training modules land and the code is worth installing, this is where to flip
`pyproject.toml` to package mode (add a `[build-system]`) and drop the `sys.path` shims.
