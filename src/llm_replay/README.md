# `src/llm_replay/` — shared research infrastructure (planned)

This is the intended home for the **reusable code** the research program in
[`research/dynamic-temperature-generative-replay.md`](../../research/dynamic-temperature-generative-replay.md)
shares across all five study areas. It is **not a package yet** — the repo is still a pure
environment (`[tool.uv] package = false` in `pyproject.toml`). When shared code lands here,
flip `pyproject.toml` to package mode, add a `[build-system]`, and add tests under
[`tests/`](../../tests/).

Maps to the doc's "Shared Infrastructure" table:

| Planned module      | Purpose                                                          | Used by areas |
|---------------------|------------------------------------------------------------------|---------------|
| `generation/`       | EDT + sampling strategies, prefix-only generation, vLLM pipeline | All           |
| `metrics/`          | Self-BLEU, Distinct-n, MMD, KL, perplexity, collapse indicators  | All           |
| `training/`         | LoRA fine-tune loop, recursive-generation driver                 | 2, 5          |
| `data/`             | Corpus load / version / mix helpers                              | All           |
| `eval/`             | Benchmarking harness, LLM-as-judge wrappers                      | All           |

Keep numeric definitions (Distinct-n, KL direction, EDT temperature math) here so they can be
unit-tested once, rather than copy-pasted into notebooks.
