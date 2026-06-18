# `scripts/` — CLI entrypoints

Command-line entrypoints that wrap [`../src/llm_replay/`](../src/llm_replay/) for the common
verbs of the research workflow:

- `serve` — start the vLLM OpenAI-compatible API (`vllm serve <model> --host 0.0.0.0 --port 8000`)
- `generate` — produce synthetic replay data for a given config
- `train` — LoRA fine-tune the next-generation model
- `evaluate` — run the benchmarking / collapse-indicator suite

Keep scripts thin: parse args (typically `--config <path>` into [`../configs/`](../configs/)),
call library functions, write to [`../runs/`](../runs/). When the package is set up, these can be
registered as `[project.scripts]` entry points in `pyproject.toml`.
