# `scripts/` — CLI entrypoints

Command-line entrypoints that wrap [`../src/llm_replay/`](../src/llm_replay/) for the common
verbs of the research workflow:

- `serve` — start the vLLM OpenAI-compatible API (`vllm serve <model> --host 0.0.0.0 --port 8000`)
- `generate` — produce synthetic replay data for a given config
- `train` — LoRA fine-tune the next-generation model
- **`evaluate.py`** — capability/regression battery via lm-evaluation-harness *(implemented)*

Keep scripts thin: parse args (typically `--config <path>` into [`../configs/`](../configs/)),
call library functions, write to [`../runs/`](../runs/). When the package is set up, these can be
registered as `[project.scripts]` entry points in `pyproject.toml`.

## `evaluate.py` — capability regression battery

Runs a fixed [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) battery
(still the 2026 gold-standard harness; the **Open LLM Leaderboard itself was retired**, but its
non-saturated battery survives as the harness's built-in `leaderboard` task group). The goal is
**regression detection** across recursive generations: hold the config fixed and compare each
generation's scores to the Gen-0 baseline.

```sh
# Cheap canary every generation (verifiable, judge-free: GSM8K + IFEval + GPQA)
python scripts/evaluate.py --config configs/eval/canary.yaml --model <id-or-path> --generation 0

# Full non-saturated battery at milestones (MMLU-Pro, GPQA, BBH, MuSR, MATH-hard, IFEval)
python scripts/evaluate.py --config configs/eval/full.yaml --model <id-or-path> --generation 0

# Evaluate a model already running under `vllm serve` (OpenAI-compatible endpoint)
python scripts/evaluate.py --config configs/eval/canary.yaml --backend local-completions \
    --model <id> --base-url http://localhost:8000/v1/completions

# Smoke test (2 items/task)
python scripts/evaluate.py --config configs/eval/canary.yaml --model <id> --limit 2
```

Backends: `hf` (load weights directly, default), `vllm` (in-process, pins `tensor_parallel_size=1`
for the single GPU), `local-completions` (hit a running `vllm serve`). Results land in
`runs/gen<N>_<config>_<ts>/eval/results.json`; pass `--wandb-project <name>` to log the summary.

**Not covered here:** code/agentic benchmarks (**SWE-bench**, **LiveCodeBench**) — these need their
own separate harnesses (Docker / time-windowed) and should be run as dedicated milestone jobs.
