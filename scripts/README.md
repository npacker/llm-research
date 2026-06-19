# `scripts/` — CLI entrypoints

Command-line entrypoints that wrap [`../src/llm_replay/`](../src/llm_replay/) for the common
verbs of the research workflow:

- **`evaluate.py`** — capability/regression battery via lm-evaluation-harness — **built**
- **`diversity.py`** — distribution/diversity metrics over corpora (collapse axis) — **built**
- `serve` — just use `vllm serve <model> --host 0.0.0.0 --port 8000` directly (no wrapper needed)
- `generate` — produce synthetic replay data for a config — *planned* (needs the generation pipeline)
- `train` — LoRA fine-tune the next-generation model — *planned*

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

## `diversity.py` — distribution/diversity metrics (collapse axis)

The corpus-vs-corpus complement to `evaluate.py`: where `evaluate.py` scores model
*capability*, this measures the *distribution/diversity* axis that detects collapse directly
(the **primary** Area-5 signal). Compares a synthetic corpus against an optional real/Gen-0
reference (logic in [`../src/llm_replay/metrics/diversity.py`](../src/llm_replay/metrics/diversity.py)).

```sh
# Full panel: synthetic vs real/Gen-0 reference
python scripts/diversity.py --synthetic gen3.txt --real gen0.txt --generation 3

# Reference-free only (diversity of one corpus)
python scripts/diversity.py --synthetic gen3.txt --generation 3 --no-mauve
```

Corpus specs accept a `.txt` (one text/line), `.jsonl` (`--text-field`), or `hf:<dataset>:<split>:<field>`.
Metrics: **reference-free** (decline under collapse) — Distinct-n, Self-BLEU, Vendi, vocab size,
tail mass; **reference-based** (grow under collapse) — MAUVE, prdc precision/recall/density/coverage,
Fréchet embedding distance, RBF-MMD, unigram-KL. Results → `runs/gen<N>_diversity_<ts>/diversity.json`.
