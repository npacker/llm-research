# `scripts/` — CLI entrypoints

Command-line entrypoints that wrap [`../src/llm_replay/`](../src/llm_replay/) for the common
verbs of the research workflow:

- **`evaluate.py`** — capability/regression battery via lm-evaluation-harness — **built**
- **`diversity.py`** — distribution/diversity metrics over corpora (collapse axis) — **built**
- **`generate.py`** — EDT synthetic-data generation (fixed / seq_edt / token_edt) — **built**
- **`validate.py`** — quality validation of a generated corpus (gates + perplexity + panel) — **built**
- **`train.py`** — LoRA fine-tuning with domain/general/synthetic corpus mixes (forgetting/replay) — **built**
- **`forgetting_report.py`** — tabulate base-vs-condition eval deltas — **built**
- `serve` — just use `vllm serve <model> --host 0.0.0.0 --port 8000` directly (no wrapper needed)

## `train.py` / `forgetting_report.py` — LoRA fine-tuning + forgetting study

LoRA continued-LM fine-tuning over domain/general/synthetic corpus mixes; logic in
[`../src/llm_replay/training/`](../src/llm_replay/training/) (see its README for the condition matrix).
`train.py` also auto-reports **held-out domain perplexity** (base vs base+adapter) into `meta.json` —
the objective-matched domain-learning signal; the medical battery is the secondary transfer check.

```sh
python scripts/train.py --config configs/train/domain_synthetic.yaml --model Qwen/Qwen3.5-4B \
    --synthetic runs/gen1_*/clean.jsonl                    # -> runs/train_<cfg>_<ts>/adapter
# eval base + adapter via vLLM (NOT a merged model — see training/README.md):
python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm --lora runs/train_*/adapter \
    --lora-rank 16 --config configs/eval/canary.yaml --generation domain_synthetic
python scripts/forgetting_report.py --base base=runs/gen0_canary_*/eval/results.json \
    --runs domain_only=<results.json> domain_general=<results.json> ...
```

`evaluate.py --lora <adapter>` applies a LoRA adapter to the base (vllm `lora_local_path` / hf `peft`),
avoiding a merge. `train.py --merge` also writes a standalone merged checkpoint (HF/portability).

## `generate.py` / `validate.py` — generation + quality validation

EDT synthetic-data generation (research plan Area 1) and a validation pipeline; logic in
[`../src/llm_replay/generation/`](../src/llm_replay/generation/) (see its README).

```sh
# Generate (strategy comes from the config: fixed | seq_edt | token_edt)
python scripts/generate.py --config configs/gen/token_edt.yaml --model Qwen/Qwen3.5-0.8B --generation 1
# Validate the corpus (gates → perplexity → diversity panel vs a real/Gen-0 reference)
python scripts/validate.py --corpus runs/gen1_token_edt_<ts>/samples.jsonl --real <seed-spec> --generation 1
```

Generation → `runs/gen<N>_<cfg>_<ts>/{samples.jsonl, meta.json}`; validation →
`runs/gen<N>_validation_<ts>/{clean.jsonl, validation.json}`. Raw continuation (no chat template).

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
