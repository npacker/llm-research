# `exp0_forgetting_signal/` — forgetting-signal calibration sweep

The **Experiment 0** runner: a domain-only (no-rehearsal) LoRA hyperparameter sweep on
Qwen3.5-4B that calibrates *where* fine-tuning measurably learns the medical domain while
forgetting general capability — the recipe Exp 1 (generative replay) builds on. Full design
in [`../../research/exp0-forgetting-signal.md`](../../research/exp0-forgetting-signal.md).

This is a thin orchestrator: it loads [`../../configs/sweep/exp0.yaml`](../../configs/sweep/exp0.yaml),
shells out to the shared tools in [`../../scripts/`](../../scripts/) (`train.py`, `evaluate.py`,
`coherence_check.py`), and reuses the reusable primitives in `src/` — grid + Pareto from
[`llm_core.sweep`](../../src/llm_core/sweep.py), the trade-off score from
[`llm_replay.forgetting`](../../src/llm_replay/forgetting.py), and result deltas from
[`llm_core.evaluation`](../../src/llm_core/evaluation.py). Artifacts land in `runs/exp0/`.

## Stages

- **train** (Stage A, all points) — one LoRA per grid point; logs domain ppl Δ (learning)
  and general held-out ppl Δ (cheap forgetting proxy).
- **shortlist** (gate) — Pareto front on (domain Δ, general Δ), capped at `shortlist_k`.
- **stageb** (Stage B, shortlist) — serve base + all shortlisted adapters from **one** vLLM
  init; run the regression battery (lm-eval `local-completions`) + the coherence probe
  against it. Resumable per point.
- **report** — aggregate to `runs/exp0/summary.{json,md}` with the proxy and best-case scores.

## Usage

```sh
# End to end (train all -> shortlist -> Stage B on shortlist -> report):
python experiments/exp0_forgetting_signal/sweep.py --phase all

# One phase at a time (phases are independent + resumable):
python experiments/exp0_forgetting_signal/sweep.py --phase train
python experiments/exp0_forgetting_signal/sweep.py --phase report

# Print the grid (and the manual multi-LoRA `vllm serve` command) without running:
python experiments/exp0_forgetting_signal/sweep.py --phase plan --emit-serve-cmd

# Reuse a server you launched yourself, or force per-point in-process engines:
python experiments/exp0_forgetting_signal/sweep.py --phase stageb --base-url http://localhost:8000
python experiments/exp0_forgetting_signal/sweep.py --phase all --no-serve
```

Single GPU only. Confirm the winning point on the **full** test sets afterwards (drop
`--limit`; add `configs/eval/full_local.yaml` + `medical.yaml`).
