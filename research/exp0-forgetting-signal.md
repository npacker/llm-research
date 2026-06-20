# Experiment 0 — Forgetting-signal calibration & hyperparameter tuning

**Status:** designed; runs **before** [`exp1-forgetting-replay.md`](exp1-forgetting-replay.md).
**Why first:** Exp 1 (replay vs. forgetting) only has signal if domain-only fine-tuning *both*
(a) measurably **learns** the domain and (b) measurably **forgets** general capability. This
experiment finds a hyperparameter regime where that tradeoff exists, and gives a runtime estimate
for budgeting Exp 1's 5-condition matrix. If `domain_only` doesn't forget at any reasonable setting,
Exp 1 is premature.

## Question
At what training scale / LR / epochs does continued-LM LoRA on medical text produce a **clear
domain-learning gain *and* clear general forgetting** on Qwen3.5-4B (instruct)?

## Setup
- **Model:** `Qwen/Qwen3.5-4B` (instruct — decided; format drift is a *measured* variable here).
- **Condition:** `domain_only` (`configs/train/domain_only.yaml`) only — no replay.
- **Sweep** (one axis at a time from a center point r=16 / lr=2e-4 / 1 epoch):
  - scale: `--num-samples` ∈ {1k, 5k, 10k}
  - learning rate ∈ {1e-4, 2e-4, 5e-4}  (config `training.learning_rate`)
  - epochs ∈ {1, 3}                       (config `training.num_train_epochs`)
  - (optional) LoRA r ∈ {8, 16, 32}
  Sweeping LR/epochs/r needs per-setting config variants (train.py reads them from the config's
  `training`/`lora` blocks); scale uses `--num-samples`. *(A small `--lr`/`--epochs` CLI override on
  train.py would make this cleaner — optional tooling.)*

## Metrics (per run; both already wired)
- **Domain learning:** held-out medical **perplexity** Δ (base vs base+adapter) — auto in `meta.json`.
- **Forgetting:** general battery via `configs/eval/canary.yaml` (base+adapter, vLLM `--lora`), Δ vs.
  the base reference (`runs/genbase_canary_ref`).
- **Separate format vs. knowledge** (format-drift caveat): read **IFEval** (instruction/format
  following — most exposed to format drift) apart from **GSM8K / MMLU-Pro / GPQA** (knowledge/reasoning).
  Genuine catastrophic forgetting should show on the knowledge tasks, not only IFEval.

## Procedure (sketch)
```sh
# Base reference (shared with Exp 1) — already running:
#   runs/genbase_canary_ref/ , runs/genbase_medical_ref/
for n in 1000 5000 10000; do
  python scripts/train.py --config configs/train/domain_only.yaml --model Qwen/Qwen3.5-4B \
      --num-samples $n --output-dir runs/exp0_domain_only_n$n          # logs domain-ppl Δ to meta.json
  python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm --lora runs/exp0_domain_only_n$n/adapter \
      --lora-rank 16 --config configs/eval/canary.yaml --generation n$n --output-dir runs/exp0_eval_n$n
done
python scripts/forgetting_report.py --base base=runs/genbase_canary_ref/results.json \
    --runs n1000=... n5000=... n10000=...
# repeat for LR/epoch config variants
```

## Decision criteria (output of this experiment)
Pick the smallest setting where **domain ppl drops clearly** (e.g. ≳ a few %) **and** the general
battery shows a clear, attributable drop (ideally on knowledge tasks, not just IFEval). Use that
LR/epochs/scale as the fixed training recipe for **all** Exp 1 conditions. Record the per-run
wall-clock to budget Exp 1 (5 conditions × train + 2 eval batteries).

## Notes
- If forgetting only shows on IFEval (format) and not on knowledge tasks, that's the format-drift
  confound — note it, and prefer a setting that also moves knowledge tasks, or lengthen training.
- Single GPU; eval = base+adapter via vLLM (`--lora`), `enable_thinking: false`. Use full test sets
  for the final picked setting (sweeps can use a smaller eval subset to save time, but confirm the
  chosen point on full sets).
