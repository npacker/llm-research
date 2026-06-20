# Experiment 0 — Forgetting-signal calibration & best-case no-rehearsal recipe

**Status:** implemented (tooling in `scripts/sweep_exp0.py`); runs **before**
[`exp1-forgetting-replay.md`](exp1-forgetting-replay.md).
**Why first:** Exp 1 (replay vs. forgetting) only has signal if domain-only fine-tuning *both*
(a) measurably **learns** the domain and (b) measurably **forgets** general capability. This
experiment finds that regime **and** picks the best-case recipe that learns the domain while
*minimizing* forgetting — both with **no data rehearsal** (`domain_only` mix). That recipe is the
fixed training recipe for all Exp 1 conditions, and the no-rehearsal baseline the later
rehearsal / diversity work (generative replay + EDT) builds on. It also budgets Exp 1's wall-clock.

## Question
At what training scale / LR / epochs / LoRA rank does continued-LM LoRA on medical text produce a
**clear domain-learning gain** while keeping **general forgetting and degeneration minimal** on
Qwen3.5-4B (instruct), without rehearsal?

## Dual goal (output of this experiment)
1. **Signal calibration** — confirm forgetting is *measurable* on **knowledge** tasks (not only
   format/IFEval) at some setting, so Exp 1 isn't premature.
2. **Best-case recipe** — the setting that maximizes **domain learning per unit knowledge
   forgetting** while staying **coherent** (non-degenerate). Pick the smallest such setting.

## Two-stage design (cheap proxy → expensive checks)
The expensive signals (regression battery, coherence generation) must **not** run on all ~20
points. They can't be skipped either — forgetting is the dependent variable, and a domain-learning
proxy alone (val/train loss, domain ppl) rewards exactly the most-forgetting corner. The resolution
is a cheap **forgetting proxy** available at training time:

- **Stage A — all points (cheap, no battery / no vLLM):** train each point with a *dual* validation
  set, `{domain, general}`. The Trainer logs `eval_domain_loss` (learning) and `eval_general_loss`
  (a held-out general corpus the run never trains on → forgetting). `eval_on_start` gives the
  base-model losses, so both **perplexity Δ** (base → best checkpoint) come straight off the curve.
- **Gate — Pareto shortlist:** keep the non-dominated front on (domain ppl Δ, general ppl Δ),
  capped at `shortlist_k`. Points that didn't learn (domain Δ ≥ 0) are excluded.
- **Stage B — shortlist only (expensive):** regression battery + coherence on the ~K survivors.
  This also **validates the proxy** — does general-ppl forgetting track *task*/knowledge forgetting?
  (Best-model selection / early stopping track `eval_domain_loss`, never the general loss, which
  only rises.)

## Setup
- **Model:** `Qwen/Qwen3.5-4B` (instruct — format drift is a *measured* variable here).
- **Condition:** `domain_only` (`configs/train/domain_only.yaml`) only — **no replay/rehearsal**.
- **Method:** LoRA (the research program is PEFT/EDT-based). Full fine-tuning is a deferred
  follow-up (see *Notes*).
- **Sweep (small grid, `configs/sweep/exp0.yaml`):** main grid LR ∈ {1e-4, 2e-4, 5e-4} × epochs ∈
  {1, 3} × scale (`--num-samples`) ∈ {1k, 5k, 10k} at rank 16 (18 points), plus a rank ∈ {8, 16, 32}
  sub-sweep at one center (2e-4 / 1 epoch / 5k) → ≈ 20 points. **Early stopping on validation loss**
  is on for every point (a free, no-rehearsal anti-forgetting lever). One grid point = one
  `train.py` invocation via CLI overrides (`--lr/--epochs/--lora-rank/...`); no per-setting configs.

## Metrics (per run)
- **Training dynamics (Stage A):** **train + dual validation loss** curves (`meta.json`:
  `loss_curve` with `train_loss`, `eval_domain_loss`, `eval_general_loss`) and `val_perplexity`
  (= exp best `eval_domain_loss`) — overfit/divergence shows here.
- **Domain learning, cheap (Stage A):** held-out **domain** perplexity Δ, base → best checkpoint,
  in `meta.json` `heldout_perplexity.domain.delta` (negative = learned).
- **Forgetting proxy, cheap (Stage A):** held-out **general** perplexity Δ,
  `heldout_perplexity.general.delta` (positive = forgetting). Drives the Pareto shortlist.
- **Forgetting, task-level (Stage B, shortlist):** `configs/eval/canary.yaml` (base+adapter, vLLM
  `--lora`) Δ vs. the base reference `runs/genbase_canary_ref/`. Reported **split**: **IFEval
  (format)** apart from **GSM8K / GPQA (knowledge/reasoning)** — genuine catastrophic forgetting
  should move knowledge, not only IFEval. (Confirms the cheap proxy.)
- **Failure modes (Stage B, shortlist):** `scripts/coherence_check.py` samples from base+adapter
  and scores per-sample gates (repetition, line-repeat, length) + distinct-2 + self-BLEU into
  `coherence.json` with a boolean `degenerate` verdict.
- **Wall-clock:** per-point train/eval seconds (`sweep_point.json`) — budgets Exp 1.

## Procedure
```sh
# 0) Base references (shared with Exp 1) — produce once, then keep as fixed snapshots:
python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm \
    --config configs/eval/canary.yaml  --generation base --output-dir runs/genbase_canary_ref
python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm \
    --config configs/eval/medical.yaml --generation base --output-dir runs/genbase_medical_ref

# 1) Run the sweep (phases are resumable; --dry-run prints the commands):
python scripts/sweep_exp0.py --phase all     # train(all) -> shortlist -> eval/coherence(shortlist) -> report
#   or step through:  --phase train | shortlist | eval | coherence | report
#   Stage A (train) runs on all ~20 points; the Pareto shortlist gates Stage B (eval/coherence)
#   to ~shortlist_k survivors. Inspect runs/exp0/shortlist.json before spending Stage-B compute.

# 2) Throughput option — amortize vLLM init across all points into ONE server:
python scripts/sweep_exp0.py --phase plan --emit-serve-cmd   # prints the vllm serve + eval cmds

# 3) Read the ranked best-case table:
cat runs/exp0/summary.md

# 4) Confirm the winner on FULL test sets (drop --limit; add knowledge + domain batteries):
python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm --lora runs/exp0/<winner>/adapter \
    --lora-rank <r> --config configs/eval/full_local.yaml --generation <winner>   # MMLU-Pro etc.
python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm --lora runs/exp0/<winner>/adapter \
    --lora-rank <r> --config configs/eval/medical.yaml  --generation <winner>     # domain transfer
```
During the sweep, `eval_limit` (config) caps items/task for speed; the winner is re-confirmed on
full sets. `sweep_exp0.py --phase report` writes `runs/exp0/summary.{json,md}` and a `best_case_score`
(domain gain per unit knowledge forgetting, **blank if degenerate**), sorted best-first.

## Decision criteria
Pick the **smallest** setting that:
1. drops domain ppl clearly (e.g. ≳ a few %), **and**
2. shows a clear, attributable forgetting Δ on **knowledge** tasks (satisfies goal 1), **and**
3. is **non-degenerate** (`coherence.json` `degenerate == false`), maximizing `best_case_score`.

Use that LR/epochs/scale/rank as the fixed recipe for **all** Exp 1 conditions; record per-run
wall-clock to budget Exp 1 (5 conditions × train + 2 eval batteries).

## Notes
- If forgetting only shows on IFEval (format) and not on knowledge tasks, that's the format-drift
  confound — prefer a setting that also moves knowledge tasks, or lengthen training.
- Single GPU; eval = base+adapter via vLLM (`--lora`), `enable_thinking: false` (auto from the model
  profile). Sweeps use eval subsets; **confirm the chosen point on full sets**.
- **Full fine-tuning (deferred):** a 4B full FT fits ~50 GB on the 96 GB Blackwell, but vLLM only
  loads Qwen3.5 as `Qwen3_5ForConditionalGeneration`; the text-only `AutoModelForCausalLM` merge is
  rejected. Serving a full-FT checkpoint needs new VLM-preserving save/merge tooling — add only if
  the LoRA best-case proves insufficient.
