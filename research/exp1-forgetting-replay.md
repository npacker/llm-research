# Experiment 1 — Catastrophic forgetting & generative replay (medical, single generation)

**Status:** tooling built and pipeline-validated on tiny smokes; full run pending.
**Scope:** one generation (base → fine-tune → measure). First rung toward the recursive
collapse loop in [`dynamic-temperature-generative-replay.md`](dynamic-temperature-generative-replay.md)
(Areas 2 & 5).

## Question
When a model is adapted to a narrow **domain**, does it forget general capability — and can
**generative replay** (synthetic data sampled from the *base* model) mitigate that forgetting as
well as replaying *real* general data?

### Hypotheses
- **H1 (forgetting):** domain-only fine-tuning degrades the general battery vs. the base model.
- **H2 (real replay):** mixing real general data reduces that degradation.
- **H3 (substitution):** synthetic general data recovers *most* of H2's mitigation (the core claim
  — synthetic can stand in for unavailable pretraining data).
- **H4 (augmentation):** synthetic *on top of* real general adds little or helps modestly.
- **H5 (reference):** synthetic-only adapts less to the domain and/or forgets differently.

## Conditions
All are corpus mixes (configs in [`../configs/train/`](../configs/train/)); the reference is the
**un-fine-tuned base** (no forgetting).

| # | Config | Mix (role: weight) | Tests |
|---|--------|--------------------|-------|
| 0 | — (base) | — | reference / no-forgetting |
| 1 | `domain_only` | domain 1.0 | H1 forgetting baseline |
| 2 | `domain_general` | domain .5 / general .5 | H2 real replay |
| 3 | `domain_synthetic` | domain .5 / synthetic .5 | H3 substitution |
| 4 | `domain_general_synthetic` | domain .5 / general .25 / synthetic .25 | H4 augmentation |
| 5 | `synthetic_only` | synthetic 1.0 | H5 reference |

## Setup
- **Model:** `Qwen/Qwen3.5-4B` (tooling is `--model`-agnostic). Note it's a **VLM**; we fine-tune
  the text tower (LoRA, continued-LM on raw text). Single GPU.
- **Domain corpus:** `hf:MedRAG/textbooks:train:content` (medical textbook prose).
- **General corpus:** `hf:Salesforce/wikitext:wikitext-103-raw-v1:train:text`.
- **Synthetic corpus:** generated from the **base** model with [`../scripts/generate.py`](../scripts/generate.py)
  (strategy `token_edt`, prefix `snippet` seeded from the general corpus → base-model general-domain
  continuations), then quality-filtered by [`../scripts/validate.py`](../scripts/validate.py); the
  resulting `clean.jsonl` is the synthetic spec. *(Generator strategy — `token_edt` vs `fixed` vs
  `seq_edt` — is a follow-up knob.)*
- **Training:** LoRA (`target_modules: all-linear`, r=16) continued-LM, `total_samples=10000`,
  `max_len=1024`. Defaults in [`../src/llm_replay/training/sft.py`](../src/llm_replay/training/sft.py).

## Metrics
- **Forgetting (primary axis):** general capability battery via
  [`../configs/eval/canary.yaml`](../configs/eval/canary.yaml) (GSM8K, IFEval, GPQA) — milestone:
  `full.yaml`. Forgetting = score Δ vs. the base model.
- **Domain gain (objective-matched, primary):** **held-out medical perplexity** (base vs.
  base+adapter on a disjoint `MedRAG/textbooks` slice), computed automatically by `train.py` into
  `meta.json`. Negative delta = learned the domain.
- **Domain transfer (secondary):** medical QA battery [`../configs/eval/medical.yaml`](../configs/eval/medical.yaml)
  (MedQA / PubMedQA / MedMCQA / MMLU-medical) — expected to move less than perplexity.
- **(Optional) replay-corpus quality:** the synthetic corpus's distribution/diversity vs. real via
  [`../scripts/diversity.py`](../scripts/diversity.py).

## Procedure
```sh
# 0. Base reference (general + medical batteries)
python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm --config configs/eval/canary.yaml  --generation base
python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm --config configs/eval/medical.yaml --generation base

# 1. Synthetic corpus from the base model (for conditions 3/4/5)
python scripts/generate.py --config configs/gen/token_edt.yaml --model Qwen/Qwen3.5-4B --generation 1 \
    --seed-corpus hf:Salesforce/wikitext:wikitext-103-raw-v1:train:text   # prefix mode: set snippet in the config
python scripts/validate.py --corpus runs/gen1_token_edt_*/samples.jsonl \
    --real hf:Salesforce/wikitext:wikitext-103-raw-v1:train:text --generation 1   # -> clean.jsonl

# 2. Train each condition (domain ppl auto-reported into meta.json)
python scripts/train.py --config configs/train/domain_only.yaml              --model Qwen/Qwen3.5-4B
python scripts/train.py --config configs/train/domain_general.yaml           --model Qwen/Qwen3.5-4B
python scripts/train.py --config configs/train/domain_synthetic.yaml         --model Qwen/Qwen3.5-4B --synthetic runs/gen1_validation_*/clean.jsonl
python scripts/train.py --config configs/train/domain_general_synthetic.yaml --model Qwen/Qwen3.5-4B --synthetic runs/gen1_validation_*/clean.jsonl
python scripts/train.py --config configs/train/synthetic_only.yaml           --model Qwen/Qwen3.5-4B --synthetic runs/gen1_validation_*/clean.jsonl

# 3. Eval each condition: base + adapter via vLLM (forgetting + domain transfer)
python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm --lora runs/train_<cond>_*/adapter --lora-rank 16 \
    --config configs/eval/canary.yaml  --generation <cond>
python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm --lora runs/train_<cond>_*/adapter --lora-rank 16 \
    --config configs/eval/medical.yaml --generation <cond>

# 4. Compare
python scripts/forgetting_report.py --base base=runs/genbase_canary_*/eval/results.json \
    --runs domain_only=... domain_general=... domain_synthetic=... domain_general_synthetic=... synthetic_only=...
```

## Analysis
- **H1:** condition 1 general Δ < 0 (forgetting present).
- **H2:** condition 2 general Δ > condition 1 (real replay mitigates).
- **H3:** condition 3 ≈ condition 2 on the general battery (synthetic substitutes); read against
  each condition's domain-perplexity delta (all should still learn the domain).
- **H4:** condition 4 vs condition 2 (does synthetic add value?).
- **H5:** condition 5 = synthetic-only reference (domain ppl delta + general retention).
- Report deltas with the per-task noise in mind (use full test sets, not `--limit`).

## Decisions & caveats (carried from build/verification)
- **Eval = base + LoRA adapter via vLLM (`--lora`), not a merged checkpoint.** Qwen3.5 is a VLM;
  the easy (text-only) merge is a sub-arch vLLM doesn't register. A full vision+text merge would
  load but is wasteful; the adapter path avoids it.
- **`enable_thinking: false` in every eval config** — thinking-on tanks answer-extraction tasks
  (GSM8K 0.00 vs 0.60). All eval configs set it.
- **Domain metric = held-out perplexity (objective-matched)**; the QA battery is a transfer check.
  No dataset change needed for the prose/QA "mismatch" (see training/README).
- **Single GPU**, `tensor_parallel_size=1`; pilot-first (`--limit`) before the full `total_samples=10000`.
- Pipeline validated on tiny smokes (loss ~1.83; held-out medical ppl 6.63→6.26; base+adapter GSM8K
  ~0.78) — these are plumbing checks, **not** results.

## Next
Generator-strategy sweep (fixed / seq_edt / token_edt synthetic); scale to 32B/70B; then chain
into the **recursive** loop (gen N+1 trained on gen N's validated synthetic) for the collapse trajectory.
