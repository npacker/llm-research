# `research/` — planning & writeups

Research planning documents and per-paper writeups for this workspace.

## Index

- [`dynamic-temperature-generative-replay.md`](dynamic-temperature-generative-replay.md) —
  a 5-area coordinated program on **entropy-based dynamic temperature (EDT) sampling for
  generative replay and model-collapse prevention**:
  1. Token-level vs. sequence-level temperature granularity
  2. Curriculum-based temperature scheduling
  3. Task-adaptive temperature
  4. Prefix × temperature interaction (16-cell factorial)
  5. Model-collapse prevention via dynamic temperature

  Includes hypotheses, experimental protocols, metrics, challenges, timelines, and a
  publication strategy.

- [`exp0-forgetting-signal.md`](exp0-forgetting-signal.md) — **Experiment 0** (runs first):
  hyperparameter sweep to find a regime where domain-only fine-tuning *both* learns the domain and
  measurably forgets, so Exp 1 has a real tradeoff to mitigate. Also budgets Exp 1's runtime.
- [`exp-mixing-unit.md`](exp-mixing-unit.md) — methods control: corpus mixing by **token budget vs
  row count** (rows differ wildly in length, so nominal ratios ≠ token ratios). Decides Exp 1's unit.
- [`exp1-forgetting-replay.md`](exp1-forgetting-replay.md) — **Experiment 1** (the headline test):
  single-generation catastrophic-forgetting & generative-replay study on a medical domain
  (Qwen3.5-4B **instruct**, LoRA continued-LM). 5 corpus-mix conditions (domain-only … synthetic-only)
  vs. the base model; forgetting on the general battery, domain gain via held-out perplexity + medical
  QA; synthetic via chat-template generation (no seeding). Runs after Exp 0.

## How this maps to the repo

The doc's "Shared Infrastructure" maps to [`../src/llm_replay/`](../src/llm_replay/);
experiments live in [`../experiments/`](../experiments/) driven by
[`../configs/`](../configs/); outputs go to [`../runs/`](../runs/) and are tracked with wandb.
See each directory's README for conventions and the single-GPU feasibility note in the plan doc.

### Implemented so far

The doc's shared **Capability Battery** and Area-5 collapse instrumentation are partly built:

- **Capability axis** — [`../scripts/evaluate.py`](../scripts/evaluate.py) (lm-eval) +
  [`../eval_tasks/`](../eval_tasks/) (SuperGPQA, IFBench) + [`../configs/eval/`](../configs/eval/).
- **Distribution/diversity axis** (primary collapse signal) —
  [`../scripts/diversity.py`](../scripts/diversity.py) + `../src/llm_replay/metrics/diversity.py`.
- **Generation (Area 1/4)** — [`../scripts/generate.py`](../scripts/generate.py) +
  [`../src/llm_replay/generation/`](../src/llm_replay/generation/): fixed / sequence-level /
  token-level EDT + prefix-only prompts, with a quality-validation pipeline
  ([`../scripts/validate.py`](../scripts/validate.py): gates + perplexity + diversity panel).
- **Training / forgetting study (Areas 2/5, single generation)** —
  [`../scripts/train.py`](../scripts/train.py) + [`../src/llm_replay/training/`](../src/llm_replay/training/):
  LoRA fine-tuning over domain/general/synthetic mixes; forgetting measured on the general battery,
  domain gain on `configs/eval/medical.yaml`, compared via
  [`../scripts/forgetting_report.py`](../scripts/forgetting_report.py).

Not yet built: curriculum / task-adaptive / hybrid temperature, and the **recursive** multi-generation
loop that chains generate → validate → train across generations.
