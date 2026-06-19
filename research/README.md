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

Not yet built: the EDT generation pipeline, LoRA training, and the recursive-generation loop that
ties both axes together across generations.
