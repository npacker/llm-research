# `src/llm_replay/` — generative-replay research code

The research layer for the program in
[`research/dynamic-temperature-generative-replay.md`](../../research/dynamic-temperature-generative-replay.md):
the parts that are **specific to the dynamic-temperature generative-replay & model-collapse
study**. General, reusable infrastructure (model profiles, eval glue, corpus loading,
diversity metrics, EDT generation, LoRA training) lives in
[`../llm_core/`](../llm_core/) and is imported via absolute `from llm_core… import …`.

| Module                    | Purpose                                                                 | Builds on `llm_core`        |
|---------------------------|-------------------------------------------------------------------------|-----------------------------|
| `generation/prompts.py`   | Prefix-only prompt construction (study conditions P1–P4 + chat mode) from a seed corpus | feeds `llm_core.generation.generator` |
| `generation/validation.py`| Synthetic-corpus quality validation — per-sample gates + perplexity + the diversity panel | `llm_core.metrics.diversity` |

Driven by [`../../scripts/generate.py`](../../scripts/generate.py),
[`../../scripts/validate.py`](../../scripts/validate.py), and
[`../../scripts/train.py`](../../scripts/train.py) (which mixes a validated synthetic corpus
into a `llm_core.training` LoRA run). See [`generation/README.md`](generation/README.md) for
the per-module detail and the experiment docs under [`../../research/`](../../research/) for the
study design (prefix conditions, forgetting/replay condition matrix).
