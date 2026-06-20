# `llm_replay.generation` — prefix prompts + synthetic-corpus validation

The generative-replay-specific half of the generation pipeline. The general EDT temperature
math and vLLM generator live in [`../../llm_core/generation/`](../../llm_core/generation/);
this package supplies *what to prompt with* and *how to quality-gate the output*. Driven by
[`scripts/generate.py`](../../../scripts/generate.py) and
[`scripts/validate.py`](../../../scripts/validate.py); configs in
[`configs/gen/`](../../../configs/gen/) and [`configs/validate/`](../../../configs/validate/).

## Modules
- **`prompts.py`** — prefix-only prompt construction (study Area 4). Modes mirror the doc's
  P1–P4 — `none` / `structural` / `snippet` / `variable` — plus `chat` for instruct-model
  replay, built from a seed corpus. The generator (`llm_core.generation.generator`) applies
  the chat template if configured.
- **`validation.py`** — synthetic-corpus quality gates: per-sample drops (empty / out-of-length
  / degenerate / wrong-language + de-dup, each attributed to a reason) then corpus scoring —
  coherence via perplexity under a fixed reference model + the `llm_core.metrics.diversity`
  panel on the survivors.

## Run
```sh
python scripts/generate.py --config configs/gen/token_edt.yaml --model <id> --generation 1
python scripts/validate.py  --corpus runs/gen1_token_edt_<ts>/samples.jsonl --real <seed> --generation 1
```
Generation writes `samples.jsonl` + `meta.json`; validation writes `clean.jsonl` +
`validation.json` (gate pass-rates, rejection breakdown, perplexity, diversity panel) under `runs/`.

## Notes
- **Raw continuation**, not chat, in the prefix modes — prompts feed `llm.generate()` directly,
  so there's no `enable_thinking` toggle (that's a chat-template arg; don't put it in a gen
  config's `model_args`). `chat` mode is the exception (it does apply the template).
- **Not yet built:** curriculum / task-adaptive / hybrid temperature; the recursive
  generate→train loop.
