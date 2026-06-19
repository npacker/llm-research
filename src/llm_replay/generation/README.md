# `llm_replay.generation` — EDT synthetic-data generation + quality validation

Generates synthetic corpora with dynamic-temperature (EDT) sampling and validates their
quality before downstream use. Driven by [`scripts/generate.py`](../../../scripts/generate.py)
and [`scripts/validate.py`](../../../scripts/validate.py); configs in
[`configs/gen/`](../../../configs/gen/) and [`configs/validate/`](../../../configs/validate/).

## Modules
- **`temperature.py`** — EDT math (`entropy`, `edt_temperature(H, T0, N, theta) = T0*N**(theta/H)`)
  and `EDTLogitsProcessor`, a vLLM v1 per-request logits processor for **token-level** EDT
  (rescales each decode step's logits; `temperature=1.0`, params via `SamplingParams.extra_args`).
- **`prompts.py`** — prefix-only prompt construction (Area 4): `none` / `structural` / `snippet` /
  `variable`, built from a seed corpus.
- **`generator.py`** — vLLM orchestration for the three core strategies.
- **`validation.py`** — per-sample gates + coherence (perplexity) + diversity panel.

## Strategies (research plan Area 1)
| `strategy` | Mechanism | Notes |
|------------|-----------|-------|
| `fixed`     | constant `SamplingParams.temperature` | baseline (cond. D) |
| `token_edt` | `EDTLogitsProcessor` rescales logits each step | cond. A; `LLM(logits_processors=[…])`, `temperature=1.0` |
| `seq_edt`   | two-pass: warmup → mean entropy → one temp/seq | cond. B; entropy from top-k logprobs (approx) |

## Run
```sh
python scripts/generate.py --config configs/gen/token_edt.yaml --model <id> --generation 1
python scripts/validate.py  --corpus runs/gen1_token_edt_<ts>/samples.jsonl --real <seed> --generation 1
```
Generation writes `samples.jsonl` + `meta.json`; validation writes `clean.jsonl` +
`validation.json` (gate pass-rates, rejection breakdown, perplexity, diversity panel) under `runs/`.

## Notes
- **Raw continuation**, not chat — prompts feed `llm.generate()` directly, so there's no
  `enable_thinking` toggle (that's a chat-template arg; don't put it in a gen config's `model_args`).
- Single GPU: `tensor_parallel_size=1`. The vLLM `extra_args` + `logits_processors=[…]` registration
  is verified against vLLM 0.23.0. EDT direction is set by `N` (≷1) — verify against the EDT source.
- **Not yet built:** curriculum / task-adaptive / hybrid temperature; the recursive generate→train loop.
