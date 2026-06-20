# `llm_core.generation` — EDT temperature + vLLM text generation

General text-generation building blocks: dynamic-temperature (EDT) sampling and a vLLM
orchestrator. No assumptions about what is generated or why — the research-specific prompt
construction and quality validation live in
[`../../llm_replay/generation/`](../../llm_replay/generation/).

## Modules
- **`temperature.py`** — EDT math (`entropy`, `edt_temperature(H, T0, N, theta) = T0*N**(theta/H)`)
  and `EDTLogitsProcessor`, a vLLM v1 per-request logits processor for **token-level** EDT
  (rescales each decode step's logits; `temperature=1.0`, params via `SamplingParams.extra_args`).
- **`generator.py`** — vLLM orchestration for the three strategies below; applies the model's
  chat template when the profile has one, and gives every request a distinct seed.

## Strategies
| `strategy` | Mechanism | Notes |
|------------|-----------|-------|
| `fixed`     | constant `SamplingParams.temperature` | baseline |
| `token_edt` | `EDTLogitsProcessor` rescales logits each step | `LLM(logits_processors=[…])`, `temperature=1.0` |
| `seq_edt`   | two-pass: warmup → mean entropy → one temp/seq | entropy from top-k logprobs (approx) |

## Notes
- Single GPU: `tensor_parallel_size=1`. The vLLM `extra_args` + `logits_processors=[…]`
  registration is verified against vLLM 0.23.0. EDT direction is set by `N` (≷1) — verify
  against the EDT source for your setup.
