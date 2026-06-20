# `tests/` — unit tests (CPU-only, no network)

Run with `pytest tests/` (or `python -m pytest tests/`). The suite covers the
numeric/logic core that was previously exercised only by manual runs and has regressed before.
Everything here runs on CPU with no network or model downloads — embedding/MAUVE/Vendi/prdc
metrics (which need models) are intentionally **not** unit-tested; they stay on manual GPU runs.

| File | Covers |
| --- | --- |
| `test_temperature.py` | `entropy` (uniform ≈ ln V, peaked ≈ 0), `edt_temperature` (H→∞ ⇒ →T0, N≷1 direction, entropy-floor guard) |
| `test_diversity.py` | `distinct_n`, `self_bleu` (diverse < repetitive), `unigram_kl` (direction + asymmetry), `tail_mass`, `vocabulary_size` |
| `test_validation.py` | `gate_sample` (empty/short/long/repetitive/language), `_repetition_ratio`, `_max_line_repeat_frac` (the <3-lines ⇒ 0 fix) |
| `test_prompts.py` | `build_prompts` for every prefix mode incl. `chat`; seed-determinism; error cases |
| `test_data.py` | `mix_corpora` row-count weighting + deterministic shuffle over local `.txt` fixtures (no `hf:`) |
| `test_evaluation.py` | `summarize` (lm-eval result flattening; stderr/alias filtering) + the forgetting-report flatten shape |
| `test_models.py` | `profile_from` capability auto-detection (is_vlm / chat template / thinking / override precedence) — the "new non-Qwen arch is config-only" guarantee |

`pytest` is a project dependency (frozen by `uv.lock`); `tests/` is not imported as a
package — the editable install of `llm_core` + `llm_replay` makes `import llm_core…` /
`import llm_replay…` resolve directly.

> Note: `edt_temperature` overflows at exactly `H = 0` for `N > 1` (the floored exponent
> `theta / 1e-6` makes `N**1e6` overflow). This never occurs on the real token-EDT path
> (the entropy of any softmax is > 0); the floor exists only to guard the division.
