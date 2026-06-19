# `tests/` — unit tests (not yet committed)

No committed `pytest` suite yet. The numeric code built so far
([`src/llm_replay/metrics/diversity.py`](../src/llm_replay/metrics/diversity.py)) has been
validated by **smoke runs** — diverse-vs-collapsed corpora where every metric must separate in
the known direction (MAUVE ↑ when diverse, Vendi ↑, Self-BLEU ↓, prdc recall ↑, KL ↓, …). That's
how the Self-BLEU brevity-penalty bug was caught. Custom eval tasks were similarly validated
(SuperGPQA reproduced its published 16.9; IFBench: 58/58 verifier coverage).

This is where those checks should become **proper `pytest` tests** — the definitions easiest to
get wrong:

- **Metrics**: Distinct-n (unique n-grams / total *n-grams*), Self-BLEU direction, KL direction
  (`KL(P_real ‖ P_synthetic)`), prdc recall as the collapse signal.
- **Temperature math** (once `generation/` exists): EDT `T = T₀ × N^(θ/entropy)` (entropy floor,
  sign/direction) and the curriculum schedules (cosine endpoints).

`pytest` isn't a dependency yet — add it to `pyproject.toml` when the suite lands.
