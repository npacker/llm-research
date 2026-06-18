# `tests/` — unit tests (planned)

Where unit tests for [`../src/llm_replay/`](../src/llm_replay/) go once shared code exists. The
**numeric core must be correct**, so prioritize tests for the definitions that are easy to get
wrong (and that the research-doc review caught):

- **Metrics**: Distinct-n (unique n-grams / total *n-grams*), Self-BLEU direction, KL direction
  (`KL(P_real ‖ P_synthetic)`), perplexity measured on the right set.
- **Temperature math**: EDT formula `T = T₀ × N^(θ/entropy)` (incl. the entropy floor and the
  sign/direction of the adjustment), and the curriculum schedules (cosine endpoints, etc.).

Run with `pytest tests/` after the package is set up. `pytest` isn't a dependency yet — add it
to `pyproject.toml` when tests land.
