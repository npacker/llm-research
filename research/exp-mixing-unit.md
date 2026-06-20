# Methods experiment — corpus mixing unit: row-count vs. token-budget

**Status:** designed. Methods study that informs Exp 0/1's mixing convention.
**Motivation:** `mix_corpora` currently mixes by **row count**, but rows differ greatly in length —
domain (`MedRAG/textbooks`) ≈ 126 words/row, general (`pile-10k`) median ≈ 250 (mean ~1970),
synthetic ≈ a 512-token cap. So a nominal **0.5/0.5 by rows is not 0.5/0.5 by training tokens** — and
the mix ratio is the *independent variable* in Exp 1. This quantifies the distortion and picks the
unit Exp 1 should use.

## Question
Does mixing by **token budget** vs **row count** change (a) the realized token shares and (b) the
forgetting/domain-gain outcome enough to matter for the replay comparison?

## Prerequisite (small build)
Add a `mix_by: tokens | rows` option to `src/llm_core/training/data.py:mix_corpora` (token-budget:
draw rows from each source until its token quota — `weight × total_tokens` — is met, using the
tokenizer). Log the **realized token share per role** either way. (Row-count is the current behavior.)

## Design
- **Model / recipe:** the setting picked by [`exp0-forgetting-signal.md`](exp0-forgetting-signal.md)
  (so there's a real forgetting signal to perturb).
- **Conditions:** one mix — `domain_general` (nominal 0.5/0.5) — trained **twice**:
  1. `mix_by: rows` (current)
  2. `mix_by: tokens`
  Hold everything else fixed (seed, total token budget, recipe).
- **Report for each:** realized domain/general **token shares**, held-out domain perplexity Δ, and
  general-battery Δ vs. base.

## Hypothesis
Row-count mixing over-weights the longer corpus in *tokens* (here: general/pile and synthetic),
so "0.5/0.5 by rows" trains on materially more general tokens than intended — making real-replay
mitigation look stronger (or domain learning weaker) than the nominal ratio implies. Token-budget
mixing should make the ratio mean what it says.

## Decision
If the two differ beyond noise (likely, given the length gaps), **adopt token-budget mixing for
Exp 1** and report ratios in tokens. If they're within noise, document that row-count is adequate
at these corpus lengths and keep it. Either way, **Exp 1 should report realized token shares**, not
just nominal row weights.

## Notes
- Token-budget mixing also makes `pile-10k`'s long documents safe to use (they'd otherwise dominate
  a row-count mix).
- This is a methods control, not a headline result — keep it small (single mix, the two units).
