"""Pure (no-embedding) diversity/distribution metrics.

These are the definitions easiest to get wrong and the ones that have regressed before
(Self-BLEU direction). Embedding/MAUVE/Vendi/prdc metrics need models and are covered by
manual GPU runs, not here.
"""

import math

from llm_core.metrics import diversity

DIVERSE = [
    "the quick brown fox",
    "a slow green turtle",
    "bright yellow sunflowers bloom",
]
REPETITIVE = ["the same thing again"] * 6


def test_distinct_n_diverse_gt_repetitive():
    assert diversity.distinct_n(DIVERSE, 2) > diversity.distinct_n(REPETITIVE, 2)


def test_distinct_n_all_unique_is_one():
    assert diversity.distinct_n(["a b c d"], 2) == 1.0


def test_distinct_n_empty_is_zero():
    assert diversity.distinct_n([""], 2) == 0.0


def test_vocabulary_size_counts_unique_tokens():
    assert diversity.vocabulary_size(["a b c", "a b"]) == 3


def test_tail_mass_fraction_in_unit_interval():
    tm = diversity.tail_mass(DIVERSE, frac=0.2)
    assert 0.0 <= tm <= 1.0


def test_tail_mass_shrinks_when_one_token_dominates():
    """A corpus dominated by one repeated token holds little mass in the rare tail."""
    dominated = ["x " * 100 + "rareword"]
    assert diversity.tail_mass(dominated, frac=0.5) < diversity.tail_mass(DIVERSE, 0.5)


def test_self_bleu_diverse_lt_repetitive():
    """Self-BLEU is lower (more diverse) for distinct texts than repeated ones."""
    assert diversity.self_bleu(DIVERSE) < diversity.self_bleu(REPETITIVE)


def test_self_bleu_single_text_is_zero():
    assert diversity.self_bleu(["only one"]) == 0.0


def test_unigram_kl_identical_is_near_zero():
    """KL(P_real || P_synth) ~ 0 when the corpora share a distribution."""
    corpus = ["a b c", "a b c", "a a b"]
    assert diversity.unigram_kl(corpus, corpus) == 0.0 or math.isclose(
        diversity.unigram_kl(corpus, corpus), 0.0, abs_tol=1e-9
    )


def test_unigram_kl_grows_with_divergence():
    """KL grows as the synthetic distribution drifts from the real one."""
    real = ["a a a a b"]
    close = ["a a a b b"]
    far = ["c c c d d"]
    assert diversity.unigram_kl(real, far) > diversity.unigram_kl(real, close)


def test_unigram_kl_directionality():
    """KL is asymmetric: KL(P||Q) != KL(Q||P) in general.

    (A two-symbol swap like a:3/4,b:1/4 vs a:1/4,b:3/4 is accidentally symmetric, so
    use unequal-entropy distributions to exercise the asymmetry.)
    """
    p = ["a a b"]  # a:2/3, b:1/3
    q = ["a b b b"]  # a:1/4, b:3/4
    assert not math.isclose(
        diversity.unigram_kl(p, q), diversity.unigram_kl(q, p), abs_tol=1e-6
    )
