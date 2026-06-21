"""Scoring for the catastrophic-forgetting / replay study.

The sweep tooling ranks fine-tuning recipes by how much domain capability they gain per
unit of general capability they forget. That trade-off score is research-specific (it
encodes the study's notion of a "good" recipe), so it lives in the research layer rather
than in the general ``llm_core`` sweep primitives.
"""

from __future__ import annotations


def learning_per_forgetting(domain_delta: float, forget: float) -> float:
    """Domain-learning gain per unit forgetting (higher = better), for ranking.

    ``domain_delta`` is the domain perplexity Δ (negative = learned); ``forget`` is the
    forgetting magnitude in whatever *single* unit the caller passes (general ppl Δ, or a
    knowledge-accuracy drop), positive = worse. A point that learned with no measurable
    forgetting (``forget <= 0``) is Pareto-best on this axis → ``+inf``, so it ranks first
    without an arbitrary epsilon-clamp distorting the scale.
    """
    learned = -domain_delta
    return float("inf") if forget <= 0 else learned / forget
