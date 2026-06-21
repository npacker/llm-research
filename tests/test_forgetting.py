"""Forgetting trade-off score (`llm_replay.forgetting.learning_per_forgetting`)."""

import pytest

from llm_replay.forgetting import learning_per_forgetting


def test_no_forgetting_ranks_first():
    # forget <= 0 (general capability unchanged/improved) = Pareto-best → +inf, no clamp.
    assert learning_per_forgetting(-5.0, 0.0) == float("inf")
    assert learning_per_forgetting(-0.001, -3.0) == float("inf")


def test_finite_ratio_with_real_forgetting():
    assert learning_per_forgetting(-5.0, 1.0) == pytest.approx(5.0)
    assert learning_per_forgetting(-1.0, 2.0) == pytest.approx(0.5)


def test_orders_strong_learner_above_weak():
    strong = learning_per_forgetting(-5.0, 0.5)  # 10.0
    weak = learning_per_forgetting(-0.5, 0.5)  # 1.0
    none = learning_per_forgetting(-0.5, -0.1)  # inf
    assert none > strong > weak
