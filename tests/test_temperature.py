"""EDT temperature math — entropy + the T = T0 * N**(theta/H) schedule.

The pure functions (`entropy`, `edt_temperature`) are the easy-to-get-wrong core; the
vLLM logits processor is wiring around them and is exercised by manual GPU runs instead.
"""

import math

import torch

from llm_core.generation.temperature import ENTROPY_FLOOR, edt_temperature, entropy


def test_entropy_uniform_is_log_vocab():
    """Uniform distribution (equal logits) has entropy ln(V)."""
    for v in (2, 10, 128):
        h = float(entropy(torch.zeros(v)))
        assert math.isclose(h, math.log(v), rel_tol=1e-5)


def test_entropy_peaked_is_near_zero():
    """A near-one-hot distribution has entropy ~0."""
    logits = torch.tensor([100.0, 0.0, 0.0, 0.0])
    assert float(entropy(logits)) < 1e-3


def test_entropy_batched_reduces_last_dim():
    """A [*, vocab] tensor reduces over the last dim → one value per row."""
    logits = torch.zeros(3, 8)
    h = entropy(logits)
    assert h.shape == (3,)
    assert torch.allclose(h, torch.full((3,), math.log(8)), atol=1e-5)


def test_edt_high_entropy_relaxes_to_t0():
    """As H -> inf the exponent -> 0 so T -> T0, independent of N."""
    for n in (0.5, 2.0):
        assert math.isclose(edt_temperature(1e9, 1.3, n, 1.0), 1.3, rel_tol=1e-6)


def test_edt_direction_n_greater_than_one():
    """N > 1: confident (low-entropy) steps get T > T0 (more diverse where sure)."""
    t0 = 1.0
    assert edt_temperature(0.1, t0, 2.0, 1.0) > t0


def test_edt_direction_n_less_than_one():
    """N < 1: confident steps get T < T0 (more conservative where sure)."""
    t0 = 1.0
    assert edt_temperature(0.1, t0, 0.5, 1.0) < t0


def test_edt_entropy_floor_no_div_by_zero():
    """H = 0 hits the entropy floor instead of dividing by zero.

    (The floor guards the ``theta / H`` division; with N == 1 the exponent is inert,
    isolating that guard. Real entropy-driven calls always have H > 0.)
    """
    val = edt_temperature(0.0, 1.3, 1.0, 1.0)  # N=1 → N**x == 1 for any finite x
    assert math.isfinite(val)
    assert math.isclose(val, 1.3, rel_tol=1e-9)
    # Same as evaluating exactly at the floor — the division is what the floor protects.
    assert edt_temperature(0.0, 1.3, 1.0, 1.0) == edt_temperature(
        ENTROPY_FLOOR, 1.3, 1.0, 1.0
    )
