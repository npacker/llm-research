"""Entropy-based Dynamic Temperature (EDT) — math + the vLLM logits processor.

EDT adjusts the sampling temperature from the entropy of the current next-token
distribution (Zhang et al., 2024): ``T = T0 * N**(theta / H)``. As ``H -> inf`` the
exponent ``-> 0`` so ``T -> T0`` (high entropy relaxes toward the base temperature);
the low-entropy (confident) end is pushed away from ``T0`` in a direction set by ``N``:

- ``N > 1``: confident steps get ``T > T0`` (more diverse where the model is sure).
- ``N < 1``: confident steps get ``T < T0`` (more conservative where the model is sure).

``N`` and ``theta`` are hyperparameters — confirm the exact values/direction against the
EDT source for your setup; this module just implements the formula faithfully.

`entropy` / `edt_temperature` are pure and unit-testable. `EDTLogitsProcessor` wires
token-level EDT into vLLM's v1 engine: it rescales each row of logits per decode step,
so requests run with ``SamplingParams(temperature=1.0, extra_args={"edt_mode": "token",
"T0": ..., "N": ..., "theta": ...})``. Requests without that ``extra_args`` flag pass
through untouched, so one engine can mix EDT and non-EDT requests.
"""

from __future__ import annotations

import torch
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

ENTROPY_FLOOR = 1e-6


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of the softmax over a logits vector/row(s).

    Accepts a 1-D `[vocab]` or 2-D `[*, vocab]` tensor; reduces over the last dim.
    """
    logp = torch.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def edt_temperature(h: float, t0: float, n: float, theta: float) -> float:
    """EDT temperature ``T0 * N**(theta / H)`` with an entropy floor."""
    return t0 * (n ** (theta / max(h, ENTROPY_FLOOR)))


class EDTLogitsProcessor(AdapterLogitsProcessor):
    """Per-request token-level EDT for the vLLM v1 engine.

    Register at engine construction: ``LLM(..., logits_processors=[EDTLogitsProcessor])``.
    Inert unless a request sets ``extra_args["edt_mode"] == "token"``.
    """

    def is_argmax_invariant(self) -> bool:
        # Rescaling logits changes the sampled distribution (not a no-op for sampling).
        return False

    def new_req_logits_processor(self, params) -> RequestLogitsProcessor | None:
        ea = params.extra_args or {}
        if ea.get("edt_mode") != "token":
            return None
        t0 = float(ea["T0"])
        n = float(ea["N"])
        theta = float(ea["theta"])

        def _edt(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            h = float(entropy(logits))
            return logits / edt_temperature(h, t0, n, theta)

        return _edt
