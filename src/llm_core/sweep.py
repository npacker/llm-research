"""General hyperparameter-sweep primitives — grid enumeration + Pareto selection.

These are study-agnostic: any sweep that enumerates a grid of points and shortlists a
multi-objective front can reuse them. The *interpretation* of the axes/objectives (what
"learning" vs "forgetting" means) stays in the study runner / the research layer
(``llm_replay``); this module only knows about cartesian products and dominance.

Pure (stdlib only) and unit-tested, so a study runner stays a thin orchestrator over it.
"""

from __future__ import annotations

from itertools import product
from typing import Any


def enumerate_grid(
    axes: dict[str, list[Any]],
    *,
    extra: list[dict] | None = None,
    id_format: str,
) -> list[dict]:
    """Enumerate grid points from ``axes`` (+ any explicit ``extra`` points).

    ``axes`` maps an axis name to its list of values; the cartesian product over them is
    taken in axis order. ``extra`` is a list of already-assembled points (e.g. a sub-sweep
    that varies one axis at a fixed center) appended after the main grid. Points are
    de-duplicated by their full axis assignment, preserving first-seen order, so a center
    shared by the grid and a sub-sweep appears once.

    Each returned point is the axis→value dict with an ``id`` added from
    ``id_format.format(**point)`` (e.g. ``"lr{learning_rate:g}_r{lora_rank}"``).
    """
    keys = list(axes)
    settings = [dict(zip(keys, combo)) for combo in product(*(axes[k] for k in keys))]
    settings += [dict(e) for e in (extra or [])]

    seen: set[tuple] = set()
    points: list[dict] = []
    for s in settings:
        key = tuple(sorted(s.items()))
        if key in seen:
            continue
        seen.add(key)
        points.append({"id": id_format.format(**s), **s})
    return points


def pareto_front(items: list[dict], x: str, y: str) -> list[dict]:
    """Non-dominated set minimizing both ``x`` and ``y``.

    A point is dominated if another is ≤ it on both axes and strictly < on at least one;
    dominated points are dropped, the rest (the Pareto front) returned in input order.
    """
    front = []
    for a in items:
        if not any(
            b is not a
            and b[x] <= a[x]
            and b[y] <= a[y]
            and (b[x] < a[x] or b[y] < a[y])
            for b in items
        ):
            front.append(a)
    return front
