"""Shared helpers for the capability-eval tooling.

`summarize` flattens an lm-evaluation-harness results dict into primary-metric rows
for any consumer that needs to compare or tabulate eval results, so it lives in the
package rather than being duplicated across scripts.
"""

from __future__ import annotations

import json
from pathlib import Path


def summarize(results: dict) -> list[tuple[str, str, float]]:
    """Flatten lm-eval results into (task, metric, value) rows for primary metrics.

    lm-eval metric keys are ``"<metric>,<filter>"`` (e.g. ``exact_match,strict-match``
    for gsm8k, ``prompt_level_strict_acc,none`` for ifeval). We keep every numeric,
    non-stderr metric and label it with its filter when it isn't the trivial ``none``.
    """
    rows: list[tuple[str, str, float]] = []
    for task, metrics in sorted(results.get("results", {}).items()):
        for key, val in metrics.items():
            if not isinstance(val, (int, float)) or "," not in key:
                continue  # skips `alias` (str) and `sample_len` (no filter)
            metric, _, flt = key.partition(",")
            if metric.endswith("_stderr"):
                continue
            label = metric if flt in ("none", "") else f"{metric} ({flt})"
            rows.append((task, label, float(val)))
    return rows


def flatten_results(results) -> dict[str, float]:
    """Map an lm-eval results.json into ``{"task/metric": value}`` via :func:`summarize`.

    Accepts either a loaded results dict or a path to a ``results.json`` file, so the
    base-vs-run delta tabulation is defined once for every consumer (forgetting_report,
    the Exp-0 sweep report) instead of being inlined per script.
    """
    r = results if isinstance(results, dict) else json.loads(Path(results).read_text())
    return {f"{t}/{m}": v for t, m, v in summarize(r)}
