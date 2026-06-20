"""Shared helpers for the capability-eval tooling.

`summarize` flattens an lm-evaluation-harness results dict into primary-metric rows
for any consumer that needs to compare or tabulate eval results, so it lives in the
package rather than being duplicated across scripts.
"""

from __future__ import annotations


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
