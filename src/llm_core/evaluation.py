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


def task_matches(task: str, names: list[str]) -> bool:
    """True if ``task`` equals or is prefixed by one of ``names``.

    lm-eval expands a group task into subtasks (e.g. ``leaderboard_gpqa`` ->
    ``leaderboard_gpqa_main`` / ``…_diamond``), so a config that names the *group* must
    match its expanded members — prefix matching does that.
    """
    return any(task == n or task.startswith(n) for n in names)


def bucket_deltas(
    current: dict[str, float],
    base: dict[str, float],
    buckets: dict[str, list[str]],
) -> dict[str, list[float]]:
    """Group ``current - base`` metric deltas by task bucket.

    ``current``/``base`` are flattened ``{"task/metric": value}`` dicts (see
    :func:`flatten_results`). ``buckets`` maps a bucket name to a list of task-name
    prefixes (e.g. ``{"format": ["leaderboard_ifeval"], "knowledge": ["gsm8k", ...]}``).
    Returns ``{bucket: [deltas]}`` over the metrics whose task matches that bucket (first
    matching bucket wins) and that exist in *both* dicts — so a missing base metric is
    skipped rather than counted as a delta. The caller decides how to reduce each list
    (mean, etc.) and can treat an empty list as "no data".
    """
    out: dict[str, list[float]] = {name: [] for name in buckets}
    for key, val in current.items():
        b = base.get(key)
        if b is None:
            continue
        task = key.split("/", 1)[0]
        for name, prefixes in buckets.items():
            if task_matches(task, prefixes):
                out[name].append(val - b)
                break
    return out
