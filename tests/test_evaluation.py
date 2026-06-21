"""`summarize` — flatten lm-eval results into primary-metric rows.

Mirrors the real lm-eval results shape (``"<metric>,<filter>"`` keys, stderr siblings,
a string ``alias``) so the filter logic is locked down for both the live eval and the
forgetting report that reuses it.
"""

import pytest

from llm_core.evaluation import bucket_deltas, summarize, task_matches

RESULTS = {
    "results": {
        "gsm8k": {
            "alias": "gsm8k",
            "exact_match,strict-match": 0.6,
            "exact_match_stderr,strict-match": 0.01,
        },
        "leaderboard_ifeval": {
            "alias": "ifeval",
            "prompt_level_strict_acc,none": 0.42,
            "prompt_level_strict_acc_stderr,none": 0.02,
        },
    }
}


def test_summarize_extracts_primary_metrics():
    rows = summarize(RESULTS)
    assert ("gsm8k", "exact_match (strict-match)", 0.6) in rows
    assert ("leaderboard_ifeval", "prompt_level_strict_acc", 0.42) in rows


def test_summarize_drops_stderr_and_alias():
    rows = summarize(RESULTS)
    labels = [m for _, m, _ in rows]
    assert not any("stderr" in m for m in labels)
    assert all(isinstance(v, float) for *_, v in rows)
    # alias (a string) and stderr siblings are dropped → exactly the 2 primary metrics.
    assert len(rows) == 2


def test_summarize_sorted_by_task():
    rows = summarize(RESULTS)
    tasks = [t for t, *_ in rows]
    assert tasks == sorted(tasks)


def test_summarize_empty_results():
    assert summarize({}) == []
    assert summarize({"results": {}}) == []


def test_forgetting_report_flatten_shape():
    """forgetting_report._flatten builds 'task/metric' -> value from summarize()."""
    flat = {f"{t}/{m}": v for t, m, v in summarize(RESULTS)}
    assert flat["gsm8k/exact_match (strict-match)"] == 0.6
    assert flat["leaderboard_ifeval/prompt_level_strict_acc"] == 0.42


def test_task_matches_exact_and_group_prefix():
    # lm-eval expands a group task into prefixed subtasks; the group name must match them.
    assert task_matches("gsm8k", ["gsm8k"])
    assert task_matches(
        "leaderboard_gpqa_main", ["leaderboard_gpqa"]
    )  # subtask of group
    assert not task_matches("leaderboard_ifeval", ["gsm8k", "leaderboard_gpqa"])


def test_bucket_deltas_groups_by_first_matching_bucket():
    current = {
        "gsm8k/acc": 0.5,
        "leaderboard_ifeval/acc": 0.30,
        "leaderboard_gpqa_main/acc": 0.20,
    }
    base = {
        "gsm8k/acc": 0.6,
        "leaderboard_ifeval/acc": 0.42,
        "leaderboard_gpqa_main/acc": 0.25,
    }
    grouped = bucket_deltas(
        current,
        base,
        {"fmt": ["leaderboard_ifeval"], "knowledge": ["gsm8k", "leaderboard_gpqa"]},
    )
    assert grouped["fmt"] == pytest.approx([-0.12])  # ifeval delta
    assert sorted(grouped["knowledge"]) == pytest.approx(
        [-0.1, -0.05]
    )  # gsm8k + gpqa deltas


def test_bucket_deltas_skips_metrics_missing_from_base():
    current = {"gsm8k/acc": 0.5, "newtask/acc": 0.9}
    base = {"gsm8k/acc": 0.6}  # newtask absent from base
    grouped = bucket_deltas(current, base, {"knowledge": ["gsm8k", "newtask"]})
    assert grouped["knowledge"] == pytest.approx([-0.1])  # newtask skipped, not counted
