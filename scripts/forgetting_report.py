#!/usr/bin/env python
"""Aggregate eval results across the base model + training conditions into a delta table.

Reads `results.json` files written by scripts/evaluate.py (one per model×battery) and
tabulates each task metric with its delta vs. the base model — so forgetting (general
battery drop) and domain gain (medical battery rise) are visible side by side.

Example
-------
    python scripts/forgetting_report.py \
        --base    base=runs/gen0_canary_<ts>/eval/results.json \
        --runs    domain_only=runs/gen.../eval/results.json \
                  domain_general=runs/.../eval/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_core.evaluation import summarize


def _flatten(results_json: Path) -> dict[str, float]:
    """task -> primary metric value, reusing the package's summarize()."""
    r = json.loads(Path(results_json).read_text())
    return {f"{t}/{m}": v for t, m, v in summarize(r)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--base",
        required=True,
        help="label=path/to/base/results.json (no-forgetting reference)",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=[],
        help="label=path/to/results.json for each condition",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_label, base_path = args.base.split("=", 1)
    base = _flatten(Path(base_path))
    runs = {lbl: _flatten(Path(p)) for lbl, p in (r.split("=", 1) for r in args.runs)}

    tasks = sorted(set(base) | {t for m in runs.values() for t in m})
    labels = list(runs)
    print(
        f"\n{'task/metric':40s} {base_label:>10s} "
        + " ".join(f"{lbl:>18s}" for lbl in labels)
    )
    print("-" * (52 + 19 * len(labels)))
    for t in tasks:
        b = base.get(t)
        cells = []
        for lbl in labels:
            v = runs[lbl].get(t)
            if v is None or b is None:
                cells.append(f"{'-':>18s}")
            else:
                cells.append(f"{v * 100:6.1f} (Δ{(v - b) * 100:+5.1f})")
        bcell = f"{b * 100:6.1f}" if b is not None else "-"
        print(f"{t:40s} {bcell:>10s} " + " ".join(cells))
    print(
        "\nΔ vs base: negative on the general battery = forgetting; positive on the medical battery = domain gain."
    )


if __name__ == "__main__":
    main()
