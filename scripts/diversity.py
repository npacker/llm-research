#!/usr/bin/env python
"""Distribution/diversity metrics over generated corpora (the collapse axis).

Compares a synthetic corpus against an optional real/Gen-0 reference and writes a
metric panel to runs/. This is the corpus-vs-corpus complement to scripts/evaluate.py
(which scores model *capability*); together they instrument both collapse axes.

Corpus specs (--synthetic / --real) accept:
  - a local .txt file  (one text per line)
  - a local .jsonl file (use --text-field to pick the field)
  - "hf:<dataset>:<split>:<field>"  (e.g. hf:stas/openwebtext-10k:train:text)

Examples
--------
Synthetic vs. real reference (full panel)::

    python scripts/diversity.py --synthetic gen3.txt --real real.txt --generation 3

Reference-free only (diversity of one corpus)::

    python scripts/diversity.py --synthetic gen3.txt --generation 3

Skip MAUVE (its GPT-2 featurisation is the slow part)::

    python scripts/diversity.py --synthetic gen3.txt --real real.txt --no-mauve
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from llm_core.corpus import load_corpus

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--synthetic", required=True, help="Synthetic corpus spec (file or hf:...)"
    )
    p.add_argument(
        "--real",
        default=None,
        help="Real/Gen-0 reference corpus spec (enables reference-based metrics)",
    )
    p.add_argument("--text-field", default=None, help="Field name for .jsonl corpora")
    p.add_argument("--generation", default="0", help="Recursive-generation label")
    p.add_argument("--limit", type=int, default=None, help="Cap texts per corpus")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument(
        "--device", default=None, help="Embedding device (e.g. cuda, cpu); default auto"
    )
    p.add_argument(
        "--no-mauve", action="store_true", help="Skip MAUVE (the slow metric)"
    )
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def _flatten(panel: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten a (possibly nested) metric panel into (name, value) rows."""
    rows: list[tuple[str, object]] = []
    for k, v in panel.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            rows.extend(_flatten(v, f"{name}."))
        else:
            rows.append((name, v))
    return rows


def main() -> None:
    args = parse_args()
    from llm_core.metrics import diversity

    synth = load_corpus(args.synthetic, args.text_field, args.limit)
    if not synth:
        raise SystemExit(f"--synthetic {args.synthetic!r} loaded 0 usable texts")
    real = load_corpus(args.real, args.text_field, args.limit) if args.real else None
    if args.real and not real:
        raise SystemExit(f"--real {args.real!r} loaded 0 usable texts")
    print(
        f"[diversity] gen={args.generation} | synth={len(synth)}"
        + (f" real={len(real)}" if real else " (reference-free)")
    )

    panel = diversity.compute_panel(
        synth,
        real,
        embed_model=args.embed_model,
        device=args.device,
        with_mauve=not args.no_mauve,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        REPO_ROOT / "runs" / f"gen{args.generation}_diversity_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diversity.json"
    out_path.write_text(json.dumps(panel, indent=2))

    print(f"\n{'metric':18s} value")
    print("-" * 34)
    for name, v in _flatten(panel):
        print(f"{name:18s} {v:.4f}" if isinstance(v, float) else f"{name:18s} {v}")
    print(f"\n[diversity] wrote {out_path}")


if __name__ == "__main__":
    main()
