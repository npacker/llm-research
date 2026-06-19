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
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))  # package not installed (package = false)


def load_corpus(spec: str, text_field: str | None, limit: int | None) -> list[str]:
    if spec.startswith("hf:"):
        _, dataset, split, field = spec.split(":", 3)
        from datasets import load_dataset

        ds = load_dataset(dataset, split=split)
        texts = ds[field]
    else:
        path = Path(spec)
        if path.suffix == ".jsonl":
            if not text_field:
                raise SystemExit("--text-field is required for .jsonl corpora")
            texts = [
                json.loads(line)[text_field]
                for line in path.read_text().splitlines()
                if line.strip()
            ]
        else:  # .txt, one text per line
            texts = [ln for ln in path.read_text().splitlines() if ln.strip()]
    texts = [t for t in texts if t and t.strip()]
    return texts[:limit] if limit else texts


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


def main() -> None:
    args = parse_args()
    from llm_replay.metrics import diversity

    synth = load_corpus(args.synthetic, args.text_field, args.limit)
    real = load_corpus(args.real, args.text_field, args.limit) if args.real else None
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
    for k, v in panel.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"{k + '.' + kk:18s} {vv:.4f}")
        elif isinstance(v, float):
            print(f"{k:18s} {v:.4f}")
        else:
            print(f"{k:18s} {v}")
    print(f"\n[diversity] wrote {out_path}")


if __name__ == "__main__":
    main()
