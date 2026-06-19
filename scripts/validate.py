#!/usr/bin/env python
"""Validate a generated corpus: per-sample quality gates + coherence + diversity panel.

Consumes the JSONL written by scripts/generate.py (or any corpus spec), applies quality
gates (empty/length/repetition/language/dedup), scores coherence via perplexity under a
fixed reference model, and computes the distribution/diversity panel against an optional
real/Gen-0 reference. Writes a clean corpus + a validation report under runs/.

Examples
--------
    python scripts/validate.py --corpus runs/gen1_token_edt_<ts>/samples.jsonl \
        --real hf:stas/openwebtext-10k:train:text --generation 1

    # reference-free, skip perplexity/MAUVE for a fast check:
    python scripts/validate.py --corpus gen1.jsonl --no-perplexity --no-mauve
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
sys.path.insert(0, str(REPO_ROOT / "src"))  # package not installed (package = false)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--corpus",
        required=True,
        help="Generated corpus: .jsonl (--text-field, default 'text') / .txt / hf:...",
    )
    p.add_argument(
        "--real",
        default=None,
        help="Real/Gen-0 reference corpus spec (enables reference-based metrics)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/validate/default.yaml",
        help="Validation config",
    )
    p.add_argument(
        "--text-field", default="text", help="Field for .jsonl corpora (default: text)"
    )
    p.add_argument("--generation", default="1", help="Recursive-generation label")
    p.add_argument("--limit", type=int, default=None, help="Cap texts (smoke tests)")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--no-perplexity",
        action="store_true",
        help="Skip the coherence/perplexity stage",
    )
    p.add_argument(
        "--no-mauve", action="store_true", help="Skip MAUVE in the diversity panel"
    )
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}

    from llm_replay import corpus
    from llm_replay.generation import validation

    synth = corpus.load_corpus(args.corpus, args.text_field, args.limit)
    if not synth:
        raise SystemExit(f"--corpus {args.corpus!r} loaded 0 texts")
    real = (
        corpus.load_corpus(args.real, args.text_field, args.limit)
        if args.real
        else None
    )
    print(
        f"[validate] gen={args.generation} | synth={len(synth)}"
        + (f" real={len(real)}" if real else "")
    )

    ppl_cfg = dict(cfg.get("perplexity", {}))
    if args.no_perplexity:
        ppl_cfg["enabled"] = False

    clean, report = validation.validate(
        synth,
        real,
        gates=cfg.get("gates"),
        perplexity_cfg=ppl_cfg if ppl_cfg else None,
        embed_model=cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
        device=args.device,
        with_mauve=not args.no_mauve,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        DEFAULT_RUNS_DIR / f"gen{args.generation}_validation_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean.jsonl").write_text(
        "".join(json.dumps({"text": t}) + "\n" for t in clean)
    )
    (out_dir / "validation.json").write_text(json.dumps(report, indent=2, default=str))

    print(
        f"\n[validate] pass_rate={report['pass_rate']:.3f} "
        f"({report['n_clean']}/{report['n_input']}) | rejections={report['rejections']}"
    )
    if "perplexity" in report:
        print(f"[validate] perplexity mean={report['perplexity']['mean']}")
    print(f"[validate] wrote {out_dir / 'validation.json'} + clean.jsonl")


if __name__ == "__main__":
    main()
