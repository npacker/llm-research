"""Corpus loading shared by the generation + validation tooling.

Accepts the same specs as ``scripts/diversity.py``:
  - a local ``.txt`` file (one text per line)
  - a local ``.jsonl`` file (pick a field with ``text_field``)
  - ``"hf:<dataset>:<split>:<field>"``
"""

from __future__ import annotations

import json
from pathlib import Path


def load_corpus(
    spec: str, text_field: str | None = None, limit: int | None = None
) -> list[str]:
    if spec.startswith("hf:"):
        _, dataset, split, field = spec.split(":", 3)
        from datasets import load_dataset

        ds = load_dataset(dataset, split=split)
        if limit:  # slice before reading the column so we don't materialise it all
            ds = ds.select(range(min(limit, len(ds))))
        texts = ds[field]
    else:
        path = Path(spec)
        if path.suffix == ".jsonl":
            if not text_field:
                raise SystemExit("text_field is required for .jsonl corpora")
            texts = [
                json.loads(line)[text_field]
                for line in path.read_text().splitlines()
                if line.strip()
            ]
        else:  # .txt, one text per line
            texts = path.read_text().splitlines()
    texts = [t for t in texts if t and t.strip()]
    return texts[:limit] if limit else texts
