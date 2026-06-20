"""Corpus mixing + tokenisation for continued-LM LoRA fine-tuning.

`mix_corpora` builds one raw-text dataset from several corpora (domain / general /
synthetic) at configurable row-count weights; `tokenize_for_lm` tokenises it for causal-LM
loss. Reuses `llm_replay.corpus.load_corpus` for all corpus specs.
"""

from __future__ import annotations

from ..corpus import load_corpus


def mix_corpora(corpora: list[dict], total_samples: int, seed: int = 0):
    """Mix corpora by row-count weight into a shuffled `text` dataset.

    Each entry: ``{role, spec, weight, text_field?}``. Takes ``int(total_samples*weight)``
    rows from each (capped by availability), tags rows with their role, concatenates and
    shuffles deterministically. Returns ``(dataset, role_counts)``.
    """
    from datasets import Dataset, concatenate_datasets

    parts, role_counts = [], {}
    for c in corpora:
        spec = c.get("spec")
        if not spec:
            raise ValueError(
                f"corpus role {c.get('role')!r} has no spec (pass --synthetic / set it in the config)"
            )
        n = max(1, int(total_samples * c["weight"]))
        texts = load_corpus(spec, c.get("text_field", "text"), limit=n)
        role = c.get("role", "?")
        role_counts[role] = role_counts.get(role, 0) + len(texts)
        parts.append(Dataset.from_dict({"text": texts, "role": [role] * len(texts)}))
    mixed = concatenate_datasets(parts).shuffle(seed=seed)
    return mixed, role_counts


def tokenize_for_lm(dataset, tokenizer, max_len: int = 1024):
    """Tokenise the `text` column for causal-LM training (truncation; no padding here —
    the data collator pads per batch). Drops non-token columns."""

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    return dataset.map(_tok, batched=True, remove_columns=dataset.column_names)
