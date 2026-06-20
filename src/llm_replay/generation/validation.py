"""Quality validation for a generated corpus.

Two stages, then a report:
  1. **Per-sample gates** — drop empty / out-of-length / degenerate (repetitive) /
     wrong-language samples, and de-duplicate. Each drop is attributed to a reason.
  2. **Corpus scoring** on the survivors — coherence via perplexity under a fixed
     reference model + the distribution/diversity panel (reused from `metrics.diversity`).

`validate()` returns `(clean_texts, report)`; the CLI (scripts/validate.py) writes both.
"""

from __future__ import annotations

import math
import re
from collections import Counter

DEFAULT_GATES = {
    "min_chars": 20,
    "max_words": 2000,
    "repetition_n": 3,
    "max_repetition_ratio": 0.6,  # reject if >60% of n-grams are repeats
    "max_line_repeat_frac": 0.5,  # reject if one line is >50% of all lines
    "lang": "en",  # None to disable the language gate
    "dedup": True,
}


def _repetition_ratio(text: str, n: int) -> float:
    """Fraction of n-grams that are repeats (1 - distinct-n) for a single text."""
    toks = text.split()
    grams = list(zip(*[toks[i:] for i in range(n)]))
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def _max_line_repeat_frac(text: str) -> float:
    """Fraction of lines that are copies of the single most-common line.

    Returns 0 for <3 lines: a one- or two-line sample isn't "line-repetition
    degeneration" (intra-line repetition is caught by the n-gram gate instead).
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return 0.0
    return Counter(lines).most_common(1)[0][1] / len(lines)


def gate_sample(text: str, gates: dict) -> list[str]:
    """Return a list of rejection reasons ([] = passes)."""
    reasons = []
    t = text.strip()
    if not t:
        return ["empty"]
    if len(t) < gates["min_chars"]:
        reasons.append("too_short")
    if len(t.split()) > gates["max_words"]:
        reasons.append("too_long")
    if _repetition_ratio(t, gates["repetition_n"]) > gates["max_repetition_ratio"]:
        reasons.append("repetitive")
    if _max_line_repeat_frac(t) > gates["max_line_repeat_frac"]:
        reasons.append("line_repeat")
    if gates.get("lang"):
        try:
            from langdetect import detect

            if detect(t) != gates["lang"]:
                reasons.append("wrong_language")
        except Exception:
            reasons.append("lang_undetected")
    return reasons


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def validate(
    synth: list[str],
    real: list[str] | None = None,
    *,
    gates: dict | None = None,
    perplexity_cfg: dict | None = None,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
    with_mauve: bool = True,
) -> tuple[list[str], dict]:
    g = {**DEFAULT_GATES, **(gates or {})}
    rejections: Counter = Counter()
    kept: list[str] = []
    seen: set[str] = set()
    for text in synth:
        reasons = gate_sample(text, g)
        if reasons:
            rejections[reasons[0]] += 1  # attribute to the first reason
            continue
        if g["dedup"]:
            key = _normalise(text)
            if key in seen:
                rejections["duplicate"] += 1
                continue
            seen.add(key)
        kept.append(text)

    report: dict = {
        "n_input": len(synth),
        "n_clean": len(kept),
        "pass_rate": len(kept) / len(synth) if synth else 0.0,
        "rejections": dict(rejections),
    }

    if kept and perplexity_cfg is not None and perplexity_cfg.get("enabled", True):
        ppls = perplexity(
            kept,
            device=device,
            **{k: v for k, v in perplexity_cfg.items() if k != "enabled"},
        )
        finite = [p for p in ppls if math.isfinite(p)]
        report["perplexity"] = {
            "model": perplexity_cfg.get("model_id", "gpt2-large"),
            "mean": sum(finite) / len(finite) if finite else None,
            "median": sorted(finite)[len(finite) // 2] if finite else None,
        }

    if kept:
        from llm_replay.metrics import diversity

        report["diversity"] = diversity.compute_panel(
            kept, real, embed_model=embed_model, device=device, with_mauve=with_mauve
        )
    return kept, report


def perplexity(
    texts: list[str],
    model_id: str = "gpt2-large",
    device: str | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    lora: str | None = None,
) -> list[float]:
    """Per-sample perplexity under a reference model (lower = better fit).

    Default use: coherence under a fixed reference (``model_id``, e.g. gpt2-large). With
    ``lora`` set, applies a LoRA adapter on top of ``model_id`` (base + adapter) — used to
    score a *fine-tuned* model on held-out domain text without merging. For a VLM base like
    Qwen3.5, ``AutoModelForCausalLM`` loads the text tower, matching how the adapter was trained.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to(device).eval()
    )
    if lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora).eval()

    out: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, :-1, :]
        labels = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:].bool()
        ce = F.cross_entropy(
            logits.transpose(1, 2), labels, reduction="none"
        ).masked_fill(~mask, 0.0)
        nll = ce.sum(1) / mask.sum(1).clamp(min=1)
        out.extend(nll.exp().float().tolist())
    return out
