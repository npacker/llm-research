"""Prefix-only prompt construction for generative replay (research plan Area 4).

Builds generation prompts from a *seed corpus* of real texts. The prefix conditions
mirror the doc's P1-P4: how much real structure/content is given before the model
continues. The seed corpus doubles as the real/Gen-0 reference for validation.

`build_prompts(seed_texts, mode=..., n=...)` returns a list of `{prompt, prefix_mode,
seed_index}` records (prompt is the raw text fed to the model; the generator applies the
chat template if configured).
"""

from __future__ import annotations

import random

PREFIX_MODES = ("none", "structural", "snippet", "variable", "chat")

# A light structural scaffold for P2 (format markers only, no real content).
_STRUCTURAL_PREFIX = "### Document\n\n"


def _word_prefix(text: str, frac: float) -> str:
    """First `frac` of a text, by whitespace tokens."""
    words = text.split()
    k = max(1, int(len(words) * frac))
    return " ".join(words[:k])


# Default user turn for `chat` mode — generic, no real-data seeding; diversity comes from
# per-sample sampling (temperature/EDT + per-request seeds), not from a corpus.
DEFAULT_CHAT_PROMPT = (
    "Write a detailed, self-contained, factual passage on a topic of general knowledge. "
    "Choose a different topic each time and do not repeat yourself."
)


def build_prompts(
    seed_texts: list[str],
    mode: str = "structural",
    n: int | None = None,
    *,
    snippet_frac: float = 0.5,
    variable_fracs: tuple[float, ...] = (0.1, 0.25, 0.5),
    chat_prompt: str | None = None,
    seed: int = 0,
) -> list[dict]:
    """Construct `n` prompts under one prefix condition.

    Modes:
      - ``none``       (P1): empty prompt — pure unconditioned continuation.
      - ``structural`` (P2): format markers only, no real content.
      - ``snippet``    (P3): the first `snippet_frac` of a real seed doc.
      - ``variable``   (P4): a snippet whose length is sampled from `variable_fracs`.
      - ``chat``       : a generic chat user-turn, **no corpus seeding** — for generative
                         replay from an instruct model (the generator applies the chat
                         template; diversity comes from per-sample sampling).
    """
    if mode not in PREFIX_MODES:
        raise ValueError(f"mode must be one of {PREFIX_MODES}, got {mode!r}")
    if not seed_texts and mode in ("snippet", "variable"):
        raise ValueError(f"mode {mode!r} needs a non-empty seed corpus")

    rng = random.Random(seed)
    count = n if n is not None else (len(seed_texts) if seed_texts else 0)
    records: list[dict] = []
    for i in range(count):
        idx = i % len(seed_texts) if seed_texts else -1
        if mode == "none":
            prompt = ""
        elif mode == "structural":
            prompt = _STRUCTURAL_PREFIX
        elif mode == "chat":
            prompt = chat_prompt or DEFAULT_CHAT_PROMPT
        elif mode == "snippet":
            prompt = _word_prefix(seed_texts[idx], snippet_frac)
        else:  # variable
            prompt = _word_prefix(seed_texts[idx], rng.choice(variable_fracs))
        records.append({"prompt": prompt, "prefix_mode": mode, "seed_index": idx})
    return records
