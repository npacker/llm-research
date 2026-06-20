"""Prefix-only prompt construction (the P1-P4 conditions + chat mode)."""

import pytest

from llm_replay.generation.prompts import (
    DEFAULT_CHAT_PROMPT,
    _STRUCTURAL_PREFIX,
    build_prompts,
)

SEED = ["alpha beta gamma delta epsilon", "one two three four five six"]


def test_none_mode_empty_prompts():
    recs = build_prompts(SEED, mode="none", n=3)
    assert len(recs) == 3
    assert all(r["prompt"] == "" for r in recs)
    assert all(r["prefix_mode"] == "none" for r in recs)


def test_structural_mode_uses_scaffold():
    recs = build_prompts(SEED, mode="structural", n=2)
    assert all(r["prompt"] == _STRUCTURAL_PREFIX for r in recs)


def test_chat_mode_default_prompt():
    recs = build_prompts([], mode="chat", n=2)
    assert all(r["prompt"] == DEFAULT_CHAT_PROMPT for r in recs)


def test_chat_mode_custom_prompt():
    recs = build_prompts([], mode="chat", n=1, chat_prompt="custom turn")
    assert recs[0]["prompt"] == "custom turn"


def test_snippet_mode_is_prefix_of_seed():
    recs = build_prompts(SEED, mode="snippet", n=1, snippet_frac=0.4)
    # 5 words * 0.4 = 2 words
    assert recs[0]["prompt"] == "alpha beta"
    assert recs[0]["seed_index"] == 0


def test_variable_mode_deterministic_under_seed():
    a = build_prompts(SEED, mode="variable", n=4, seed=7)
    b = build_prompts(SEED, mode="variable", n=4, seed=7)
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]


def test_n_defaults_to_seed_length():
    recs = build_prompts(SEED, mode="structural")
    assert len(recs) == len(SEED)


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        build_prompts(SEED, mode="bogus")


def test_snippet_requires_seed_corpus():
    with pytest.raises(ValueError):
        build_prompts([], mode="snippet", n=2)
