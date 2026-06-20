"""Per-sample quality gates for generated corpora.

Includes the regression we fixed: `_max_line_repeat_frac` must return 0 for <3 lines
(a one/two-line sample is not line-repetition degeneration).
"""

from llm_replay.generation.validation import (
    DEFAULT_GATES,
    _max_line_repeat_frac,
    _repetition_ratio,
    coherence_report,
    gate_sample,
)

# Gates with the language check disabled — keeps the gate tests deterministic
# (langdetect is exercised separately).
GATES = {**DEFAULT_GATES, "lang": None}


def test_repetition_ratio_all_unique_is_zero():
    assert _repetition_ratio("a b c d e", 3) == 0.0


def test_repetition_ratio_all_repeated_high():
    assert _repetition_ratio("a a a a a a", 3) > 0.5


def test_repetition_ratio_no_ngrams_is_zero():
    """Fewer tokens than n → no n-grams → 0 (not a crash)."""
    assert _repetition_ratio("a b", 3) == 0.0


def test_max_line_repeat_frac_under_three_lines_is_zero():
    """The fixed bug: 1- or 2-line samples are never 'line-repetition'."""
    assert _max_line_repeat_frac("same\nsame") == 0.0
    assert _max_line_repeat_frac("only one line") == 0.0


def test_max_line_repeat_frac_detects_repeated_lines():
    text = "dup\ndup\ndup\nunique"
    assert _max_line_repeat_frac(text) == 0.75


def test_gate_empty():
    assert gate_sample("   ", GATES) == ["empty"]


def test_gate_too_short():
    assert "too_short" in gate_sample("hi", GATES)


def test_gate_too_long():
    long_text = " ".join(["word"] * (GATES["max_words"] + 10))
    assert "too_long" in gate_sample(long_text, GATES)


def test_gate_repetitive():
    reasons = gate_sample("spam " * 50, GATES)
    assert "repetitive" in reasons


def test_gate_line_repeat():
    text = "\n".join(["the same line"] * 5)
    # Long enough not to trip too_short; intra-line n-grams won't dominate.
    assert "line_repeat" in gate_sample(text, GATES)


def test_gate_clean_sample_passes():
    text = (
        "This is a reasonably long and varied sentence that should pass every "
        "quality gate without raising any rejection reasons at all."
    )
    assert gate_sample(text, GATES) == []


def test_gate_language_detection():
    """With the language gate on, clearly non-target text is flagged."""
    gates = {**DEFAULT_GATES, "lang": "en"}
    german = (
        "Dies ist ein vollständiger deutscher Satz, der eindeutig nicht auf "
        "Englisch verfasst wurde und daher abgelehnt werden sollte."
    )
    assert "wrong_language" in gate_sample(german, gates)


# --- coherence_report: the looping/incoherence verdict used by the Exp-0 sweep --- #

_CLEAN_SAMPLES = [
    "The mitochondria is the membrane-bound organelle that produces most of a cell's ATP.",
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    "Antibiotics treat bacterial infections but have no effect on viral illnesses at all.",
    "The hippocampus plays a central role in forming and consolidating new long-term memories.",
    "Vaccines train the adaptive immune system by presenting harmless antigen fragments.",
    "Insulin is a hormone secreted by the pancreas that regulates blood glucose levels.",
    "The renal nephron filters blood and reabsorbs water, salts, and useful nutrients.",
    "Hemoglobin in red blood cells carries oxygen from the lungs to peripheral tissues.",
    "Neurons communicate across synapses using both electrical and chemical signalling.",
    "The liver metabolizes drugs and detoxifies a wide range of circulating compounds.",
]


def test_coherence_clean_samples_not_degenerate():
    report = coherence_report(_CLEAN_SAMPLES)
    assert report["pass_rate"] == 1.0
    assert report["degenerate"] is False
    assert report["distinct_2"] > 0.3


def test_coherence_looping_is_degenerate():
    """Each sample loops a single line → gated as repetitive/line_repeat → degenerate."""
    looping = ["\n".join(["the same line over and over"] * 6) for _ in range(10)]
    report = coherence_report(looping)
    assert report["pass_rate"] < 0.8
    assert report["degenerate"] is True
    assert set(report["rejections"]) & {"repetitive", "line_repeat"}


def test_coherence_mode_collapse_is_degenerate():
    """Identical (individually coherent) samples pass the gates but collapse diversity."""
    identical = [_CLEAN_SAMPLES[0]] * 10
    report = coherence_report(identical)
    assert report["pass_rate"] == 1.0  # each passes per-sample gates
    assert report["distinct_2"] < 0.3 or report["self_bleu"] > 0.5
    assert report["degenerate"] is True


def test_coherence_thresholds_overridable():
    report = coherence_report(_CLEAN_SAMPLES, thresholds={"min_pass_rate": 1.01})
    assert report["degenerate"] is True  # impossible bar → always degenerate
