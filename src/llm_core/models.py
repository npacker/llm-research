"""Model-agnostic capability detection.

Per-architecture behaviour (is it a VLM? does the tokenizer have a chat template?
does it support a thinking toggle? which modules does LoRA target?) used to be
scattered across the generation/eval/training code as Qwen-specific literals and
inline try/excepts. This centralises it: capabilities are **auto-detected** from the
HF config/tokenizer so a new architecture is config-only, and a `model:` override
block is the single escape hatch (no per-model registry).

`profile_from` is **pure** — it takes already-loaded objects and does no I/O, so it
is unit-testable with lightweight stand-ins. `resolve_profile` is the convenience
wrapper that loads `AutoConfig` (+ tokenizer) and calls it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelProfile:
    """Detected capabilities of a model, with derived eval/generation policy.

    Capabilities (`is_vlm`, `has_chat_template`, `supports_thinking`) are *what the
    model can do*; the helpers below derive *policy* from them (e.g. "thinking off
    for answer-extraction evals").
    """

    is_vlm: bool = False
    has_chat_template: bool = False
    supports_thinking: bool = False
    lora_target_modules: str | list[str] = "all-linear"

    def render_chat(self, tokenizer, content: str) -> str:
        """Render a single user turn through the chat template (thinking off).

        Centralises the ``enable_thinking`` try/except that used to live in the
        generator: pass the kwarg only when the model supports it.
        """
        msgs = [{"role": "user", "content": content}]
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if self.supports_thinking:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(msgs, **kwargs)

    def eval_model_args(self) -> dict:
        """Backend ``model_args`` implied by the profile for capability evals.

        Policy: turn thinking off for answer-extraction tasks (a verbose reasoning
        preamble overruns the answer-extraction token budget — see CLAUDE.md). Only
        emitted when the model actually supports the toggle.
        """
        return {"enable_thinking": False} if self.supports_thinking else {}


def _arch_names(config) -> list[str]:
    return list(getattr(config, "architectures", None) or [])


def _detect_is_vlm(config) -> bool:
    """A vision-language model: has a ``vision_config`` or a ``*ForConditionalGeneration`` arch."""
    if getattr(config, "vision_config", None) is not None:
        return True
    return any(a.endswith("ForConditionalGeneration") for a in _arch_names(config))


def _detect_supports_thinking(tokenizer) -> bool:
    """True iff the chat template accepts ``enable_thinking`` (renders without TypeError)."""
    if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
        return False
    msgs = [{"role": "user", "content": "x"}]
    try:
        tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        return False
    except Exception:
        # Some templates raise other errors only on edge content; absence of a
        # TypeError on the kwarg is what we're testing, so treat as supported.
        return True
    return True


def profile_from(
    config=None, tokenizer=None, overrides: dict | None = None
) -> ModelProfile:
    """Build a `ModelProfile` from already-loaded HF objects (pure — no I/O).

    `config` is an ``AutoConfig`` (or any object exposing ``architectures`` /
    ``vision_config``); pass ``None`` to profile from a tokenizer alone (the arch-based
    ``is_vlm`` detection is then skipped). `tokenizer` is optional (chat-template +
    thinking detection need it). `overrides` (a config ``model:`` block) is applied last
    and wins.
    """
    profile = ModelProfile(
        is_vlm=_detect_is_vlm(config),
        has_chat_template=(
            tokenizer is not None
            and getattr(tokenizer, "chat_template", None) is not None
        ),
        supports_thinking=_detect_supports_thinking(tokenizer),
    )
    if overrides:
        for key, val in overrides.items():
            if hasattr(profile, key):
                setattr(profile, key, val)
    return profile


def resolve_profile(
    model_id: str, tokenizer=None, overrides: dict | None = None
) -> ModelProfile:
    """Load ``AutoConfig`` (+ tokenizer if not supplied) for ``model_id``, then profile it."""
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id)
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            tokenizer = None  # base models without a tokenizer still get a profile
    return profile_from(config, tokenizer, overrides)
