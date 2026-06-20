"""`profile_from` — auto-detect arch capabilities from config/tokenizer stand-ins.

Pure, no network: this is the "new non-Qwen arch is config-only" guarantee. We feed
fake config/tokenizer objects mimicking the few HF attributes the detector reads.
"""

from llm_core.models import ModelProfile, profile_from


class FakeConfig:
    def __init__(self, architectures=None, vision_config=None):
        self.architectures = architectures or []
        self.vision_config = vision_config


class FakeTokenizer:
    """Instruct tokenizer: has a chat template that accepts enable_thinking."""

    chat_template = "{{ messages }}"

    def apply_chat_template(self, msgs, **kwargs):
        return "rendered"


class ThinkinglessTokenizer:
    """Chat template exists but rejects the enable_thinking kwarg (TypeError)."""

    chat_template = "{{ messages }}"

    def apply_chat_template(self, msgs, *, add_generation_prompt, tokenize):
        return "rendered"


class BaseTokenizer:
    """Raw base model: no chat template at all."""

    chat_template = None

    def apply_chat_template(self, msgs, **kwargs):  # pragma: no cover - not reached
        raise AssertionError("base model has no chat template")


def test_text_instruct_model():
    p = profile_from(FakeConfig(["LlamaForCausalLM"]), FakeTokenizer())
    assert p.is_vlm is False
    assert p.has_chat_template is True
    assert p.supports_thinking is True
    assert p.eval_model_args() == {"enable_thinking": False}


def test_vlm_detected_by_vision_config():
    p = profile_from(
        FakeConfig(["FooForCausalLM"], vision_config={"x": 1}), FakeTokenizer()
    )
    assert p.is_vlm is True


def test_vlm_detected_by_arch_suffix():
    p = profile_from(FakeConfig(["Qwen3VLForConditionalGeneration"]), BaseTokenizer())
    assert p.is_vlm is True
    assert p.has_chat_template is False
    assert p.supports_thinking is False
    assert p.eval_model_args() == {}


def test_thinkingless_chat_template():
    """Chat template present but no enable_thinking support → supports_thinking False."""
    p = profile_from(FakeConfig(["MistralForCausalLM"]), ThinkinglessTokenizer())
    assert p.has_chat_template is True
    assert p.supports_thinking is False
    assert p.eval_model_args() == {}


def test_no_tokenizer():
    p = profile_from(FakeConfig(["LlamaForCausalLM"]))
    assert p.has_chat_template is False
    assert p.supports_thinking is False


def test_overrides_win_last():
    p = profile_from(
        FakeConfig(["LlamaForCausalLM"]),
        FakeTokenizer(),
        overrides={"is_vlm": True, "lora_target_modules": ["q_proj", "v_proj"]},
    )
    assert p.is_vlm is True
    assert p.lora_target_modules == ["q_proj", "v_proj"]


def test_default_lora_target_modules():
    p = profile_from(FakeConfig(["LlamaForCausalLM"]), FakeTokenizer())
    assert p.lora_target_modules == "all-linear"


def test_render_chat_passes_enable_thinking_only_when_supported():
    captured = {}

    class RecordingTok:
        chat_template = "x"

        def apply_chat_template(self, msgs, **kwargs):
            captured.update(kwargs)
            return "ok"

    ModelProfile(supports_thinking=True).render_chat(RecordingTok(), "hi")
    assert captured.get("enable_thinking") is False

    captured.clear()
    ModelProfile(supports_thinking=False).render_chat(RecordingTok(), "hi")
    assert "enable_thinking" not in captured
