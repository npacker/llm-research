"""`llm_core.generation.served.generate_served` — the HTTP twin of generator.generate.

CPU-only, no network: the OpenAI client is stubbed, so we assert the request fields
(model = served LoRA name, per-index seed, sampling params) and that the returned records
match generator.generate's shape (text/strategy/temperature + original keys preserved).
"""

import pytest

from llm_core.generation import served
from llm_core.models import ModelProfile


class _FakeChoice:
    def __init__(self, text):
        self.text = text


class _FakeResp:
    def __init__(self, text):
        self.choices = [_FakeChoice(text)]


class _FakeCompletions:
    def __init__(self, log):
        self._log = log

    def create(self, **kw):
        self._log.append(kw)
        return _FakeResp(f"out-for-{kw['prompt']}")


class _FakeClient:
    LOG: list = []

    def __init__(self, *a, **k):
        self.init_kwargs = k
        self.completions = _FakeCompletions(_FakeClient.LOG)


class _FakeTok:
    def apply_chat_template(self, msgs, **kw):
        return "<TPL>" + msgs[0]["content"]


@pytest.fixture
def fake_openai(monkeypatch):
    _FakeClient.LOG = []
    monkeypatch.setattr("openai.OpenAI", _FakeClient)
    return _FakeClient


def test_request_fields_and_record_shape(fake_openai):
    prompts = [{"prompt": "a", "seed_index": 0}, {"prompt": "b", "seed_index": 1}]
    recs = served.generate_served(
        prompts,
        base_url="http://h/v1",
        served_model="pt7",
        profile=ModelProfile(has_chat_template=False),
        tokenizer=_FakeTok(),
        gen_kwargs={"temperature": 0.8, "top_p": 0.95, "max_tokens": 256, "seed": 7},
        concurrency=1,
    )
    log = sorted(fake_openai.LOG, key=lambda r: r["prompt"])
    assert [r["model"] for r in log] == ["pt7", "pt7"]  # served by LoRA name
    assert {r["seed"] for r in log} == {7, 8}  # seed0 + index
    assert log[0]["temperature"] == 0.8 and log[0]["top_p"] == 0.95
    assert log[0]["max_tokens"] == 256
    # record shape mirrors generator.generate
    assert all(r["strategy"] == "fixed" and r["temperature"] == 0.8 for r in recs)
    assert recs[0]["text"] == "out-for-a" and recs[0]["seed_index"] == 0  # original keys kept


def test_chat_template_applied_client_side_when_supported(fake_openai):
    served.generate_served(
        [{"prompt": "hello"}],
        base_url="http://h/v1",
        served_model="m",
        profile=ModelProfile(has_chat_template=True),
        tokenizer=_FakeTok(),
        gen_kwargs={"temperature": 0.7, "max_tokens": 16, "seed": 0},
        concurrency=1,
    )
    assert fake_openai.LOG[0]["prompt"] == "<TPL>hello"  # render_chat ran before the POST


def test_no_chat_template_sends_raw_prompt(fake_openai):
    served.generate_served(
        [{"prompt": "raw"}],
        base_url="http://h/v1",
        served_model="m",
        profile=ModelProfile(has_chat_template=False),
        tokenizer=_FakeTok(),
        gen_kwargs={"temperature": 0.7, "max_tokens": 16, "seed": 0},
        concurrency=1,
    )
    assert fake_openai.LOG[0]["prompt"] == "raw"
