"""`llm_core.serving` — the vllm-serve argv builder + lifecycle manager.

CPU-only, no network/GPU: `build_serve_cmd` is pure; `ServedVLLM`'s launch/health/teardown
is exercised with a monkeypatched subprocess + health probe (no real server).
"""

import pytest

from llm_core.serving import ServedVLLM, build_serve_cmd
from llm_core.serving import server as server_mod


def test_build_serve_cmd_basics():
    cmd = build_serve_cmd(
        "Qwen/Qwen3.5-4B",
        {"p1": "/runs/p1/adapter", "p2": "/runs/p2/adapter"},
        host="127.0.0.1",
        port=8001,
        max_lora_rank=32,
    )
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen3.5-4B"]
    assert "--tensor-parallel-size" not in cmd  # single-GPU rule (CLAUDE.md)
    assert "--enable-lora" in cmd
    assert cmd[cmd.index("--max-lora-rank") + 1] == "32"
    assert cmd[cmd.index("--max-loras") + 1] == "2"  # defaults to #modules
    assert cmd[cmd.index("--port") + 1] == "8001"
    mods = cmd[cmd.index("--lora-modules") + 1 :]
    assert "p1=/runs/p1/adapter" in mods and "p2=/runs/p2/adapter" in mods


def test_build_serve_cmd_no_lora_omits_lora_flags():
    cmd = build_serve_cmd("m", {})
    assert "--enable-lora" not in cmd and "--lora-modules" not in cmd


def test_build_serve_cmd_explicit_max_loras_and_model_len():
    cmd = build_serve_cmd("m", {"a": "x"}, max_loras=4, max_model_len=2048)
    assert cmd[cmd.index("--max-loras") + 1] == "4"
    assert cmd[cmd.index("--max-model-len") + 1] == "2048"


def test_urls_derived_from_host_port():
    srv = ServedVLLM("m", {"a": "x"}, host="127.0.0.1", port=9000)
    assert srv.completions_url == "http://127.0.0.1:9000/v1/completions"
    assert srv.openai_base_url == "http://127.0.0.1:9000/v1"
    assert srv.health_url == "http://127.0.0.1:9000/health"


def test_external_base_url_is_reused_not_launched(monkeypatch):
    launched = []
    monkeypatch.setattr(server_mod.subprocess, "Popen", lambda *a, **k: launched.append(a))
    with ServedVLLM("m", {"a": "x"}, base_url="http://gpu-box:8000/") as srv:
        assert srv.completions_url == "http://gpu-box:8000/v1/completions"
    assert launched == []  # targeted an external server; never spawned one


class _FakeProc:
    def __init__(self, alive=True, returncode=None):
        self.pid = 4321
        self._alive = alive
        self.returncode = returncode

    def poll(self):
        return None if self._alive else self.returncode

    def wait(self, timeout=None):
        self._alive = False
        self.returncode = 0
        return 0


def test_launch_polls_until_healthy_then_terminates(monkeypatch):
    proc = _FakeProc()
    monkeypatch.setattr(server_mod.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(server_mod.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_health(url, timeout=5.0):
        calls["n"] += 1
        return calls["n"] >= 3  # not ready, not ready, ready

    monkeypatch.setattr(server_mod, "_health_ok", fake_health)
    killed = []
    monkeypatch.setattr(server_mod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(server_mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))

    with ServedVLLM("m", {"a": "x"}, port=8123, poll_interval_s=0) as srv:
        assert calls["n"] >= 3  # waited for /health
        assert srv.openai_base_url == "http://127.0.0.1:8123/v1"
    assert killed and killed[0][0] == proc.pid  # SIGTERM to the process group on exit


def test_launch_raises_if_process_exits_before_health(monkeypatch):
    dead = _FakeProc(alive=False, returncode=1)
    monkeypatch.setattr(server_mod.subprocess, "Popen", lambda *a, **k: dead)
    monkeypatch.setattr(server_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(server_mod, "_health_ok", lambda *a, **k: False)
    with pytest.raises(RuntimeError):
        with ServedVLLM("m", {"a": "x"}, poll_interval_s=0):
            pass
