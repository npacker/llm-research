"""Exp-0 sweep runner: cfg→grid wiring + Stage-B served orchestration.

The runner lives under `experiments/` (not an installed package), so load it from its path.
Its pure primitives (grid enumeration, Pareto front, the forgetting score, result-delta
bucketing) now live in `llm_core`/`llm_replay` and are tested there (test_sweep.py,
test_forgetting.py, test_evaluation.py); this file covers what stays runner-local — that the
study config maps onto `enumerate_grid`, and the shared-server Stage-B flow. Top-level imports
are stdlib + yaml + the pure `llm_core`/`llm_replay` helpers (pandas/torch/vllm are deferred
inside functions), so this is CPU-only.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "experiments/exp0_forgetting_signal/sweep.py"


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("exp0_sweep", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load_sweep_module()


def test_build_points_main_grid_plus_rank_subsweep_dedups_center():
    """The runner assembles this study's axes (+ rank sub-sweep) for enumerate_grid."""
    cfg = {
        "grid": {
            "learning_rate": [1e-4, 2e-4],
            "num_train_epochs": [1],
            "num_samples": [1000],
        },
        "ranks_main": [16],
        "rank_subsweep": {
            "at": {"learning_rate": 2e-4, "num_train_epochs": 1, "num_samples": 1000},
            "ranks": [8, 16, 32],  # r16 here duplicates the main-grid center → deduped
        },
    }
    ids = [p["id"] for p in sweep.build_points(cfg)]
    # 2 main (lr1e-4/r16, lr2e-4/r16) + r8/r32 at the center = 4 unique points.
    assert len(ids) == len(set(ids)) == 4
    assert "lr0.0002_e1_n1000_r16" in ids  # center appears once
    assert "lr0.0002_e1_n1000_r8" in ids and "lr0.0002_e1_n1000_r32" in ids


# --- Stage B served orchestration: one shared vLLM init for eval + coherence --- #

_STAGEB_CFG = {
    "model": "Qwen/Qwen3.5-4B",
    "eval_config": "configs/eval/canary.yaml",
    "grid": {"learning_rate": [2e-4], "num_train_epochs": [1], "num_samples": [5000]},
    "ranks_main": [16],
    "eval_limit": 200,
    "coherence_samples": 64,
}


def test_served_eval_cmd_targets_server_by_lora_name(tmp_path):
    pt = {"id": "lr0.0002_e1_n5000_r16", "lora_rank": 16}
    cmd = sweep._served_eval_cmd(
        _STAGEB_CFG, pt, tmp_path, "http://127.0.0.1:8000/v1/completions"
    )
    assert cmd[cmd.index("--backend") + 1] == "local-completions"
    assert cmd[cmd.index("--model") + 1] == pt["id"]  # served LoRA-module name
    assert (
        cmd[cmd.index("--tokenizer") + 1] == _STAGEB_CFG["model"]
    )  # base for tokenisation
    assert cmd[cmd.index("--base-url") + 1] == "http://127.0.0.1:8000/v1/completions"
    assert cmd[cmd.index("--limit") + 1] == "200"


def test_served_eval_cmd_omits_limit_when_unset(tmp_path):
    cfg = {"eval_config": "c.yaml", "model": "m"}
    cmd = sweep._served_eval_cmd(cfg, {"id": "p", "lora_rank": 8}, tmp_path, "u")
    assert "--limit" not in cmd


def test_served_coherence_cmd_uses_served_model(tmp_path):
    cmd = sweep._served_coherence_cmd(
        _STAGEB_CFG,
        {"id": "pt1", "lora_rank": 16},
        tmp_path,
        "http://127.0.0.1:8000/v1",
    )
    assert cmd[cmd.index("--served-model") + 1] == "pt1"
    assert cmd[cmd.index("--base-url") + 1] == "http://127.0.0.1:8000/v1"
    assert (
        cmd[cmd.index("--model") + 1] == _STAGEB_CFG["model"]
    )  # base id for the tokenizer
    assert cmd[cmd.index("--num-samples") + 1] == "64"


class _FakeServer:
    """Stand-in for ServedVLLM: no launch, exposes the two URLs phase_stage_b needs."""

    completions_url = "http://127.0.0.1:8000/v1/completions"
    openai_base_url = "http://127.0.0.1:8000/v1"

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stageb_cfg(tmp_path):
    return {**_STAGEB_CFG, "out_root": str(tmp_path)}


def test_phase_stage_b_runs_pending_point(tmp_path, monkeypatch):
    cfg = _stageb_cfg(tmp_path)
    points = sweep.build_points(cfg)
    monkeypatch.setattr("llm_core.serving.ServedVLLM", _FakeServer)
    ran = []
    monkeypatch.setattr(sweep, "run", lambda cmd, dry_run: ran.append(cmd) or 1.0)

    sweep.phase_stage_b(cfg, points, dry_run=False)

    flat = [tok for cmd in ran for tok in cmd]
    assert any("evaluate.py" in tok for tok in flat)
    assert any("coherence_check.py" in tok for tok in flat)
    assert "local-completions" in flat  # eval went through the shared server


def test_phase_stage_b_skips_completed_points(tmp_path, monkeypatch):
    cfg = _stageb_cfg(tmp_path)
    points = sweep.build_points(cfg)
    pid = points[0]["id"]
    # Pre-write both Stage-B outputs → resumable skip should run nothing.
    (tmp_path / pid / "eval").mkdir(parents=True)
    (tmp_path / pid / "eval" / "results.json").write_text("{}")
    (tmp_path / pid / "coherence.json").write_text("{}")
    monkeypatch.setattr("llm_core.serving.ServedVLLM", _FakeServer)
    ran = []
    monkeypatch.setattr(sweep, "run", lambda cmd, dry_run: ran.append(cmd) or 1.0)

    sweep.phase_stage_b(cfg, points, dry_run=False)

    assert ran == []  # both eval and coherence already done
