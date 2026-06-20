"""CLI hyperparameter overrides on scripts/train.py layer over the config blocks.

`scripts/` isn't an installed package, so load train.py from its path. Its top-level
imports are stdlib + yaml (torch/trl are deferred inside main), so this stays CPU-only.
"""

import importlib.util
import math
from pathlib import Path

import pytest
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_script", REPO_ROOT / "scripts/train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load_train_module()


def _args(**kw):
    base = dict(
        lr=None, epochs=None, weight_decay=None, lora_rank=None, lora_alpha=None,
        lora_dropout=None, eval_steps=None, wandb_project=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_overrides_win_over_config():
    cfg = {
        "training": {"learning_rate": 2e-4, "num_train_epochs": 1, "weight_decay": 0.0},
        "lora": {"r": 16, "lora_alpha": 32},
    }
    out = train.apply_overrides(cfg, _args(lr=5e-4, epochs=3, lora_rank=8, weight_decay=0.1))
    assert out["training"]["learning_rate"] == 5e-4
    assert out["training"]["num_train_epochs"] == 3
    assert out["training"]["weight_decay"] == 0.1
    assert out["lora"]["r"] == 8
    assert out["lora"]["lora_alpha"] == 32  # untouched override falls through to config


def test_unset_overrides_preserve_config():
    cfg = {"training": {"learning_rate": 1e-4}, "lora": {"r": 32}}
    out = train.apply_overrides(cfg, _args())
    assert out["training"]["learning_rate"] == 1e-4
    assert out["lora"]["r"] == 32


def test_wandb_project_sets_report_to():
    out = train.apply_overrides({"training": {}, "lora": {}}, _args(wandb_project="exp0"))
    assert out["training"]["report_to"] == "wandb"


def test_overrides_on_empty_config_create_blocks():
    out = train.apply_overrides({}, _args(lr=3e-4, lora_rank=4))
    assert out["training"]["learning_rate"] == 3e-4
    assert out["lora"]["r"] == 4


def test_parse_args_exposes_expected_flags(monkeypatch):
    """parse_args wires up the new sweep flags with sensible defaults."""
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--config", "configs/train/domain_only.yaml", "--model", "m", "--lr", "5e-4"],
    )
    ns = train.parse_args()
    assert ns.lr == 5e-4
    assert ns.val_frac == 0.05  # default validation split
    assert ns.early_stopping_patience == 0
    assert ns.general_heldout is None and ns.general_heldout_n == 256
    for attr in ("epochs", "lora_rank", "lora_alpha", "lora_dropout", "eval_steps", "wandb_project"):
        assert hasattr(ns, attr)


# --- heldout_perplexity_deltas: base-vs-best ppl Δ per eval set, from the loss curve --- #


def test_heldout_perplexity_deltas_dual_evals_use_best_domain_step():
    # step 0 = base (eval_on_start); domain loss is lowest at step 10, then overfits up at 20.
    # The deployed checkpoint is the best-domain one (step 10) — general Δ is read there.
    curve = [
        {"step": 0, "eval_domain_loss": math.log(10), "eval_general_loss": math.log(8)},
        {"step": 10, "eval_domain_loss": math.log(6), "eval_general_loss": math.log(9)},
        {"step": 20, "eval_domain_loss": math.log(7), "eval_general_loss": math.log(12)},
    ]
    out = train.heldout_perplexity_deltas(curve)
    assert out["domain"]["base"] == pytest.approx(10) and out["domain"]["best"] == pytest.approx(6)
    assert out["domain"]["delta"] == pytest.approx(-4)  # learned (ppl fell 10 → 6)
    # general read at the best-domain step (10), not the min-general step (0)
    assert out["general"]["best"] == pytest.approx(9)
    assert out["general"]["delta"] == pytest.approx(1)  # forgot (ppl rose 8 → 9 at deployed ckpt)


def test_heldout_perplexity_deltas_single_eval_loss():
    curve = [
        {"step": 0, "eval_loss": math.log(5)},
        {"step": 10, "eval_loss": math.log(4)},
    ]
    out = train.heldout_perplexity_deltas(curve)
    assert set(out) == {"val"}
    assert round(out["val"]["delta"], 6) == -1


def test_heldout_perplexity_deltas_no_eval_is_empty():
    assert train.heldout_perplexity_deltas([{"step": 10, "train_loss": 2.0}]) == {}


def test_heldout_perplexity_deltas_worsening_uses_deployed_step_not_base():
    # Domain loss is lowest at the base step (step 0, eval_on_start) — the model only
    # worsens. load_best_model_at_end can't restore step 0 (never saved), so the deployed
    # checkpoint is the best *non-base* step (10). The Δ must reflect that worse adapter
    # (positive = didn't learn), not a deceptive Δ=0 from comparing base against itself.
    curve = [
        {"step": 0, "eval_domain_loss": math.log(5)},   # base, lowest
        {"step": 10, "eval_domain_loss": math.log(6)},  # deployed (best saved)
        {"step": 20, "eval_domain_loss": math.log(8)},
    ]
    out = train.heldout_perplexity_deltas(curve)
    assert out["domain"]["best"] == pytest.approx(6)  # step 10, not step 0
    assert out["domain"]["delta"] == pytest.approx(1)  # +1 = worsened over base


def test_heldout_perplexity_deltas_only_base_step_falls_back():
    # No later eval (e.g. eval_steps > total steps): the only record is step-0 → Δ 0.
    out = train.heldout_perplexity_deltas([{"step": 0, "eval_domain_loss": math.log(5)}])
    assert out["domain"]["delta"] == 0
