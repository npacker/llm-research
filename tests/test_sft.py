"""`build_sft_config` / `build_lora_config` — the SFT/LoRA config contract.

CPU-only, no network: builds the config objects (no model load, no training) and
asserts the fields that keep the TRL `SFTTrainer` path equivalent to raw-text
continued-LM training — full-sequence loss (no completion masking), `text`-column
tokenisation, truncation at `max_len`, no packing, and our save-it-ourselves policy.
The end-to-end training run is validated by a manual GPU smoke test (see scripts/README).
"""

from pathlib import Path

from llm_core.training.sft import _loss_curve, build_lora_config, build_sft_config


def test_sft_config_defaults_match_rawtext_lm():
    cfg = build_sft_config(Path("/tmp/out"), {}, max_len=512)
    assert cfg.dataset_text_field == "text"  # mix_corpora emits a `text` column
    assert cfg.max_length == 512  # truncation length
    assert cfg.packing is False  # no cross-document concatenation
    # plain-text dataset -> loss over the whole sequence (no prompt/completion mask)
    assert cfg.completion_only_loss in (None, False)
    assert cfg.save_strategy == "no"  # train_lora saves the adapter/merge itself
    assert cfg.report_to == []  # "none" normalises to an empty list


def test_sft_config_no_eval_by_default():
    """Without an eval split, evaluation stays off and we save the adapter ourselves."""
    cfg = build_sft_config(Path("/tmp/out"), {}, max_len=512)
    assert cfg.eval_strategy == "no"  # HF strategy enums compare equal to their str value
    assert cfg.load_best_model_at_end is False


def test_sft_config_eval_enabled_tracks_and_keeps_best():
    """An eval split turns on periodic eval + best-by-metric checkpointing (early stop)."""
    cfg = build_sft_config(
        Path("/tmp/out"), {"eval_steps": 25}, max_len=512, eval_enabled=True
    )
    assert cfg.eval_strategy == "steps"
    assert cfg.save_strategy == "steps"
    assert cfg.eval_steps == 25
    assert cfg.eval_on_start is True  # step-0 eval ≈ base model → enables base-vs-best Δ
    assert cfg.load_best_model_at_end is True
    assert cfg.metric_for_best_model == "eval_loss"  # default single-set metric
    assert cfg.greater_is_better is False


def test_sft_config_best_metric_override_for_named_evals():
    """With a dict of eval sets, selection tracks the domain loss, not the general one."""
    cfg = build_sft_config(
        Path("/tmp/out"), {}, max_len=512, eval_enabled=True, best_metric="eval_domain_loss"
    )
    assert cfg.metric_for_best_model == "eval_domain_loss"


def test_loss_curve_extracts_train_and_named_eval_losses():
    history = [
        {"step": 0, "eval_domain_loss": 2.2, "eval_general_loss": 1.8, "eval_domain_runtime": 1.0},
        {"step": 10, "loss": 2.0},
        {"step": 10, "eval_domain_loss": 2.0, "eval_general_loss": 1.9},
        {"step": 20, "loss": 1.5},
        {"step": 30, "learning_rate": 1e-4},  # no loss → dropped
        {"train_runtime": 99.0},  # final summary → dropped
    ]
    curve = _loss_curve(history)
    assert curve == [
        {"step": 0, "eval_domain_loss": 2.2, "eval_general_loss": 1.8},
        {"step": 10, "train_loss": 2.0},
        {"step": 10, "eval_domain_loss": 2.0, "eval_general_loss": 1.9},
        {"step": 20, "train_loss": 1.5},
    ]


def test_loss_curve_single_eval_loss_still_works():
    curve = _loss_curve([{"step": 5, "eval_loss": 2.1, "eval_runtime": 1.0}])
    assert curve == [{"step": 5, "eval_loss": 2.1}]


def test_sft_config_passes_through_training_knobs():
    cfg = build_sft_config(
        Path("/tmp/out"),
        {"per_device_train_batch_size": 8, "num_train_epochs": 3, "packing": True},
        max_len=128,
    )
    assert cfg.per_device_train_batch_size == 8
    assert cfg.num_train_epochs == 3
    assert cfg.packing is True


def test_lora_config_defaults_and_profile_targets():
    lc = build_lora_config({}, default_target_modules="all-linear")
    assert lc.r == 16 and lc.lora_alpha == 32
    assert lc.target_modules == "all-linear"  # falls back to profile-supplied default
    assert lc.task_type == "CAUSAL_LM"


def test_lora_config_overrides_win_over_profile():
    lc = build_lora_config(
        {"r": 4, "target_modules": ["q_proj", "v_proj"]},
        default_target_modules="all-linear",
    )
    assert lc.r == 4
    # config beats profile default (peft normalises a list to a set)
    assert set(lc.target_modules) == {"q_proj", "v_proj"}
