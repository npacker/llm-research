"""`build_sft_config` / `build_lora_config` — the SFT/LoRA config contract.

CPU-only, no network: builds the config objects (no model load, no training) and
asserts the fields that keep the TRL `SFTTrainer` path equivalent to raw-text
continued-LM training — full-sequence loss (no completion masking), `text`-column
tokenisation, truncation at `max_len`, no packing, and our save-it-ourselves policy.
The end-to-end training run is validated by a manual GPU smoke test (see scripts/README).
"""

from pathlib import Path

from llm_core.training.sft import build_lora_config, build_sft_config


def test_sft_config_defaults_match_rawtext_lm():
    cfg = build_sft_config(Path("/tmp/out"), {}, max_len=512)
    assert cfg.dataset_text_field == "text"  # mix_corpora emits a `text` column
    assert cfg.max_length == 512  # truncation length
    assert cfg.packing is False  # no cross-document concatenation
    # plain-text dataset -> loss over the whole sequence (no prompt/completion mask)
    assert cfg.completion_only_loss in (None, False)
    assert cfg.save_strategy == "no"  # train_lora saves the adapter/merge itself
    assert cfg.report_to == []  # "none" normalises to an empty list


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
