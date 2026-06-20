"""LoRA continued-LM fine-tuning (TRL `SFTTrainer` + peft) — single GPU.

`train_lora` loads a base model, attaches a LoRA adapter, and trains on the raw-text
corpus mix with causal-LM loss; it saves the adapter (and, optionally, a merged
checkpoint). The LoRA target modules and the merged-checkpoint caveat are derived from
the model `profile` (auto-detected), not hardcoded per architecture.

This uses TRL's `SFTTrainer`: it owns tokenisation (the `text` column, truncated to
`max_len`), the LM data collator, and the peft wiring. The dataset is plain text (no
prompt/completion split), so `completion_only_loss` stays off — loss is taken over the
whole sequence, matching continued-LM pre-training on raw text.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from ..models import ModelProfile


def _loss_curve(log_history: list[dict]) -> list[dict]:
    """Compact (step, train_loss?, eval_*_loss?) trace from the Trainer's log history.

    The Trainer logs train loss (``loss``) and eval loss in separate records. Captures the
    single-set ``eval_loss`` *and* per-set ``eval_<name>_loss`` keys (e.g. ``eval_domain_loss``,
    ``eval_general_loss`` when ``eval_dataset`` is a dict) so ``meta.json`` carries every curve.
    """
    curve = []
    for rec in log_history:
        losses = {}
        if "loss" in rec:
            losses["train_loss"] = rec["loss"]
        losses.update(
            {k: v for k, v in rec.items() if k.startswith("eval_") and k.endswith("_loss")}
        )
        if losses:
            curve.append({"step": rec.get("step"), **losses})
    return curve


def build_lora_config(
    cfg: dict, default_target_modules: str | list[str] = "all-linear"
):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.get("r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        target_modules=cfg.get("target_modules", default_target_modules),
        task_type="CAUSAL_LM",
    )


def build_sft_config(
    output_dir: Path,
    tcfg: dict,
    max_len: int,
    *,
    eval_enabled: bool = False,
    best_metric: str = "eval_loss",
):
    """SFTConfig (a TrainingArguments subclass) for raw-text continued-LM SFT.

    `max_length` truncates each text to `max_len`; `packing` stays off so samples are
    not concatenated across document boundaries (per-sample, padded per batch).

    With ``eval_enabled`` (an eval split was supplied), turn on periodic evaluation so
    training/validation loss are both logged, and keep the best-by-``best_metric`` weights
    (``load_best_model_at_end``) so early stopping can trim overfitting — the main
    no-rehearsal lever against catastrophic forgetting. ``best_metric`` is ``eval_loss`` for
    a single eval set, or ``eval_<name>_loss`` (e.g. ``eval_domain_loss``) when eval is a
    dict of named sets — selection tracks the *domain* loss, never the general/forgetting
    one (which only rises). ``eval_on_start`` logs the step-0 (base-model) losses so a
    base-vs-best Δ is available from the curve. This needs ``save_strategy`` to match
    ``eval_strategy``; the final adapter is still saved explicitly by ``train_lora``.
    Without an eval split, behaviour is unchanged (no eval, save-it-ourselves).
    """
    from trl import SFTConfig

    eval_steps = tcfg.get("eval_steps", 50)
    eval_args = (
        {
            "eval_strategy": "steps",
            "eval_steps": eval_steps,
            "eval_on_start": True,  # step-0 eval ≈ base model (LoRA starts at zero) → Δ
            "per_device_eval_batch_size": tcfg.get(
                "per_device_eval_batch_size", tcfg.get("per_device_train_batch_size", 4)
            ),
            "save_strategy": "steps",
            "save_steps": eval_steps,
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": best_metric,
            "greater_is_better": False,
        }
        if eval_enabled
        else {"save_strategy": "no"}  # we save the adapter (and optional merge) ourselves
    )

    return SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 4),
        num_train_epochs=tcfg.get("num_train_epochs", 1),
        learning_rate=tcfg.get("learning_rate", 2e-4),
        warmup_ratio=tcfg.get("warmup_ratio", 0.03),
        weight_decay=tcfg.get("weight_decay", 0.0),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        bf16=tcfg.get("bf16", True),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", False),
        logging_steps=tcfg.get("logging_steps", 10),
        report_to=tcfg.get("report_to", "none"),
        seed=tcfg.get("seed", 0),
        # SFT-specific: tokenise the mixed dataset's `text` column, truncate, no packing.
        dataset_text_field="text",
        max_length=max_len,
        packing=tcfg.get("packing", False),
        **eval_args,
    )


def train_lora(
    model_id: str,
    mixed_dataset,
    *,
    eval_dataset=None,
    early_stopping_patience: int = 0,
    lora_cfg: dict,
    train_cfg: dict,
    max_len: int,
    output_dir: Path,
    merge: bool = False,
    profile: ModelProfile | None = None,
) -> dict:
    """Train a LoRA adapter; save it to ``output_dir/'adapter'``.

    The adapter (not a merged model) is the default output: it's tiny and lets the
    eval reuse the base model's own config (so vLLM loads it via ``lora_local_path``).
    Pass ``merge=True`` to also write a standalone merged checkpoint (``output_dir/'merged'``).
    For a VLM base (``profile.is_vlm``), merging produces a text-only sub-arch that vLLM
    may not register — so the merged checkpoint is for HF/portability, not vLLM. The LoRA
    target modules default to ``profile.lora_target_modules`` (config can override).

    ``eval_dataset`` may be a single set (logged as ``eval_loss``) or a dict of named sets
    (e.g. ``{"domain": ..., "general": ...}`` → ``eval_domain_loss``/``eval_general_loss``).
    With a dict, best-model selection / early stopping track ``eval_domain_loss`` (domain fit),
    while ``eval_general_loss`` is logged purely as the cheap forgetting proxy. When eval is on,
    the best weights are restored at the end; ``early_stopping_patience > 0`` adds an
    EarlyStoppingCallback. The returned summary carries ``best_eval_loss`` and a ``loss_curve``.
    Returns a summary dict.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
    from trl import SFTTrainer

    profile = profile or ModelProfile()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    callbacks = []
    if eval_dataset is not None and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    # Select best/early-stop by the learning loss. For a dict of named eval sets, track
    # the "domain" set when present (the learning signal — never the general/forgetting
    # one, which only rises), else the first set; for a single set, the plain `eval_loss`.
    # metric_for_best_model must name a metric the Trainer actually logs, or transformers
    # raises KeyError at the first eval — so derive it from the keys, don't assume "domain".
    if isinstance(eval_dataset, dict):
        learning_set = "domain" if "domain" in eval_dataset else next(iter(eval_dataset))
        best_metric = f"eval_{learning_set}_loss"
    else:
        best_metric = "eval_loss"

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(
            output_dir,
            train_cfg,
            max_len,
            eval_enabled=eval_dataset is not None,
            best_metric=best_metric,
        ),
        train_dataset=mixed_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(lora_cfg, profile.lora_target_modules),
        callbacks=callbacks or None,
    )
    # After construction trainer.model is the peft-wrapped model.
    trainable, total = trainer.model.get_nb_trainable_parameters()
    result = trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))  # LoRA adapter + tokenizer (small)

    summary = {
        "train_loss": float(result.training_loss),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "adapter_dir": str(adapter_dir),
        "lora_rank": int(lora_cfg.get("r", 16)),
        "loss_curve": _loss_curve(trainer.state.log_history),
    }
    if eval_dataset is not None and trainer.state.best_metric is not None:
        summary["best_eval_loss"] = float(trainer.state.best_metric)
    if merge:
        if profile.is_vlm:
            warnings.warn(
                "Merging a LoRA into a VLM base produces a text-only sub-architecture "
                "that vLLM may not register — eval the merged checkpoint with the HF "
                "backend, or eval base + adapter (lora_local_path) on vLLM instead.",
                stacklevel=2,
            )
        merged_dir = output_dir / "merged"
        trainer.model.merge_and_unload().save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        summary["merged_dir"] = str(merged_dir)
    return summary
