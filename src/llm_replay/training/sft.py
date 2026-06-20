"""LoRA continued-LM fine-tuning (transformers Trainer + peft) — single GPU.

`train_lora` loads a base model, attaches a LoRA adapter, trains on a tokenised raw-text
dataset with causal-LM loss, then **merges the adapter into the base** and saves the merged
model (so the existing `scripts/evaluate.py` can score it unchanged). `trl` is not installed;
this uses transformers `Trainer` directly.
"""

from __future__ import annotations

from pathlib import Path


def build_lora_config(cfg: dict):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.get("r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        target_modules=cfg.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )


def build_training_args(output_dir: Path, tcfg: dict):
    from transformers import TrainingArguments

    return TrainingArguments(
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
        save_strategy="no",  # we save the merged model ourselves
        report_to="none",
        seed=tcfg.get("seed", 0),
    )


def train_lora(
    model_id: str,
    mixed_dataset,
    *,
    lora_cfg: dict,
    train_cfg: dict,
    max_len: int,
    output_dir: Path,
    merge: bool = False,
) -> dict:
    """Train a LoRA adapter; save it to ``output_dir/'adapter'``.

    The adapter (not a merged model) is the default output: it's tiny and lets the
    eval reuse the base model's own config (so vLLM loads it via ``lora_local_path``).
    Pass ``merge=True`` to also write a standalone merged checkpoint (``output_dir/'merged'``;
    note vLLM can't load merged Qwen3.5 — its config becomes text-only — so merged is for
    HF/portability only). Returns a summary dict.
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
    )
    from peft import get_peft_model

    from .data import tokenize_for_lm

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
    targs = build_training_args(output_dir, train_cfg)
    if targs.gradient_checkpointing:
        model.enable_input_require_grads()  # needed for grad-checkpointing + LoRA
    model = get_peft_model(model, build_lora_config(lora_cfg))
    trainable, total = model.get_nb_trainable_parameters()

    tok_ds = tokenize_for_lm(mixed_dataset, tokenizer, max_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=targs, train_dataset=tok_ds, data_collator=collator
    )
    result = trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(
        adapter_dir
    )  # LoRA adapter only (small; eval via base + lora_local_path)
    tokenizer.save_pretrained(adapter_dir)

    summary = {
        "train_loss": float(result.training_loss),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "adapter_dir": str(adapter_dir),
        "lora_rank": int(lora_cfg.get("r", 16)),
    }
    if merge:
        merged_dir = output_dir / "merged"
        model.merge_and_unload().save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        summary["merged_dir"] = str(merged_dir)
    return summary
