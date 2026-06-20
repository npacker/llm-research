#!/usr/bin/env python
"""LoRA continued-LM fine-tuning for the forgetting / generative-replay study.

Mixes domain / general / synthetic corpora at config ratios, fine-tunes a LoRA adapter
(causal-LM loss on raw text), and saves the adapter under runs/. Also auto-reports held-out
domain perplexity (base vs base+adapter) — the direct domain-learning signal. Eval forgetting
(general battery) + domain transfer (medical battery) with scripts/evaluate.py --lora.

Conditions are just different corpus mixes (configs/train/*.yaml):
  domain_only | domain_general | domain_synthetic | domain_general_synthetic | synthetic_only

Examples
--------
Forgetting baseline (domain only)::

    python scripts/train.py --config configs/train/domain_only.yaml --model Qwen/Qwen3.5-4B

Substitution (synthetic stands in for general); fill the synthetic role::

    python scripts/train.py --config configs/train/domain_synthetic.yaml --model Qwen/Qwen3.5-4B \
        --synthetic runs/gen1_token_edt_<ts>/clean.jsonl

Quick partial / pilot run (tiny)::

    python scripts/train.py --config configs/train/domain_only.yaml --model Qwen/Qwen3.5-4B --limit 64

Single GPU only.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Training config YAML (configs/train/*.yaml)",
    )
    p.add_argument(
        "--model", required=True, help="Base HF model id or local checkpoint path"
    )
    p.add_argument(
        "--synthetic",
        default=None,
        help="Spec for the 'synthetic' corpus role (generate.py/validate.py output)",
    )
    p.add_argument(
        "--num-samples", type=int, default=None, help="Override config total_samples"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total_samples for a quick partial / pilot run",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to runs/train_<config>_<ts>/",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="Also save a standalone merged checkpoint (HF/portability; vLLM may not load a merged VLM base)",
    )
    p.add_argument(
        "--general-heldout",
        default=None,
        help="General-corpus spec (hf:.../file) held out as a forgetting probe — its eval "
        "loss/perplexity rise vs. base = catastrophic forgetting (cheap, no battery)",
    )
    p.add_argument(
        "--general-heldout-n",
        type=int,
        default=256,
        help="Rows from --general-heldout for the forgetting-probe eval set",
    )
    # Hyperparameter overrides — applied on top of the config's training/lora blocks so a
    # sweep is one train.py invocation per point (no per-setting config files).
    p.add_argument("--lr", type=float, default=None, help="Override training.learning_rate")
    p.add_argument("--epochs", type=float, default=None, help="Override training.num_train_epochs")
    p.add_argument("--weight-decay", type=float, default=None, help="Override training.weight_decay")
    p.add_argument("--lora-rank", type=int, default=None, help="Override lora.r")
    p.add_argument("--lora-alpha", type=int, default=None, help="Override lora.lora_alpha")
    p.add_argument("--lora-dropout", type=float, default=None, help="Override lora.lora_dropout")
    # Validation split → train + validation loss / perplexity (and early stopping).
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.05,
        help="Fraction of the mixed corpus held out for validation loss (0 disables eval)",
    )
    p.add_argument("--eval-steps", type=int, default=None, help="Override training.eval_steps")
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after N evals without eval_loss improvement (0 = off; needs --val-frac > 0)",
    )
    p.add_argument(
        "--wandb-project",
        default=None,
        help="If set, report train/eval loss to this wandb project (sets report_to=wandb)",
    )
    return p.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Layer CLI hyperparameter overrides on top of the config's training/lora blocks."""
    tcfg = dict(cfg.get("training", {}))
    lcfg = dict(cfg.get("lora", {}))
    for key, val in (
        ("learning_rate", args.lr),
        ("num_train_epochs", args.epochs),
        ("weight_decay", args.weight_decay),
        ("eval_steps", args.eval_steps),
    ):
        if val is not None:
            tcfg[key] = val
    for key, val in (
        ("r", args.lora_rank),
        ("lora_alpha", args.lora_alpha),
        ("lora_dropout", args.lora_dropout),
    ):
        if val is not None:
            lcfg[key] = val
    if args.wandb_project:
        tcfg["report_to"] = "wandb"
    cfg["training"], cfg["lora"] = tcfg, lcfg
    return cfg


def heldout_perplexity_deltas(loss_curve: list[dict]) -> dict:
    """Base-vs-deployed held-out perplexity Δ per eval set, read off the training loss curve.

    `eval_on_start` logs the step-0 (≈ base model) losses, so the base step is the earliest.
    The deployed checkpoint is the lowest-*domain*-loss one that `load_best_model_at_end`
    can actually restore — i.e. a *saved* checkpoint, never step 0 (no checkpoint is written
    at `eval_on_start`). So we pick the best domain step **excluding the base step**; the
    reported Δ then matches the adapter that ships (a point that only worsens over the base
    gets a positive domain Δ here and is correctly read as "didn't learn", rather than a
    deceptive Δ=0 from comparing the base against itself). For each set we report its
    perplexity at the base step and at that deployed step — **domain** Δ = learning, **general**
    Δ = the forgetting incurred at the deployed checkpoint. Negative Δ = perplexity fell.
    """
    keys = sorted(
        {k for e in loss_curve for k in e if k.startswith("eval_") and k.endswith("_loss")}
    )
    if not keys:
        return {}
    by_key: dict[str, dict] = {}
    for e in loss_curve:
        for k in keys:
            if k in e:
                by_key.setdefault(k, {})[e.get("step")] = e[k]

    domain_key = "eval_domain_loss" if "eval_domain_loss" in keys else keys[0]
    steps = by_key[domain_key]
    base_step = min(steps)
    # Deployable = best domain loss among saved (non-base) checkpoints; fall back to the
    # base step only when there is no later eval (e.g. eval_steps > total optimiser steps).
    non_base = {s: v for s, v in steps.items() if s != base_step}
    best_step = min(non_base, key=non_base.get) if non_base else base_step

    out = {}
    for k in keys:
        name = k[len("eval_") : -len("_loss")] or "val"  # eval_domain_loss→domain, eval_loss→val
        per = by_key[k]
        if base_step in per and best_step in per:
            base, best = math.exp(per[base_step]), math.exp(per[best_step])
            out[name] = {"base": base, "best": best, "delta": best - base}
    return out


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(yaml.safe_load(args.config.read_text()), args)
    total = args.limit or args.num_samples or cfg.get("total_samples", 10000)

    # Fill the synthetic role's spec from --synthetic when the config left it null.
    corpora = [dict(c) for c in cfg["corpora"]]
    for c in corpora:
        if c.get("role") == "synthetic" and not c.get("spec"):
            if not args.synthetic:
                raise SystemExit(
                    "config has a synthetic corpus with no spec — pass --synthetic <spec>"
                )
            c["spec"] = args.synthetic

    # Deferred so --help works without torch.
    from llm_core import corpus
    from llm_core.models import resolve_profile
    from llm_core.training import data, sft

    mixed, role_counts = data.mix_corpora(
        corpora, total_samples=total, seed=cfg.get("seed", 0)
    )
    # Hold out a validation slice so train + validation loss are both tracked. Use an
    # integer row count and require >=2 rows on each side; tiny pilot runs (--limit) with
    # too few rows skip eval gracefully (a 1-row eval is too noisy to select a checkpoint on).
    # With --general-heldout, add a second named eval set (a general corpus the run never
    # trains on) → the Trainer logs eval_domain_loss + eval_general_loss as separate curves;
    # the general one is the cheap forgetting proxy.
    train_ds, eval_ds = mixed, None
    n_val = round(len(mixed) * args.val_frac) if args.val_frac else 0
    if n_val >= 2 and len(mixed) - n_val >= 2:
        split = mixed.train_test_split(test_size=n_val, seed=cfg.get("seed", 0))
        train_ds, eval_ds = split["train"], split["test"]
        if args.general_heldout:
            from datasets import Dataset

            # Keep the domain eval slice domain-role-pure so eval_domain_loss tracks domain
            # *learning* even for a multi-role mix (exp1 rehearsal), not the blended corpus.
            domain_val = eval_ds
            if "role" in eval_ds.column_names:
                pure = eval_ds.filter(lambda r: r["role"] == "domain")
                domain_val = pure if len(pure) else eval_ds
            gen = corpus.load_corpus(args.general_heldout, limit=args.general_heldout_n)
            eval_ds = {"domain": domain_val, "general": Dataset.from_dict({"text": gen})}
    if isinstance(eval_ds, dict):
        val_desc = ", ".join(f"{k}={len(v)}" for k, v in eval_ds.items())
    else:
        val_desc = str(len(eval_ds)) if eval_ds else "0"
    print(f"[train] config={cfg['name']} model={args.model}")
    print(
        f"[train] mix={role_counts} total={len(mixed)} "
        f"train={len(train_ds)} val=({val_desc}) max_len={cfg.get('max_len', 1024)}"
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (DEFAULT_RUNS_DIR / f"train_{cfg['name']}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect arch capabilities (LoRA targets / merge caveat); config `model:` overrides.
    profile = resolve_profile(args.model, overrides=cfg.get("model"))

    summary = sft.train_lora(
        args.model,
        train_ds,
        eval_dataset=eval_ds,
        early_stopping_patience=args.early_stopping_patience,
        lora_cfg=cfg.get("lora", {}),
        train_cfg=cfg.get("training", {}),
        max_len=cfg.get("max_len", 1024),
        output_dir=out_dir,
        merge=args.merge,
        profile=profile,
    )
    # Held-out perplexity Δ per eval set (base vs best checkpoint), straight from the eval
    # curve — domain Δ = learning, general Δ = forgetting. Replaces the old post-hoc
    # base-vs-adapter perplexity passes (no extra model reloads; both axes from one source).
    if summary.get("loss_curve"):
        summary["heldout_perplexity"] = heldout_perplexity_deltas(summary["loss_curve"])
    if "best_eval_loss" in summary:
        summary["val_perplexity"] = math.exp(summary["best_eval_loss"])
    summary.update(
        {"config": cfg, "model": args.model, "mix": role_counts, "n": len(mixed)}
    )

    (out_dir / "meta.json").write_text(json.dumps(summary, indent=2, default=str))

    val = (
        f" val_loss={summary['best_eval_loss']:.4f} val_ppl={summary['val_perplexity']:.2f}"
        if "best_eval_loss" in summary
        else ""
    )
    print(
        f"\n[train] train_loss={summary['train_loss']:.4f}{val} "
        f"trainable={summary['trainable_params'] / 1e6:.1f}M/{summary['total_params'] / 1e9:.2f}B"
    )
    for name, p in (summary.get("heldout_perplexity") or {}).items():
        tag = "learning" if name == "domain" else "forgetting" if name == "general" else ""
        print(f"[train] {name} ppl: base={p['base']:.2f} -> best={p['best']:.2f} (Δ{p['delta']:+.2f}) {tag}")
    print(f"[train] adapter -> {summary['adapter_dir']}")
    # Eval the BASE model + adapter via vLLM (base keeps its own config → vLLM-loadable
    # with the arch's native kernels; no merge needed).
    print(
        f"[train] next: python scripts/evaluate.py --model {args.model} --backend vllm "
        f"--lora {summary['adapter_dir']} --lora-rank {summary['lora_rank']} "
        f"--config configs/eval/canary.yaml --generation {cfg['name']}"
    )


if __name__ == "__main__":
    main()
