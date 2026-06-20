#!/usr/bin/env python
"""LoRA continued-LM fine-tuning for the forgetting / generative-replay study.

Mixes domain / general / synthetic corpora at config ratios, fine-tunes a LoRA adapter
(causal-LM loss on raw text), merges it into the base, and saves the merged model under
runs/ — ready for scripts/evaluate.py (forgetting = general battery; domain gain = medical
battery + held-out perplexity).

Conditions are just different corpus mixes (configs/train/*.yaml):
  domain_only | domain_general | domain_synthetic | domain_general_synthetic | synthetic_only

Examples
--------
Forgetting baseline (domain only)::

    python scripts/train.py --config configs/train/domain_only.yaml --model Qwen/Qwen3.5-4B

Substitution (synthetic stands in for general); fill the synthetic role::

    python scripts/train.py --config configs/train/domain_synthetic.yaml --model Qwen/Qwen3.5-4B \
        --synthetic runs/gen1_token_edt_<ts>/clean.jsonl

Smoke (tiny)::

    python scripts/train.py --config configs/train/domain_only.yaml --model Qwen/Qwen3.5-4B --limit 64

Single GPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
sys.path.insert(0, str(REPO_ROOT / "src"))  # package not installed (package = false)


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
        "--limit", type=int, default=None, help="Cap total_samples (smoke tests)"
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
        help="Also save a standalone merged checkpoint (HF/portability; vLLM can't load merged Qwen3.5)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
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
    from llm_replay.training import data, sft

    mixed, role_counts = data.mix_corpora(
        corpora, total_samples=total, seed=cfg.get("seed", 0)
    )
    print(f"[train] config={cfg['name']} model={args.model}")
    print(
        f"[train] mix={role_counts} total={len(mixed)} max_len={cfg.get('max_len', 1024)}"
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (DEFAULT_RUNS_DIR / f"train_{cfg['name']}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = sft.train_lora(
        args.model,
        mixed,
        lora_cfg=cfg.get("lora", {}),
        train_cfg=cfg.get("training", {}),
        max_len=cfg.get("max_len", 1024),
        output_dir=out_dir,
        merge=args.merge,
    )
    summary.update(
        {"config": cfg, "model": args.model, "mix": role_counts, "n": len(mixed)}
    )
    (out_dir / "meta.json").write_text(json.dumps(summary, indent=2, default=str))

    print(
        f"\n[train] loss={summary['train_loss']:.4f} "
        f"trainable={summary['trainable_params'] / 1e6:.1f}M/{summary['total_params'] / 1e9:.2f}B"
    )
    print(f"[train] adapter -> {summary['adapter_dir']}")
    # Eval the BASE model + adapter via vLLM (base keeps its own config → vLLM-loadable
    # + fast kernels for Qwen3.5's linear attention; no merge needed).
    print(
        f"[train] next: python scripts/evaluate.py --model {args.model} --backend vllm "
        f"--lora {summary['adapter_dir']} --lora-rank {summary['lora_rank']} "
        f"--config configs/eval/canary.yaml --generation {cfg['name']}"
    )


if __name__ == "__main__":
    main()
