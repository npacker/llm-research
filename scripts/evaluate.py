#!/usr/bin/env python
"""Per-generation capability evaluation via EleutherAI lm-evaluation-harness.

Runs a fixed capability battery (defined by a YAML in ``configs/eval/``) against a
model checkpoint and writes results to ``runs/``. The point is *regression
detection* across recursive generations: keep the harness/config fixed and compare
each generation's scores to the Generation-0 baseline.

Examples
--------
Canary battery on a HF checkpoint (loads weights directly)::

    python scripts/evaluate.py --config configs/eval/canary.yaml \
        --model Qwen/Qwen2.5-7B-Instruct --generation 0

Same, but against an already-running ``vllm serve`` endpoint (OpenAI-compatible)::

    python scripts/evaluate.py --config configs/eval/canary.yaml \
        --backend local-completions \
        --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1/completions

Quick partial run (2 items/task, no GPU work beyond model load)::

    python scripts/evaluate.py --config configs/eval/canary.yaml \
        --model <id> --limit 2

Single GPU only: the vLLM backend pins ``tensor_parallel_size=1`` per the repo's
hardware constraints (see CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from llm_core.evaluation import summarize

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
        help="Eval config YAML (e.g. configs/eval/canary.yaml)",
    )
    p.add_argument(
        "--model",
        required=True,
        help="HF repo id or local checkpoint path (the model under test)",
    )
    p.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "vllm", "local-completions"],
        help="hf: load weights directly | vllm: in-process vLLM | local-completions: hit a running vllm serve endpoint",
    )
    p.add_argument(
        "--generation",
        default="0",
        help="Recursive-generation label for this checkpoint (e.g. 0, 1, 2)",
    )
    p.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task override (default: the config's task list)",
    )
    p.add_argument(
        "--include-path",
        type=Path,
        default=None,
        help="Directory of custom lm-eval task YAMLs to register (e.g. eval_tasks/supergpqa)",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="For --backend local-completions: the served /v1/completions URL",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="For --backend local-completions: tokenizer to use (defaults to --model). "
        "Set to the base model id when --model is a served LoRA-adapter name.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit items per task for a quick partial run (omit for the full battery)",
    )
    p.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Override the config's few-shot setting",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to runs/<gen>_<config>_<timestamp>/eval",
    )
    p.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9, help="vLLM backend only"
    )
    p.add_argument("--max-model-len", type=int, default=None, help="vLLM backend only")
    p.add_argument(
        "--lora",
        default=None,
        help="Path to a LoRA adapter to apply to the base --model (vllm: lora_local_path; hf: peft). "
        "Avoids merging — base loads with its own config.",
    )
    p.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="max LoRA rank for the vllm backend (>= adapter r)",
    )
    p.add_argument(
        "--wandb-project",
        default=None,
        help="If set, log the summary to this wandb project",
    )
    return p.parse_args()


def lm_eval_model_args(
    args: argparse.Namespace,
    profile_args: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the lm-eval ``model_args`` dict for the chosen backend.

    Merge order (later wins): backend defaults → ``profile_args`` (policy derived from
    the model's detected capabilities, e.g. ``enable_thinking: false`` for a reasoning
    model) → ``extra`` (the config's ``model_args:`` block). So configs only need to
    spell out *non-default* overrides; the profile supplies the rest.
    """
    if args.backend == "hf":
        base = {"pretrained": args.model, "dtype": "bfloat16"}
        if args.lora:
            base["peft"] = args.lora  # HFLM applies the PEFT adapter to the base
    elif args.backend == "vllm":
        # Single GPU: tensor_parallel_size is pinned to 1 (see CLAUDE.md).
        base = {
            "pretrained": args.model,
            "tensor_parallel_size": 1,
            "dtype": "auto",
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
        if args.max_model_len is not None:
            base["max_model_len"] = args.max_model_len
        if args.lora:
            # Load the base (its own config → vLLM-compatible) + apply the adapter.
            base["lora_local_path"] = args.lora
            base["max_lora_rank"] = args.lora_rank
    else:  # local-completions: evaluate a model already served by `vllm serve`.
        if not args.base_url:
            raise SystemExit(
                "--backend local-completions requires --base-url (e.g. http://localhost:8000/v1/completions)"
            )
        base = {
            "model": args.model,
            "base_url": args.base_url,
            "num_concurrent": 8,
            # A served LoRA's --model is the adapter *name* (not a resolvable HF id);
            # --tokenizer supplies the base id so lm-eval can tokenise.
            "tokenizer": args.tokenizer or args.model,
        }
    if profile_args:
        base.update(profile_args)
    if extra:
        base.update(extra)
    return base


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else cfg["tasks"]
    num_fewshot = (
        args.num_fewshot if args.num_fewshot is not None else cfg.get("num_fewshot")
    )

    # Imported here so --help works without importing torch/vllm.
    import lm_eval

    from llm_core.models import resolve_profile

    # Register custom task YAMLs (e.g. SuperGPQA) when an include path is given.
    task_manager = None
    if args.include_path:
        from lm_eval.tasks import TaskManager

        task_manager = TaskManager(include_path=str(args.include_path))

    # Auto-detect arch capabilities (loads config/tokenizer, not weights); a config
    # `model:` block overrides. Supplies the thinking-off policy + chat-template default
    # so eval configs only spell out non-default overrides. For local-completions the
    # --model is a served adapter *name* (not an HF id), so detect from --tokenizer (the
    # base id) when it's given.
    profile = resolve_profile(args.tokenizer or args.model, overrides=cfg.get("model"))
    model_args = lm_eval_model_args(
        args, profile.eval_model_args(), cfg.get("model_args")
    )
    apply_chat = cfg.get("apply_chat_template")
    if apply_chat is None:
        apply_chat = profile.has_chat_template
    print(
        f"[evaluate] config={cfg['name']} gen={args.generation} backend={args.backend}"
    )
    print(f"[evaluate] tasks={tasks}")
    print(f"[evaluate] model_args={model_args}")

    results = lm_eval.simple_evaluate(
        model=args.backend,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=cfg.get("batch_size", "auto"),
        limit=args.limit,
        apply_chat_template=apply_chat,
        gen_kwargs=cfg.get("gen_kwargs"),
        task_manager=task_manager,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        DEFAULT_RUNS_DIR / f"gen{args.generation}_{cfg['name']}_{stamp}" / "eval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # lm-eval results include non-JSON types (e.g. numpy); default=str is a safe dump.
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))

    rows = summarize(results)
    print(f"\n{'task':32s} {'metric':24s} value")
    print("-" * 66)
    for task, metric, val in rows:
        print(f"{task:32s} {metric:24s} {val:.4f}")
    print(f"\n[evaluate] wrote {results_path}")

    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=f"gen{args.generation}_{cfg['name']}",
            config={
                "generation": args.generation,
                "battery": cfg["name"],
                "tasks": tasks,
            },
        )
        run.log({f"{t}/{m}": v for t, m, v in rows})
        run.finish()


if __name__ == "__main__":
    main()
