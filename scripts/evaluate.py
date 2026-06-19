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

Quick smoke test (2 items/task, no GPU work beyond model load)::

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
        "--limit",
        type=int,
        default=None,
        help="Limit items per task (smoke tests only — not for real runs)",
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
        "--wandb-project",
        default=None,
        help="If set, log the summary to this wandb project",
    )
    return p.parse_args()


def build_model_args(args: argparse.Namespace, extra: dict | None = None) -> dict:
    """Assemble the lm-eval ``model_args`` dict for the chosen backend.

    ``extra`` (from the config's ``model_args:`` block) is merged on top, so a
    config can set e.g. ``enable_thinking: false`` / ``think_end_token: </think>``
    for reasoning models without changing this script.
    """
    if args.backend == "hf":
        base = {"pretrained": args.model, "dtype": "bfloat16"}
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
    else:  # local-completions: evaluate a model already served by `vllm serve`.
        if not args.base_url:
            raise SystemExit(
                "--backend local-completions requires --base-url (e.g. http://localhost:8000/v1/completions)"
            )
        base = {
            "model": args.model,
            "base_url": args.base_url,
            "num_concurrent": 8,
            "tokenizer": args.model,
        }
    if extra:
        base.update(extra)
    return base


def summarize(results: dict) -> list[tuple[str, str, float]]:
    """Flatten lm-eval results into (task, metric, value) rows for primary metrics.

    lm-eval metric keys are ``"<metric>,<filter>"`` (e.g. ``exact_match,strict-match``
    for gsm8k, ``prompt_level_strict_acc,none`` for ifeval). We keep every numeric,
    non-stderr metric and label it with its filter when it isn't the trivial ``none``.
    """
    rows: list[tuple[str, str, float]] = []
    for task, metrics in sorted(results.get("results", {}).items()):
        for key, val in metrics.items():
            if not isinstance(val, (int, float)) or "," not in key:
                continue  # skips `alias` (str) and `sample_len` (no filter)
            metric, _, flt = key.partition(",")
            if metric.endswith("_stderr"):
                continue
            label = metric if flt in ("none", "") else f"{metric} ({flt})"
            rows.append((task, label, float(val)))
    return rows


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else cfg["tasks"]
    num_fewshot = (
        args.num_fewshot if args.num_fewshot is not None else cfg.get("num_fewshot")
    )

    # Imported here so --help works without importing torch/vllm.
    import lm_eval

    # Register custom task YAMLs (e.g. SuperGPQA) when an include path is given.
    task_manager = None
    if args.include_path:
        from lm_eval.tasks import TaskManager

        task_manager = TaskManager(include_path=str(args.include_path))

    model_args = build_model_args(args, cfg.get("model_args"))
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
        apply_chat_template=cfg.get("apply_chat_template", False),
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
