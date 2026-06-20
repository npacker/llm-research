#!/usr/bin/env python
"""Failure-mode check on a fine-tuned model: does it loop / degenerate / collapse?

Samples N continuations from the model (optionally base + LoRA adapter, no merge) and
scores them with the per-sample gates (repetition, line-repeat, length) plus two
reference-free diversity signals (distinct-2, self-BLEU). Emits a `coherence.json` with a
boolean ``degenerate`` verdict — the failure-mode gate the Exp-0 sweep uses to disqualify
a hyperparameter setting that learned the domain but broke the model.

Examples
--------
Check a tuned adapter (base + LoRA via vLLM, no merge)::

    python scripts/coherence_check.py --model Qwen/Qwen3.5-4B \
        --lora runs/exp0/lr2e-4_e1_n5000/adapter --num-samples 200

Or sample from an already-running ``vllm serve`` (no engine init here — amortised across a
sweep) by the served LoRA-module name::

    python scripts/coherence_check.py --model Qwen/Qwen3.5-4B \
        --base-url http://localhost:8000/v1 --served-model lr2e-4_e1_n5000 --num-samples 200

Single GPU only: tensor_parallel_size is pinned to 1.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="Base HF model id or local checkpoint")
    p.add_argument("--lora", default=None, help="LoRA adapter dir to apply to --model (no merge)")
    p.add_argument("--num-samples", type=int, default=200, help="Number of probe generations")
    p.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per sample")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--prefix-mode",
        default="chat",
        help="Prompt condition (chat|none|structural|snippet|variable); chat needs no corpus",
    )
    p.add_argument("--seed-corpus", default=None, help="Seed corpus for snippet/variable modes")
    p.add_argument("--lora-rank", type=int, default=32, help="max LoRA rank for vLLM")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument(
        "--base-url",
        default=None,
        help="Sample from a running vllm serve (OpenAI /v1 base URL) instead of an "
        "in-process engine — no GPU init here. Used by the Exp-0 sweep to share one engine.",
    )
    p.add_argument(
        "--served-model",
        default=None,
        help="With --base-url: the served model / LoRA-module name to sample (defaults to --model)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <lora>/.. or runs/coherence_<ts>/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Deferred so --help works without torch/vllm.
    from llm_core import corpus
    from llm_replay.generation import prompts, validation

    seed_texts: list[str] = []
    if args.prefix_mode in ("snippet", "variable"):
        if not args.seed_corpus:
            raise SystemExit(f"prefix mode {args.prefix_mode!r} needs --seed-corpus")
        seed_texts = corpus.load_corpus(args.seed_corpus, limit=args.num_samples)

    prompt_records = prompts.build_prompts(
        seed_texts, mode=args.prefix_mode, n=args.num_samples, seed=args.seed
    )
    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }

    if args.base_url:
        # HTTP path: reuse a running engine, no GPU init. Templating/profile are CPU-only.
        from transformers import AutoTokenizer

        from llm_core.generation import served
        from llm_core.models import resolve_profile

        served_model = args.served_model or args.model
        profile = resolve_profile(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print(
            f"[coherence] served model={served_model} base_url={args.base_url} "
            f"n={len(prompt_records)}"
        )
        records = served.generate_served(
            prompt_records,
            base_url=args.base_url,
            served_model=served_model,
            profile=profile,
            tokenizer=tokenizer,
            gen_kwargs=gen_kwargs,
        )
        target = served_model
    else:
        from vllm.lora.request import LoRARequest

        from llm_core.generation import generator

        llm, profile, _ = generator.build_engine(
            args.model,
            strategy="fixed",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            model_args_extra={"max_lora_rank": args.lora_rank} if args.lora else None,
            enable_lora=bool(args.lora),
        )
        lora_request = LoRARequest("adapter", 1, args.lora) if args.lora else None
        print(f"[coherence] model={args.model} lora={args.lora} n={len(prompt_records)}")
        records = generator.generate(
            llm,
            prompt_records,
            strategy="fixed",
            gen_kwargs=gen_kwargs,
            profile=profile,
            lora_request=lora_request,
        )
        target = args.lora

    texts = [r["text"] for r in records]
    report = validation.coherence_report(texts)
    report.update({"model": args.model, "lora": args.lora, "served_model": args.served_model})

    out_dir = args.output_dir or (
        Path(target).parent
        if target and not args.base_url
        else DEFAULT_RUNS_DIR
        / f"coherence_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coherence.json").write_text(json.dumps(report, indent=2, default=str))

    print(
        f"[coherence] pass_rate={report['pass_rate']:.2f} distinct_2={report['distinct_2']:.3f} "
        f"self_bleu={report['self_bleu']:.3f} -> "
        f"{'DEGENERATE' if report['degenerate'] else 'ok'}"
    )
    print(f"[coherence] rejections={report['rejections']}")
    print(f"[coherence] wrote {out_dir / 'coherence.json'}")


if __name__ == "__main__":
    main()
