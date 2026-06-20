#!/usr/bin/env python
"""Generate a synthetic corpus via dynamic-temperature (EDT) replay.

Strategies (set by the config's `strategy`): `fixed`, `seq_edt`, `token_edt`
(research plan Area 1). Prompts are built from a seed corpus under a prefix condition
(Area 4). Output is a JSONL corpus under runs/, ready for scripts/validate.py.

Examples
--------
Token-level EDT, 1000 samples::

    python scripts/generate.py --config configs/gen/token_edt.yaml \
        --model Qwen/Qwen3.5-0.8B --generation 1

Quick partial run (20 samples)::

    python scripts/generate.py --config configs/gen/fixed.yaml \
        --model Qwen/Qwen3.5-0.8B --limit 20

Single GPU only: tensor_parallel_size is pinned to 1.
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
        help="Generation config YAML (configs/gen/*.yaml)",
    )
    p.add_argument(
        "--model", required=True, help="HF model id or local checkpoint path"
    )
    p.add_argument(
        "--generation", default="1", help="Recursive-generation label (e.g. 1, 2, 3)"
    )
    p.add_argument(
        "--num-samples", type=int, default=None, help="Override config's num_samples"
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Cap samples for a quick partial run"
    )
    p.add_argument(
        "--seed-corpus",
        default=None,
        help="Override config's seed_corpus spec (file or hf:...)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to runs/gen<N>_<config>_<ts>/",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    strategy = cfg["strategy"]
    prefix = cfg.get("prefix", {"mode": "structural"})
    num_samples = args.limit or args.num_samples or cfg.get("num_samples", 1000)

    # Deferred so --help works without torch/vllm.
    from llm_core import corpus
    from llm_core.generation import generator
    from llm_replay.generation import prompts

    seed_spec = args.seed_corpus or cfg.get("seed_corpus")
    seed_texts: list[str] = []
    if prefix["mode"] in ("snippet", "variable"):
        if not seed_spec:
            raise SystemExit(
                f"prefix mode {prefix['mode']!r} needs a seed_corpus (config or --seed-corpus)"
            )
        seed_texts = corpus.load_corpus(
            seed_spec, cfg.get("text_field"), limit=num_samples
        )

    prompt_records = prompts.build_prompts(
        seed_texts,
        mode=prefix["mode"],
        n=num_samples,
        snippet_frac=prefix.get("snippet_frac", 0.5),
        variable_fracs=tuple(prefix.get("variable_fracs", (0.1, 0.25, 0.5))),
        chat_prompt=prefix.get("chat_prompt"),
        seed=cfg.get("sampling", {}).get("seed", 0),
    )

    # Auto-detect arch capabilities (chat template / thinking toggle); a config
    # `model:` block overrides. Drives chat-template rendering in the generator.
    llm, profile, model_args = generator.build_engine(
        args.model,
        strategy=strategy,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        model_args_extra=cfg.get("model_args"),
        profile_overrides=cfg.get("model"),
    )
    print(f"[generate] config={cfg['name']} gen={args.generation} strategy={strategy}")
    print(
        f"[generate] model={args.model} prefix={prefix['mode']} n={len(prompt_records)}"
    )
    print(f"[generate] model_args={model_args}")

    records = generator.generate(
        llm,
        prompt_records,
        strategy=strategy,
        gen_kwargs=cfg.get("sampling", {}),
        edt=cfg.get("edt"),
        seq_edt=cfg.get("seq_edt"),
        apply_chat_template=cfg.get("apply_chat_template"),
        profile=profile,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        DEFAULT_RUNS_DIR / f"gen{args.generation}_{cfg['name']}_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    with samples_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "config": cfg,
                "model": args.model,
                "generation": args.generation,
                "seed_corpus": seed_spec,
                "n": len(records),
            },
            indent=2,
            default=str,
        )
    )

    n_empty = sum(1 for r in records if not r["text"].strip())
    print(
        f"\n[generate] wrote {samples_path} ({len(records)} samples, {n_empty} empty)"
    )
    print(
        f"[generate] next: python scripts/validate.py --corpus {samples_path}"
        + (f" --real {seed_spec}" if seed_spec else "")
    )


if __name__ == "__main__":
    main()
