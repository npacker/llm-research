# `llm_core.training` — LoRA continued-LM fine-tuning

Continued causal-LM **LoRA** fine-tuning over a row-count-weighted mix of corpora. General
tooling — the roles you mix (`domain` / `general` / `synthetic` / …) and ratios are supplied
by the caller's config, not baked in. Driven by [`scripts/train.py`](../../../scripts/train.py);
configs in [`configs/train/`](../../../configs/train/). Uses `transformers.Trainer` + `peft`
(`trl` not installed).

## Modules
- **`data.py`** — `mix_corpora(corpora, total_samples, seed)` (row-count-weighted mix of any
  roles, reuses `llm_core.corpus.load_corpus`) + `tokenize_for_lm` (causal-LM tokenisation).
- **`sft.py`** — `build_lora_config`, `build_training_args`, `train_lora` (load → LoRA → Trainer →
  save **adapter**; optional `merge=True`).

## Run + eval (adapter + vLLM LoRA — NOT merge)
```sh
# Train (saves a LoRA adapter under runs/train_<cfg>_<ts>/adapter)
python scripts/train.py --config configs/train/<cfg>.yaml --model <base>
# Eval base + adapter via vLLM (no merge needed):
python scripts/evaluate.py --model <base> --backend vllm \
    --lora runs/train_*/adapter --lora-rank 16 --config configs/eval/canary.yaml
```

## Why adapter + vLLM-LoRA (not merge-then-eval)
`scripts/evaluate.py --lora <adapter>` loads the **base** model (with its own config) and applies
the adapter (vLLM `lora_local_path` / HF `peft`) — no merge needed, and adapters are tiny (~MBs).
This matters for **VLM bases** (e.g. Qwen3.5, `Qwen3_5ForConditionalGeneration`, whose config has
both `vision_config` and `text_config`):
- A merged checkpoint made the easy way is **text-only**: `AutoModelForCausalLM` maps Qwen3.5 →
  `Qwen3_5ForCausalLM` (text tower only). vLLM registers **only** the full
  `Qwen3_5ForConditionalGeneration`, so it rejects that text-only sub-architecture.
- A **full** vision+text merge *does* load in vLLM (load via `AutoModelForImageTextToText`, scope
  LoRA to `language_model.*`, merge, save) — but drags the vision tower along as dead weight for a
  text-only fine-tune. The adapter path sidesteps the issue entirely.
- `train.py --merge` writes the (text-only) merged checkpoint for HF / portability — eval **that**
  with `--backend hf`, not vLLM. `train_lora` warns when merging a VLM base (`profile.is_vlm`).

(The HF backend is *not* the problem — it runs on the GPU fine. An earlier 17-min HF run was
`batch_size: auto` thrashing on WSL's over-reported memory, not the model; a fixed batch is quick.)

## Notes
- Continued-LM on **raw text** (LM loss) — instruction-format SFT is a later variant.
- Single GPU; LoRA defaults to `target_modules: all-linear` (config / profile can override).
  `--limit` for quick partial / pilot runs.
- For the forgetting / generative-replay study that uses this tooling — the condition matrix,
  held-out-domain-perplexity metric, and run/eval recipe — see
  [`research/exp1-forgetting-replay.md`](../../../research/exp1-forgetting-replay.md).
