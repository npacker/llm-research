# `llm_replay.training` — LoRA fine-tuning for forgetting / generative replay

Continued causal-LM **LoRA** fine-tuning that mixes domain / general / synthetic corpora at
configurable ratios, to study **catastrophic forgetting** and whether **synthetic replay**
mitigates it. Driven by [`scripts/train.py`](../../../scripts/train.py); configs in
[`configs/train/`](../../../configs/train/). Uses `transformers.Trainer` + `peft` (`trl` not installed).

## Modules
- **`data.py`** — `mix_corpora(corpora, total_samples, seed)` (row-count-weighted mix, reuses
  `llm_replay.corpus.load_corpus`) + `tokenize_for_lm` (causal-LM tokenisation).
- **`sft.py`** — `build_lora_config`, `build_training_args`, `train_lora` (load → LoRA → Trainer →
  save **adapter**; optional `merge=True`).

## Conditions (research plan Areas 2/5; single-generation)
Each is a corpus mix in `configs/train/*.yaml` (roles: `domain` | `general` | `synthetic`):

| Config | Mix | Tests |
|--------|-----|-------|
| `domain_only` | domain 1.0 | forgetting **baseline** |
| `domain_general` | domain 0.5 + general 0.5 | real-data replay mitigation |
| `domain_synthetic` | domain 0.5 + synthetic 0.5 | substitution (synthetic ↔ real general) |
| `domain_general_synthetic` | 0.5 / 0.25 / 0.25 | augmentation |
| `synthetic_only` | synthetic 1.0 | completeness |

Reference = the un-fine-tuned base model. Defaults: medical domain
(`hf:MedRAG/textbooks:train:content`), general (`hf:Salesforce/wikitext:wikitext-103-raw-v1:train:text`),
synthetic = a `generate.py`→`validate.py` `clean.jsonl` (pass `--synthetic`).

## Run + eval (adapter + vLLM LoRA — NOT merge)
```sh
# Train (saves a LoRA adapter under runs/train_<cfg>_<ts>/adapter)
python scripts/train.py --config configs/train/domain_synthetic.yaml --model Qwen/Qwen3.5-4B \
    --synthetic runs/gen1_*/clean.jsonl
# Forgetting (general battery) + domain gain (medical battery) on base + adapter via vLLM:
python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm \
    --lora runs/train_*/adapter --lora-rank 16 --config configs/eval/canary.yaml  --generation domain_synthetic
python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm \
    --lora runs/train_*/adapter --lora-rank 16 --config configs/eval/medical.yaml --generation domain_synthetic
# Compare base vs conditions:
python scripts/forgetting_report.py --base base=runs/gen0_canary_*/eval/results.json \
    --runs domain_only=... domain_general=... domain_synthetic=...
```

## Why adapter + vLLM-LoRA (not merge-then-eval)
`scripts/evaluate.py --lora` loads the **base** model (its own config) and applies the adapter via
vLLM's `lora_local_path`. This is the right path for **Qwen3.5** because:
- vLLM **rejects a merged Qwen3.5 checkpoint** — merging via `AutoModelForCausalLM` saves a
  *text-only* `Qwen3_5TextConfig`, but vLLM's loader for this arch expects the multimodal wrapper.
- The **HF backend on a merged model is pathologically slow** for Qwen3.5: it's a linear-attention/
  conv hybrid and, without `flash-linear-attention`/`causal-conv1d`, falls back to a slow torch path.
- Adapters are tiny (~MBs) and keep vLLM's fast kernels. `train.py --merge` still writes a standalone
  merged checkpoint for HF/portability if needed.

## Notes
- Continued-LM on **raw text** (LM loss) — instruction-format SFT is a later variant.
- Single GPU; LoRA `target_modules: all-linear`. `--limit` for smoke runs.
- Domain gain also via held-out medical perplexity: `generation/validation.py:perplexity(texts,
  model_id=<merged>)` (needs a merged checkpoint; pass `--merge`).
- **Not built:** the recursive multi-generation loop (chain generate→validate→train).
