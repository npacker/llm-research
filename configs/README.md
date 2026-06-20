# `configs/` — experiment configuration

One YAML file per experiment **condition**, kept declarative so runs are reproducible.

## `configs/eval/` — capability batteries (built)

Consumed by [`scripts/evaluate.py`](../scripts/evaluate.py) (lm-evaluation-harness). Schema:
`tasks`, `num_fewshot`, `apply_chat_template`, `batch_size`, and a `model_args:` block merged
into the backend args (e.g. `enable_thinking: false` to pin reasoning mode).

| Config | Tasks | Use |
|--------|-------|-----|
| `canary.yaml` | GSM8K, IFEval, GPQA (+`model_args: enable_thinking: false`) | cheap per-generation regression check |
| `full.yaml` | `leaderboard` group (MMLU-Pro, GPQA, BBH, MuSR, MATH-hard, IFEval) + TruthfulQA | milestone battery |
| `full_local.yaml` | MMLU-Pro, MMLU-Redux, C-Eval, MMMLU (non-thinking) | Tier-1 published-comparison battery |

Custom tasks (SuperGPQA, IFBench) are referenced by name with `--tasks ... --include-path eval_tasks/<task>`.

## `configs/gen/` — generation (built)

Consumed by [`scripts/generate.py`](../scripts/generate.py). Schema: `name`, `strategy`
(`fixed` | `seq_edt` | `token_edt`), `prefix` (mode + params), `sampling` (vLLM SamplingParams),
`edt`/`seq_edt` blocks (T0/N/theta etc.), optional `seed_corpus`, and `model_args` for **real
vLLM engine kwargs only** (quantization/kv_cache_dtype — *not* `enable_thinking`; generation is
raw continuation). `base.yaml` documents defaults; `fixed.yaml` / `seq_edt.yaml` / `token_edt.yaml`
are the Area-1 conditions.

## `configs/validate/` — quality validation (built)

`default.yaml` for [`scripts/validate.py`](../scripts/validate.py): per-sample `gates`
(min_chars / max_words / repetition / line-repeat / language / dedup), a `perplexity` block
(reference model for coherence), and `embed_model` for the diversity panel.

## `configs/train/` — LoRA fine-tuning (built)

Consumed by [`scripts/train.py`](../scripts/train.py). Each condition is a `corpora` list of
`{role, spec, weight}` (roles: domain/general/synthetic; a synthetic spec left null is filled by
`--synthetic`); `lora`/`training` fall back to `src/llm_replay/training/sft.py` defaults
(`base.yaml` documents them). Conditions: `domain_only`, `domain_general`, `domain_synthetic`,
`domain_general_synthetic`, `synthetic_only`. `configs/eval/medical.yaml` is the paired domain-gain
battery (medqa_4options, pubmedqa, medmcqa, mmlu-medical).

Corpus specs accept an optional dataset config: `hf:<dataset>:<config>:<split>:<field>` (e.g.
`hf:Salesforce/wikitext:wikitext-103-raw-v1:train:text`) as well as the 4-part `hf:<dataset>:<split>:<field>`.

## Experiment configs (planned)

One YAML per condition for the Area 1–5 sweeps (e.g. Area 4's 16-cell prefix × temperature
design). Not built yet — the generation/training pipeline they'd drive doesn't exist. Intended
layout:

```
configs/
├── eval/                     # capability batteries (above) — built
├── base.yaml                 # shared defaults (model id, seeds, output paths)
├── area1_token_vs_seq/       # one YAML per condition (A/B/C/D)
├── area2_curriculum/
└── ...
```

**Single-GPU defaults** (this repo targets one RTX PRO 6000 Blackwell, 96 GB — see
[`../CLAUDE.md`](../CLAUDE.md)) belong in `base.yaml`:

- `tensor_parallel_size: 1` (never raise — single GPU)
- `kv_cache_dtype: fp8` (longer context)
- LoRA for fine-tuning; prefer fp8/nvfp4 quantization
- `num_proc <= 16` for `datasets`, dataloader `num_workers ~ 8`
