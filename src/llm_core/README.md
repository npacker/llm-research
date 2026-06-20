# `src/llm_core/` — general, reusable LLM infrastructure

Project-agnostic building blocks shared across this repo's tooling and reusable by any
LLM project on the **vLLM + HuggingFace** stack. Nothing here assumes the
generative-replay study — that research layer lives in
[`../llm_replay/`](../llm_replay/) and imports this package.

| Module        | Purpose                                                                                          |
|---------------|--------------------------------------------------------------------------------------------------|
| `models.py`   | Model capability detection (VLM? chat template? thinking toggle? LoRA targets) → eval/gen policy  |
| `evaluation.py` | Flatten lm-evaluation-harness results into primary-metric rows                                  |
| `corpus.py`   | Corpus loader (local `.txt` / `.jsonl` / `hf:<dataset>:<split>:<field>` spec)                     |
| `metrics/`    | Distribution/diversity metrics over corpora (MAUVE, Vendi, prdc, Self-BLEU, Distinct-n, Fréchet, MMD, KL, tail-mass) — reference-free + reference-based |
| `generation/` | EDT temperature (token/seq/fixed) + vLLM text generator                                           |
| `training/`   | LoRA continued-LM fine-tuning + row-count-weighted corpus mixing                                  |

Intra-package imports are relative (`from ..corpus import …`); consumers use absolute
`import llm_core…`. Heavy deps (sentence-transformers, mauve, torch, vllm, transformers,
peft) are deferred into functions so importing a module stays cheap. A CPU-only,
no-network `pytest` suite lives in [`../../tests/`](../../tests/); the embedding/MAUVE/GPU
paths are validated by manual GPU runs.
