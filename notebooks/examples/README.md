# `notebooks/examples/` — a cookbook of the external stack

Runnable Jupyter notebooks that demonstrate **idiomatic, raw usage of the third-party
libraries this project is built on** — vLLM, transformers, TRL, peft,
lm-evaluation-harness, `datasets`, and the diversity-metric libs (sentence-transformers,
vendi-score, prdc, mauve-text, sacrebleu).

These are **reference / documentation** notebooks. They call the external APIs directly
(`from vllm import LLM`, `from trl import SFTTrainer`, …) — they do **not** use this repo's
`llm_core` / `llm_replay` wrappers. Each notebook ends with a *project pointer* noting
where `llm_core`/`scripts` wrap that same primitive, so you can jump from "how does vLLM
work" to "how does this repo use it".

For the wrapped, config-driven versions of these operations, see [`scripts/`](../../scripts/)
and [`CLAUDE.md`](../../CLAUDE.md).

## Running

The project env lives at `/opt/venv` and is already on `PATH` (no activation needed). In
JupyterLab, pick the **"Python (llm-research)"** kernel. To run a notebook headless:

```sh
jupyter nbconvert --to notebook --execute notebooks/examples/01_vllm_offline.ipynb
```

Notebooks default to the real research model `Qwen/Qwen3.5-4B` (a `MODEL = …` constant at
the top of each — swap it for a smaller model for a quick smoke run) and cap work with a
small `LIMIT` / `max_tokens` / `num_samples`. Any artifacts are written under `runs/`
(gitignored).

**Hardware:** single RTX PRO 6000 Blackwell (96 GB). Notebooks never set
`tensor_parallel_size` / `--tensor-parallel-size` (leave TP at 1) and prefer native FP8
where shown. See [`CLAUDE.md`](../../CLAUDE.md) for the full constraints.

## Notebooks

| # | Notebook | Demonstrates |
|---|----------|--------------|
| 00 | [datasets & tokenizers](00_datasets_and_tokenizers.ipynb) | `datasets.load_dataset`, `AutoTokenizer`, chat templates + the thinking toggle |
| 01 | [vLLM offline](01_vllm_offline.ipynb) | `vllm.LLM` + `SamplingParams` — offline batched generation |
| 02 | [vLLM serving + OpenAI client](02_vllm_serve_openai.ipynb) | `vllm serve` (subprocess + `/health`), the `openai` client, serving LoRA adapters |
| 03 | [HF backend generation](03_hf_transformers_generate.ipynb) | `AutoModelForCausalLM.generate()` — greedy/sampling/batching |
| 04 | [vLLM custom logits processor](04_vllm_custom_logits_processor.ipynb) | vLLM `logits_processors` extension point (the EDT mechanism) |
| 05 | [LoRA training](05_train_lora_peft_trl.ipynb) | `peft.LoraConfig` + `trl.SFTTrainer` |
| 06 | [Full fine-tuning](06_train_fft_trl.ipynb) | `trl.SFTTrainer` full-weight SFT (no peft) |
| 07 | [Benchmarking](07_benchmark_lm_eval.ipynb) | `lm_eval.simple_evaluate` (vllm + hf), custom tasks, base+LoRA |
| 08 | [Diversity metrics](08_diversity_metrics.ipynb) | sentence-transformers, vendi-score, prdc, mauve, self-BLEU |

**Suggested order:** `00` first (data + tokenizers underpin everything), then any notebook
— they are independent. Lightest GPU footprint first: `00 → 03 → 08`, then `01`/`04`
(offline vLLM), `02` (serving), `05`/`06` (training), `07` (eval).
