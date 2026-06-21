# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A GPU research workspace for LLM inference/experimentation on the **vLLM + HuggingFace** stack, oriented around a research program on dynamic-temperature generative replay & model collapse (see [`research/`](research/)).

It remains environment-first (a pinned dependency set + CUDA devcontainer), but it now carries a working **evaluation toolkit** and is installed as two editable packages: **`llm_core`** (general, reusable LLM infra — model profiles, eval glue, corpus loading, diversity metrics, EDT generation, LoRA training) and **`llm_replay`** (the research layer on top — prefix-prompt construction + synthetic-corpus quality validation for the generative-replay study):

- `scripts/evaluate.py` — capability/regression battery via **lm-evaluation-harness** (the "is the model still capable?" axis).
- `scripts/diversity.py` + `src/llm_core/metrics/diversity.py` — corpus-vs-corpus **distribution/diversity** metrics (the model-collapse axis: MAUVE, Vendi, prdc, Self-BLEU, …).
- `scripts/generate.py` + `src/llm_core/generation/` (EDT temperature + vLLM generator) + `src/llm_replay/generation/prompts.py` (prefix-only prompts) — **EDT synthetic-data generation** (fixed / sequence-level / token-level EDT; vLLM).
- `scripts/validate.py` + `src/llm_replay/generation/validation.py` — **quality validation** of a generated corpus (per-sample gates + perplexity + diversity panel).
- `scripts/train.py` + `src/llm_core/training/` — **LoRA fine-tuning** over domain/general/synthetic corpus mixes (forgetting/replay study); `scripts/forgetting_report.py` tabulates base-vs-condition deltas.
- `eval_tasks/` — custom lm-eval tasks not shipped upstream (**SuperGPQA**, **IFBench**), loaded via `--include-path`.
- `configs/eval/`, `configs/gen/`, `configs/validate/`, `configs/train/` — declarative eval / generation / validation / training configs.

The repo **is** installed as editable packages via a hatchling build backend
(`[tool.hatch.build.targets.wheel] packages = ["src/llm_core", "src/llm_replay"]`); `uv sync` installs
both, so scripts and tests `import llm_core…` / `import llm_replay…` directly (no `sys.path` shims).
Intra-package imports are relative (`from ..corpus import …`); `llm_replay` reaches general code via
absolute `from llm_core… import …`; scripts use absolute imports. A CPU-only, no-network `pytest`
suite lives in [`tests/`](tests/) (run `pytest tests/`); the heavier embedding/MAUVE/GPU paths are
still validated by manual GPU runs.

**Not built yet:** the **recursive** multi-generation collapse loop (chaining
generate→validate→train across generations); single-generation training now exists.

## Environment & dependency management (uv)

The project env is **not** in the workspace — it lives at `/opt/venv` (baked into the image, set via `UV_PROJECT_ENVIRONMENT`). `python`, `vllm`, `jupyter`, etc. are already on `PATH`; no venv activation needed.

- Add/upgrade a dependency: edit `pyproject.toml`, then `uv add <pkg>` or `uv lock && uv sync`.
- Reproducible install from the committed lock: `uv sync --locked` (this is what `post-create.sh` runs).
- `uv.lock` **is committed** (not gitignored) and is the source of truth for exact versions — always commit it alongside `pyproject.toml` changes.

### Pinning philosophy — do not break this

ABI-critical packages are hard `==` pinned and must stay coherent: **torch 2.11.0+cu130, torchvision 0.26.0, vllm 0.23.0** all target CUDA major 13. `torch`/`torchvision` come from the explicit PyTorch `cu130` index (`[tool.uv.sources]`), not PyPI. **Never `--upgrade` torch or vllm blindly** — they must move together and stay on CUDA 13. Convenience tools (jupyterlab, numpy, pandas, ruff, wandb, openai, …) and the eval/metrics stack (lm-eval, sentence-transformers, mauve-text, vendi-score, prdc, emoji, syllapy, …) are intentionally unpinned in `pyproject.toml` but frozen exactly by `uv.lock`. When adding eval/metrics deps, re-lock and confirm the torch/torchvision/vllm/transformers pins did not move (`grep -A1 -E '^name = "(torch|vllm|transformers)"' uv.lock`). `tokenizers` is deliberately not pinned (transformers caps it and the obvious version was never published — let uv resolve it).

### Research existing packages before reimplementing (and before swapping)

When a task could be solved by a library — or when proposing to replace custom code *with* a library — do the research **before** writing code or committing to a plan, and verify it rather than asserting from memory (see [[verify-dependency-claims]]):

- **Check what's already a dependency first.** The eval/metrics stack is broad (lm-eval, sentence-transformers, mauve-text, vendi-score, prdc, sacrebleu, langdetect, datasets, peft, …). Prefer reusing an installed package over adding a new one; prefer stdlib over a new dependency for small, stable utilities.
- **Verify the package is real and compatible before depending on it.** Resolve it in a throwaway manifest (`uv add --dry-run` / a scratch `uv.lock`) and confirm it doesn't move the ABI-critical pins (`grep -A1 -E '^name = "(torch|vllm|transformers)"' uv.lock`). Do **not** cite a PyPI package, version, or API from training data as fact — packages get abandoned, renamed, or never existed.
- **Read the code you're proposing to replace.** A swap is only a win if it preserves behavior. Some custom code is small *and* load-bearing — it encodes project-specific semantics a general library won't: the `hf:<dataset>:<split>:<field>` spec in `corpus.py`, the LoRA base+adapter path in `validation.perplexity()`, EDT temperature, prefix-only prompts, the masked-CE perplexity. "Fewer lines" that silently drops a split argument, a LoRA path, or a gate's meaning is a regression, not a cleanup.
- **Count the real cost.** A new dependency is not free: lock churn, ABI-coherence risk, import-time cost (scripts defer heavy imports to keep `--help` fast), and supply-chain surface. A correct 20-line stdlib helper usually beats a heavy dependency that does the same thing.

The bar: reuse > stdlib > a *verified* new dependency > custom code. Reach for custom code when it's research-specific or when the library swap would change behavior.

## Common commands

```sh
# Serve a model — OpenAI-compatible API on forwarded port 8000:
vllm serve <model-id> --host 0.0.0.0 --port 8000

# Download weights ahead of time (optional — `vllm serve` pulls on first run).
# Use `hf`, NOT the deprecated `huggingface-cli`. Lands in the fast hf-cache volume.
hf download <repo-id>

# Check a running server (no `vllm` status subcommand — liveness is /health):
curl -s localhost:8000/health      # -> 200 when ready
curl -s localhost:8000/v1/models   # which model is loaded

# Lint / format (ruff is installed):
ruff check .
ruff format .

# Capability eval battery (lm-eval). Backends: hf | vllm | local-completions.
python scripts/evaluate.py --config configs/eval/canary.yaml --model <id> --backend vllm
# Custom tasks (SuperGPQA / IFBench) need --include-path:
python scripts/evaluate.py --config configs/eval/canary.yaml --model <id> --backend vllm \
    --tasks supergpqa --include-path eval_tasks/supergpqa

# Distribution/diversity metrics (collapse axis): synthetic corpus vs real/Gen-0 reference.
python scripts/diversity.py --synthetic gen3.txt --real gen0.txt --generation 3

# LoRA fine-tune (corpus mix from config); then eval base+adapter via vLLM (NOT a merged model):
python scripts/train.py --config configs/train/domain_general.yaml --model Qwen/Qwen3.5-4B
python scripts/evaluate.py --model Qwen/Qwen3.5-4B --backend vllm --lora runs/train_*/adapter \
    --lora-rank 16 --config configs/eval/canary.yaml --generation domain_general

# EDT synthetic-data generation (strategy from config: fixed | seq_edt | token_edt) + validate.
python scripts/generate.py --config configs/gen/token_edt.yaml --model <id> --generation 1
python scripts/validate.py --corpus runs/gen1_token_edt_<ts>/samples.jsonl --real <seed> --generation 1

# Jupyter kernel "Python (llm-research)" is pre-registered by post-create.sh.
```

Eval/diversity results write to `runs/` (gitignored). See [`scripts/README.md`](scripts/README.md)
and [`eval_tasks/README.md`](eval_tasks/README.md) for details.

### Evaluation & training procedures (verified the hard way — don't skip)

- **Reasoning models: thinking is auto-disabled for evals by `src/llm_core/models.py`.** Left on, the
  model emits a thinking preamble that overruns answer-extraction tasks' token budget — e.g. GSM8K scored
  **0.00 with thinking on vs 0.60 off** on the same checkpoint. The model profile detects the thinking
  toggle and supplies `enable_thinking: false` (and the chat-template default) automatically, so eval
  configs no longer hardcode it — set `model_args: {enable_thinking: true}` in a config only to force a
  thinking run. (`scripts/evaluate.py` merges the profile's args under the config's, threaded to the
  `hf`/`vllm` backends.)
- **Evaluate a fine-tuned model as base + LoRA adapter via vLLM**, not a merged checkpoint:
  `python scripts/evaluate.py --model <base> --backend vllm --lora runs/train_*/adapter --lora-rank <r> --config configs/eval/canary.yaml`.
  Qwen3.5 is a **VLM**; `train.py`'s default merge is text-only (`Qwen3_5ForCausalLM`), a sub-arch vLLM
  doesn't register, so vLLM can't load it. `train.py --merge` warns generically when the base is a VLM
  (`profile.is_vlm`) — eval the merged checkpoint with `--backend hf`.
- **HF backend `batch_size: auto` can thrash on WSL** (the GPU memory budget is over-reported), turning a
  ~2-min eval into ~17 min of batch probing. Use a fixed `batch_size` with `--backend hf`, or prefer vLLM at scale.
- The HF backend **does** run on the GPU (`HFLM` defaults to `cuda`); it's a valid backend, just slower than
  vLLM at scale (vLLM's fixed init cost amortizes over large runs).

Gated HuggingFace models: set `HF_TOKEN` in the **host** environment before opening the container — it's passed through automatically. See `.devcontainer/README.md` ("Day-to-day") for download/cache-location and status-check details.

## Hardware constraints (single RTX PRO 6000 Blackwell, 96 GB)

These shape every serving decision — see `.devcontainer/README.md` for the full rationale:

- **Single GPU → leave tensor-parallelism at 1** (do not pass `--tensor-parallel-size`).
- **Blackwell is sm_120** (compute capability 12.0); requires CUDA 12.8+ / driver r570+. `TORCH_CUDA_ARCH_LIST=12.0+PTX` is already set for any from-source CUDA builds.
- **Prefer native FP8/FP4 (nvfp4) quantization** over older INT4/AWQ. `--kv-cache-dtype fp8` buys longer context.
- Rough capacity in 96 GB: ~70B in FP8/nvfp4, ~32B in BF16.
- If attention kernels fail on this arch, try `VLLM_ATTENTION_BACKEND=FLASHINFER`.
- Cap CPU parallelism (8c/16t): `datasets` `num_proc<=16`, dataloader `num_workers~8`.

## Devcontainer notes

- Caches (HF hub weights, CUDA/torch/vllm compile caches, uv/pip) live in **named volumes**, so model downloads and compiled kernels survive container rebuilds. The env-var routing for these is in `devcontainer.json` (`HF_HOME`, `VLLM_CACHE_ROOT`, etc.).
- The base image is **pinned by digest** in both `devcontainer.json` and `Dockerfile`. Re-resolve with `docker buildx imagetools inspect <tag>`.
- Model weights (`*.safetensors`, `*.bin`, `*.gguf`, `*.pt`, …) are gitignored — push to the HF Hub or track with Git LFS (`git lfs track`), and force-add intentional small artifacts with `git add -f`.
