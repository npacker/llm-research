# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A GPU research workspace for LLM inference/experimentation built on the **vLLM + HuggingFace** stack. There is no application source code or test suite yet — the repo is currently the *environment*: a pinned dependency set (`pyproject.toml` + `uv.lock`) and a CUDA devcontainer. Work here means running models, notebooks, and experiments, not building a package (`[tool.uv] package = false`).

## Environment & dependency management (uv)

The project env is **not** in the workspace — it lives at `/opt/venv` (baked into the image, set via `UV_PROJECT_ENVIRONMENT`). `python`, `vllm`, `jupyter`, etc. are already on `PATH`; no venv activation needed.

- Add/upgrade a dependency: edit `pyproject.toml`, then `uv add <pkg>` or `uv lock && uv sync`.
- Reproducible install from the committed lock: `uv sync --locked` (this is what `post-create.sh` runs).
- `uv.lock` **is committed** (not gitignored) and is the source of truth for exact versions — always commit it alongside `pyproject.toml` changes.

### Pinning philosophy — do not break this

ABI-critical packages are hard `==` pinned and must stay coherent: **torch 2.11.0+cu130, torchvision 0.26.0, vllm 0.23.0** all target CUDA major 13. `torch`/`torchvision` come from the explicit PyTorch `cu130` index (`[tool.uv.sources]`), not PyPI. **Never `--upgrade` torch or vllm blindly** — they must move together and stay on CUDA 13. Convenience tools (jupyterlab, numpy, pandas, ruff, wandb, openai, …) are intentionally unpinned in `pyproject.toml` but frozen exactly by `uv.lock`. `tokenizers` is deliberately not pinned (transformers caps it and the obvious version was never published — let uv resolve it).

## Common commands

```sh
# Serve a model — OpenAI-compatible API on forwarded port 8000:
vllm serve <model-id> --host 0.0.0.0 --port 8000

# Lint / format (ruff is installed):
ruff check .
ruff format .

# Jupyter kernel "Python (llm-research)" is pre-registered by post-create.sh.
```

Gated HuggingFace models: set `HF_TOKEN` in the **host** environment before opening the container — it's passed through automatically.

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
