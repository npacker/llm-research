# LLM Research devcontainer

GPU dev container for vLLM / HuggingFace research. CUDA 13.0 base, Python 3.12,
dependencies pinned and managed with [uv](https://docs.astral.sh/uv/).

## Pinned stack

Versions below are PyPI-verified and confirmed by a real build + GPU smoke test
on the target hardware (2026-06-18).

| Package | Version | Notes |
|---|---|---|
| vllm | `0.23.0` | Latest stable; its cu13 default matches Blackwell sm_120 + torch cu130. |
| torch / torchvision | `2.11.0+cu130` / `0.26.0` | Hard pin from vLLM; CUDA-13 build pulled from the PyTorch `cu130` index. |
| transformers | `5.12.1` | |
| tokenizers | `0.22.2` | Not pinned directly — transformers caps it `<=0.23.0` and 0.23.0 was never published, so uv resolves the latest in-range 0.22.x. |
| accelerate / datasets / huggingface-hub | `1.14.0` / `5.0.0` / `1.19.0` | |
| peft / bitsandbytes / safetensors / sentencepiece / einops / hf-transfer | pinned | see [`pyproject.toml`](../pyproject.toml) |

Convenience tools (jupyterlab, numpy, pandas, wandb, openai, ruff, …) are
unpinned in `pyproject.toml` but frozen exactly by `uv.lock`.

## First-build steps (do these once)

1. **Pin the base image by digest.** ✅ Done (2026-06-17). `BASE_IMAGE` in
   [`devcontainer.json`](devcontainer.json) and the Dockerfile are pinned to
   `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:e071e85c52ad91fc9ea24158ff5330876b2d1a5c4ac83ccc6066976835873c01`.
   To re-resolve in future: `docker buildx imagetools inspect <tag>`.

2. **`uv.lock` is generated.** ✅ The image has been built and the resolved
   lockfile extracted to [`../uv.lock`](../uv.lock) in the workspace root (461 KB,
   torch confirmed at `2.11.0+cu130` from the PyTorch index). Commit it once this
   becomes a git repo:
   ```sh
   git add uv.lock pyproject.toml && git commit -m "Add uv lockfile"
   ```
   Rebuilds then use `uv sync --locked` for bit-stable installs. `post-create.sh`
   re-seeds the lock automatically if it's ever missing.

## Day-to-day

- Add/upgrade a dep: edit `pyproject.toml`, then `uv add <pkg>` / `uv lock && uv sync`.
  Keep `torch`/`vllm` on CUDA major 13 — don't `--upgrade` them blindly.
- Serve a model (OpenAI-compatible API on forwarded port 8000):
  ```sh
  vllm serve <model-id> --host 0.0.0.0 --port 8000
  ```
- Gated models: export `HF_TOKEN` on the host before opening the container; it's
  passed through automatically.

## Tuning for this machine (single RTX PRO 6000 Blackwell, 96 GB / 128 GB RAM / 8c-16t)

- **CUDA base is required, not just convenient.** Blackwell is compute
  capability **12.0 (sm_120)** and needs **CUDA 12.8+ and driver r570+** — the
  CUDA 13.0 base + torch 2.11 stack here is exactly right. Confirm the host
  driver is 570 or newer (`nvidia-smi`).
- **Single GPU → leave tensor-parallelism at 1** (no `--tensor-parallel-size`).
  `--gpus=all` exposes the one card; that's all vLLM needs.
- **Lean on Blackwell's native FP8/FP4.** This card has hardware FP8 and FP4
  (nvfp4) — prefer those quantizations over older INT4/AWQ to use the silicon
  and stretch the 96 GB. FP8 KV cache (`--kv-cache-dtype fp8`) buys longer
  context too.
- **What fits in 96 GB:** ~70B in FP8/nvfp4, ~32B in BF16, or smaller models with
  very long context / big batch. Examples:
  ```sh
  # ~32B class, BF16, leave VRAM for KV cache:
  vllm serve Qwen/Qwen3-32B --gpu-memory-utilization 0.90 --max-model-len 32768
  # ~70B class in FP8 to fit one 96 GB card, with FP8 KV cache:
  vllm serve <70b-fp8> --quantization fp8 --kv-cache-dtype fp8 --gpu-memory-utilization 0.92
  ```
  If you hit attention-kernel errors on this new arch, try
  `VLLM_ATTENTION_BACKEND=FLASHINFER` (good Blackwell support).
- **`/dev/shm` = 32 GB** (set in `devcontainer.json`) — plenty for single-GPU
  inference; it's there for multiprocessing dataloaders, not TP.
- **8c/16t CPU:** cap data-prep parallelism (`datasets` `num_proc<=16`,
  dataloader `num_workers` ~8). For source kernel builds (flash-attn etc.),
  `MAX_JOBS=16` plus the `TORCH_CUDA_ARCH_LIST=12.0+PTX` already set in the
  container keeps compiles fast and arch-correct.

## GPU requirement

Needs the NVIDIA Container Toolkit on the host — on Windows that means Docker
Desktop with the WSL2 backend and a recent NVIDIA driver. The image still builds
without a GPU; vLLM just can't start an engine.
