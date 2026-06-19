#!/usr/bin/env bash
# Runs once after the container is created.
set -euo pipefail

echo "==> Fixing ownership of cache mounts (named volumes mount as root)..."
sudo mkdir -p \
  "$HOME/.cache/huggingface/hub" \
  "$HOME/.nv/ComputeCache" \
  "$HOME/.cache/torch/inductor" \
  "$HOME/.cache/vllm" \
  "$HOME/.cache/uv" \
  "$HOME/.cache/pip" \
  "$HOME/.claude"
sudo chown -R "$(id -u):$(id -g)" "$HOME/.cache" "$HOME/.nv" "$HOME/.claude"

# Seed the lockfile resolved during the image build into the workspace so the
# committed environment is reproducible. Only if you haven't committed one yet.
if [ ! -f uv.lock ] && [ -f /opt/uv.lock.generated ]; then
  echo "==> No uv.lock in workspace yet — seeding from the image build."
  echo "    (commit this uv.lock for reproducible rebuilds)"
  cp /opt/uv.lock.generated uv.lock
fi

# Reconcile /opt/venv with the workspace lockfile. Fast no-op if they already
# match (the common case); re-installs if you've edited pyproject.toml/uv.lock.
echo "==> Syncing dependencies (uv sync --locked)..."
if [ -f uv.lock ]; then
  uv sync --locked || {
    echo "    Lock is stale vs pyproject.toml — relocking..."
    uv lock && uv sync
  }
else
  uv sync
fi

echo "==> Versions:"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import vllm; print('vllm', vllm.__version__)" || echo "(vllm import failed — check GPU/driver)"
hf version 2>/dev/null || huggingface-cli version || true

echo "==> HuggingFace auth (needed for gated models/datasets, e.g. Idavidrein/gpqa):"
# devcontainer.json passes the host HF_TOKEN through via ${localEnv:HF_TOKEN},
# but that resolves only at container *creation* — so a token added to the host
# AFTER the container was built needs a rebuild/reopen to take effect. A token
# stored via `hf auth login` (in the persistent hf-cache volume) is independent
# of that and survives rebuilds.
if [ -n "${HF_TOKEN:-}" ]; then
  if hf auth whoami >/dev/null 2>&1; then
    echo "    OK — HF_TOKEN from host is valid (user: $(hf auth whoami 2>/dev/null | head -1))."
  else
    echo "    HF_TOKEN is set but validation failed — token may be invalid or expired."
  fi
elif hf auth whoami >/dev/null 2>&1; then
  echo "    OK — authenticated via cached 'hf auth login' (user: $(hf auth whoami 2>/dev/null | head -1))."
else
  echo "    No HF token detected. Gated downloads will 401. To fix, either:"
  echo "      - set HF_TOKEN on the HOST, then rebuild/reopen the container, or"
  echo "      - run 'hf auth login' in a terminal (persists in the hf-cache volume)."
fi

echo "==> GPU check:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "nvidia-smi present but no GPU visible — check --gpus=all and the NVIDIA Container Toolkit."
else
  echo "nvidia-smi not found inside container (GPU passthrough not active)."
fi

# Register the venv as a Jupyter kernel for notebook work.
python -m ipykernel install --user --name llm-research --display-name "Python (llm-research)" >/dev/null 2>&1 || true

echo "==> Done. Start the vLLM OpenAI server with e.g.:"
echo "    vllm serve <model-id> --host 0.0.0.0 --port 8000"
