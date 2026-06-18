# syntax=docker/dockerfile:1

# ============================================================================
# Base image — CUDA 13.0.2 + cuDNN, "devel" flavor (ships nvcc + headers for
# building custom CUDA/C++ extensions and torch.compile). CUDA major 13 matches
# torch 2.11.0's bundled CUDA, which is what vLLM 0.22.1 pins.
#
# !! BEFORE FIRST BUILD, pin this by digest for reproducibility:
#      docker buildx imagetools inspect nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
#    then set BASE_IMAGE to  ...ubuntu24.04@sha256:<digest>
#    If that exact tag isn't published, pick the nearest CUDA-13 cudnn-devel tag
#    from https://hub.docker.com/r/nvidia/cuda/tags and update PYTHON note below.
#    Ubuntu 24.04 is chosen because it ships Python 3.12 natively.
# ============================================================================
ARG BASE_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:e071e85c52ad91fc9ea24158ff5330876b2d1a5c4ac83ccc6066976835873c01
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Make all GPUs and capabilities visible inside the container.
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---- System packages + a non-root user --------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        git-lfs \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        sudo \
        vim \
        less \
        htop \
        tmux \
        openssh-client \
        # Python 3.12 is the default python3 on Ubuntu 24.04.
        python3 \
        python3-venv \
        python3-dev \
    && git lfs install --system \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create the 'vscode' user the devcontainer connects as, with passwordless sudo.
# Ubuntu 24.04 ships a default 'ubuntu' user at UID/GID 1000, so free that slot
# first, then claim 1000:1000 for vscode (the conventional devcontainer UID).
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN set -eux; \
    if getent passwd ${USER_UID} >/dev/null; then \
        userdel -r "$(getent passwd ${USER_UID} | cut -d: -f1)" 2>/dev/null || true; \
    fi; \
    if getent group ${USER_GID} >/dev/null; then \
        groupdel "$(getent group ${USER_GID} | cut -d: -f1)" 2>/dev/null || true; \
    fi; \
    groupadd --gid ${USER_GID} ${USERNAME}; \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}; \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# ---- uv (Astral) — static binary, no installer -------------------------------
# Pin uv itself; bump this tag when a newer uv ships.
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /usr/local/bin/

# uv project env lives OUTSIDE the workspace so it survives the bind mount and
# is baked into the image. PATH below makes `python`/`vllm`/`jupyter` resolve.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=3.12 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
ENV PATH="/opt/venv/bin:${PATH}"

# ---- Resolve + install the pinned dependency stack ---------------------------
# Bind-mount only pyproject.toml; resolve to a lockfile and install into
# /opt/venv. The generated lock is stashed at /opt/uv.lock.generated so
# post-create.sh can seed it into the (bind-mounted) workspace for you to commit.
# The cache mount keeps wheels across rebuilds; UV_LINK_MODE=copy is required
# because the cache and the venv are on different filesystems.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=/build/pyproject.toml \
    cd /build \
    && uv sync \
    && cp uv.lock /opt/uv.lock.generated

# Hand the baked env + stashed lock to the non-root user so runtime `uv sync`
# (in post-create) can update it without sudo.
RUN chown -R ${USER_UID}:${USER_GID} /opt/venv /opt/uv.lock.generated

USER ${USERNAME}
WORKDIR /workspaces/llm-research

# overrideCommand defaults to true for devcontainers; this keeps the image
# usable as a plain `docker run` dev box too.
CMD ["sleep", "infinity"]
