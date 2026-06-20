"""vLLM OpenAI-compatible server lifecycle helpers.

`ServedVLLM` launches one `vllm serve` process holding a base model plus any number of
LoRA adapters, waits for it to become healthy, and tears it down — so many evals /
generations can share a *single* engine init instead of paying the (expensive) vLLM
startup per call. `build_serve_cmd` is the pure argv builder, kept separate so it is
unit-testable without launching anything.
"""

from __future__ import annotations

from .server import ServedVLLM, build_serve_cmd

__all__ = ["ServedVLLM", "build_serve_cmd"]
