"""Manage a single `vllm serve` process so many calls share one engine init.

vLLM startup (weight load + CUDA-graph capture) is the dominant fixed cost when a sweep
evaluates / probes many checkpoints. Serving the base model once with all the LoRA
adapters registered (``--lora-modules name=path``) lets every consumer hit it by name
over HTTP — lm-eval's ``local-completions`` backend for the battery, the ``openai``
client for free-form coherence probes — instead of booting an in-process engine each time.

``build_serve_cmd`` is a pure argv builder (no launch) so it can be unit-tested. The
``ServedVLLM`` context manager owns the lifecycle: launch → poll ``/health`` → tear down
(killing the whole process group so no server is orphaned on error). Pass an existing
``base_url`` to skip launching and just target a server someone else started.

Single GPU → ``--tensor-parallel-size`` is never passed (see CLAUDE.md).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.request


def build_serve_cmd(
    model: str,
    lora_modules: dict[str, str],
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_lora_rank: int = 32,
    max_loras: int | None = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
) -> list[str]:
    """Build the ``vllm serve`` argv for a base model + named LoRA adapters.

    ``lora_modules`` maps a request-time name (the value passed as the OpenAI ``model``
    field) to the adapter directory. ``max_loras`` defaults to the number of adapters so
    they all stay resident (adapters are tiny; this avoids LRU swap mid-sweep). No
    ``--tensor-parallel-size`` is emitted — this box is single-GPU.
    """
    cmd = [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]
    if lora_modules:
        cmd += [
            "--enable-lora",
            "--max-lora-rank",
            str(max_lora_rank),
            "--max-loras",
            str(max_loras or len(lora_modules)),
            "--lora-modules",
            *[f"{name}={path}" for name, path in lora_modules.items()],
        ]
    return cmd


def _health_ok(url: str, timeout: float = 5.0) -> bool:
    """True iff ``GET url`` returns HTTP 200 (server ready)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (localhost)
            return resp.status == 200
    except Exception:
        return False


class ServedVLLM:
    """Context manager around a `vllm serve` process (or an externally-served URL).

    ::

        with ServedVLLM("Qwen/Qwen3.5-4B", {"pt1": "runs/.../adapter"}) as srv:
            # srv.completions_url -> lm-eval --base-url
            # srv.openai_base_url -> openai.OpenAI(base_url=...)

    If ``base_url`` (a ``http://host:port`` root) is given, no process is launched and the
    manager just targets that server. Otherwise a process is spawned in its own session so
    ``__exit__`` can kill the whole group, and ``/health`` is polled until ready.
    """

    def __init__(
        self,
        model: str,
        lora_modules: dict[str, str] | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        max_lora_rank: int = 32,
        max_loras: int | None = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        health_timeout_s: float = 600.0,
        poll_interval_s: float = 3.0,
        base_url: str | None = None,
        dry_run: bool = False,
    ):
        self.model = model
        self.lora_modules = dict(lora_modules or {})
        self.host = host
        self.port = port
        self.max_lora_rank = max_lora_rank
        self.max_loras = max_loras
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.health_timeout_s = health_timeout_s
        self.poll_interval_s = poll_interval_s
        self.dry_run = dry_run
        self._external = base_url is not None
        self._root = base_url.rstrip("/") if base_url else f"http://{host}:{port}"
        self._proc: subprocess.Popen | None = None

    # URLs the two consumers need.
    @property
    def root_url(self) -> str:
        return self._root

    @property
    def openai_base_url(self) -> str:
        """Base URL for the `openai` client (it appends /completions itself)."""
        return f"{self._root}/v1"

    @property
    def completions_url(self) -> str:
        """Full /v1/completions URL for lm-eval's local-completions --base-url."""
        return f"{self._root}/v1/completions"

    @property
    def health_url(self) -> str:
        return f"{self._root}/health"

    def cmd(self) -> list[str]:
        return build_serve_cmd(
            self.model,
            self.lora_modules,
            host=self.host,
            port=self.port,
            max_lora_rank=self.max_lora_rank,
            max_loras=self.max_loras,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
        )

    def __enter__(self) -> ServedVLLM:
        if self._external:
            print(f"[serve] reusing external server at {self._root}")
            return self
        cmd = self.cmd()
        print("[serve] $", " ".join(cmd))
        if self.dry_run:
            return self
        # Own session → __exit__ can signal the whole group (vLLM spawns workers).
        self._proc = subprocess.Popen(cmd, start_new_session=True)
        self._wait_healthy()
        return self

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.health_timeout_s
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited early (code {self._proc.returncode}) before /health"
                )
            if _health_ok(self.health_url):
                print(f"[serve] healthy at {self._root}")
                return
            time.sleep(self.poll_interval_s)
        self._terminate()
        raise TimeoutError(
            f"vllm serve not healthy after {self.health_timeout_s:.0f}s ({self.health_url})"
        )

    def _terminate(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=10)
        except ProcessLookupError:
            pass
        finally:
            self._proc = None

    def __exit__(self, *exc) -> None:
        if not self._external:
            print("[serve] tearing down vllm serve")
            self._terminate()


__all__ = ["ServedVLLM", "build_serve_cmd"]
