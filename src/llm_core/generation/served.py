"""Generate against an already-running `vllm serve` endpoint over HTTP.

The in-process :func:`llm_core.generation.generator.generate` boots a vLLM engine per
call; this is its HTTP twin for when an engine is already served (see
:mod:`llm_core.serving`). It samples by LoRA-module name through the OpenAI-compatible
``/v1/completions`` API, so a sweep can probe many adapters off **one** engine init.

Scope: **fixed sampling only** (what the coherence probe needs). Token/sequence EDT use
vLLM-internal logits processors / a warmup pass that have no OpenAI-API equivalent, so
they stay in the in-process path (``generator.generate``). The returned record shape
matches ``generator.generate`` (``text``/``strategy``/``temperature`` added to each input
record) so downstream consumers are identical regardless of transport.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..models import ModelProfile


def generate_served(
    prompts: list[dict],
    *,
    base_url: str,
    served_model: str,
    profile: ModelProfile,
    tokenizer,
    gen_kwargs: dict,
    apply_chat_template: bool | None = None,
    concurrency: int = 8,
    api_key: str = "EMPTY",
) -> list[dict]:
    """Sample one fixed-temperature continuation per prompt from a served model.

    ``served_model`` is the OpenAI ``model`` field — for a LoRA served via
    ``vllm serve --lora-modules name=path`` it's the registered ``name``. Chat templating
    is applied **client-side** via ``profile.render_chat`` (the same path the in-process
    generator uses, so ``enable_thinking`` is handled identically); the templated text is
    POSTed to ``/v1/completions`` as a raw prompt. Each request gets a distinct seed
    (``seed0 + index``) so identical prompts (chat mode) still yield diverse samples.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)

    if apply_chat_template is None:
        apply_chat_template = profile.has_chat_template
    texts = [p["prompt"] for p in prompts]
    if apply_chat_template:
        texts = [profile.render_chat(tokenizer, t) for t in texts]

    base = dict(gen_kwargs)
    seed0 = base.pop("seed", 0) or 0
    temperature = base.get("temperature")
    sampling = {
        "temperature": temperature,
        "top_p": base.get("top_p"),
        "max_tokens": base.get("max_tokens"),
    }
    sampling = {k: v for k, v in sampling.items() if v is not None}

    def _one(idx_text: tuple[int, str]) -> str:
        i, text = idx_text
        resp = client.completions.create(
            model=served_model, prompt=text, seed=seed0 + i, **sampling
        )
        return resp.choices[0].text

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        completions = list(pool.map(_one, enumerate(texts)))

    records = []
    for p, text in zip(prompts, completions):
        rec = dict(p)
        rec["text"] = text
        rec["strategy"] = "fixed"
        rec["temperature"] = temperature
        records.append(rec)
    return records
