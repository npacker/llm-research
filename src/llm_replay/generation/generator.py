"""vLLM-backed synthetic-data generation for the three core EDT strategies.

Strategies (research plan Area 1):
  - ``fixed``     : constant temperature (baseline, cond. D).
  - ``token_edt`` : per-token EDT via `EDTLogitsProcessor` (cond. A).
  - ``seq_edt``   : one temperature per sequence, estimated from a warmup pass (cond. B).

This is the repo's first direct vLLM use. Single GPU → ``tensor_parallel_size=1``.
"""

from __future__ import annotations

import math

from .temperature import EDTLogitsProcessor, edt_temperature


def build_model_args(
    model: str,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    extra: dict | None = None,
) -> dict:
    """vLLM `LLM(**kwargs)` args — mirrors scripts/evaluate.py's single-GPU defaults."""
    args = {
        "model": model,
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_model_len is not None:
        args["max_model_len"] = max_model_len
    if extra:
        args.update(extra)
    return args


def make_llm(model_args: dict, strategy: str):
    """Construct the vLLM engine; register the EDT processor only for token-EDT."""
    from vllm import LLM

    if strategy == "token_edt":
        return LLM(logits_processors=[EDTLogitsProcessor], **model_args)
    return LLM(**model_args)


def _topk_entropy(logprobs_step: dict) -> float:
    """Approximate next-token entropy (nats) from vLLM's top-k logprobs for one step.

    Renormalises the returned top-k probabilities — an *under*-estimate of the true
    entropy (the tail is missing), but monotonic enough to drive a per-sequence temp.
    """
    lps = [lp.logprob for lp in logprobs_step.values()]
    probs = [math.exp(x) for x in lps]
    z = sum(probs) or 1.0
    probs = [p / z for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def _render_chat(llm, texts: list[str]) -> list[str]:
    """Wrap each user-turn in the model's chat template (no thinking) for chat-mode replay."""
    tok = llm.get_tokenizer()
    out = []
    for t in texts:
        msgs = [{"role": "user", "content": t}]
        try:
            r = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:  # tokenizer doesn't take enable_thinking
            r = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
        out.append(r)
    return out


def generate(
    llm,
    prompts: list[dict],
    *,
    strategy: str,
    gen_kwargs: dict,
    edt: dict | None = None,
    seq_edt: dict | None = None,
    apply_chat_template: bool = False,
) -> list[dict]:
    """Generate one continuation per prompt record; return enriched records.

    With `apply_chat_template`, each prompt is wrapped in the model's chat template (for
    `chat`-mode generative replay from an instruct model). Every request gets a distinct
    seed (base + index), so identical prompts (chat mode) still yield diverse samples.
    """
    from vllm import SamplingParams

    texts = [p["prompt"] for p in prompts]
    if apply_chat_template:
        texts = _render_chat(llm, texts)

    base = dict(gen_kwargs)
    seed0 = base.pop("seed", 0) or 0
    n = len(texts)

    if strategy == "fixed":
        sp = [SamplingParams(seed=seed0 + i, **base) for i in range(n)]
        per_req_temp = [base.get("temperature")] * n

    elif strategy == "token_edt":
        if not edt:
            raise ValueError(
                "strategy 'token_edt' requires an `edt` block (T0, N, theta)"
            )
        ea = {"edt_mode": "token", **{k: edt[k] for k in ("T0", "N", "theta")}}
        b = {k: v for k, v in base.items() if k != "temperature"}
        sp = [
            SamplingParams(seed=seed0 + i, temperature=1.0, extra_args=ea, **b)
            for i in range(n)
        ]
        per_req_temp = ["token_edt"] * n

    elif strategy == "seq_edt":
        if not edt:
            raise ValueError(
                "strategy 'seq_edt' requires an `edt` block (T0, N, theta)"
            )
        cfg = seq_edt or {}
        warmup_tokens = int(cfg.get("warmup_tokens", 32))
        k = int(cfg.get("logprobs_k", 20))
        # Pass 1: short warmup at T0 to estimate each sequence's mean entropy.
        warm_sp = [
            SamplingParams(
                seed=seed0 + i,
                temperature=edt["T0"],
                max_tokens=warmup_tokens,
                logprobs=k,
                top_p=base.get("top_p", 1.0),
            )
            for i in range(n)
        ]
        warm = llm.generate(texts, warm_sp)
        per_req_temp = []
        for out in warm:
            steps = out.outputs[0].logprobs or []
            hs = [_topk_entropy(s) for s in steps if s]
            h_avg = sum(hs) / len(hs) if hs else edt["T0"]
            per_req_temp.append(
                edt_temperature(h_avg, edt["T0"], edt["N"], edt["theta"])
            )
        # Pass 2: full generation, one temperature per request.
        b = {k2: v for k2, v in base.items() if k2 != "temperature"}
        sp = [
            SamplingParams(seed=seed0 + i, temperature=per_req_temp[i], **b)
            for i in range(n)
        ]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    outs = llm.generate(texts, sp)
    records = []
    for p, out, t in zip(prompts, outs, per_req_temp):
        rec = dict(p)
        rec["text"] = out.outputs[0].text
        rec["strategy"] = strategy
        rec["temperature"] = t
        if strategy in ("token_edt", "seq_edt"):
            rec["edt"] = edt
        records.append(rec)
    return records
