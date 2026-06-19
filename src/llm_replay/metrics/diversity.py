"""Distribution / diversity metrics for the model-collapse axis.

These compare *corpora* (e.g. generation-N synthetic vs. generation-0 / real text)
rather than scoring a model on tasks — the primary collapse signal in the research
plan's Area 5. Split into:

- **reference-free** (one corpus): diversity that should *decline* as a model
  collapses — Distinct-n, Self-BLEU, Vendi score, vocabulary size, tail mass.
- **reference-based** (synthetic vs. a real/Gen-0 reference): distance that should
  *grow* — MAUVE, precision/recall + density/coverage (prdc), Fréchet embedding
  distance, RBF-MMD, unigram-KL.

Heavy imports (sentence-transformers, mauve, prdc, torch) are deferred into the
functions so importing this module stays cheap.
"""

from __future__ import annotations

import functools
import math
import random
from collections import Counter

import numpy as np

# --------------------------------------------------------------------------- #
# tokenisation helpers
# --------------------------------------------------------------------------- #


def _tokens(text: str) -> list[str]:
    return text.split()


def _unigram_counts(texts: list[str]) -> tuple[Counter, int]:
    c: Counter = Counter()
    for t in texts:
        c.update(_tokens(t))
    return c, sum(c.values())


# --------------------------------------------------------------------------- #
# reference-free (one corpus) — diversity declines under collapse
# --------------------------------------------------------------------------- #


def distinct_n(texts: list[str], n: int = 2) -> float:
    """Unique n-grams / total n-grams (higher = more diverse)."""
    total, uniq = 0, set()
    for t in texts:
        toks = _tokens(t)
        grams = list(zip(*[toks[i:] for i in range(n)]))
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / total if total else 0.0


def vocabulary_size(texts: list[str]) -> int:
    return len(_unigram_counts(texts)[0])


def tail_mass(texts: list[str], frac: float = 0.2) -> float:
    """Fraction of total token mass held by the rarest `frac` of token *types*.

    Collapse = tail disappearance, so this shrinks across generations.
    """
    counts, total = _unigram_counts(texts)
    freqs = sorted(counts.values())  # ascending → rarest first
    k = max(1, int(len(freqs) * frac))
    return sum(freqs[:k]) / total if total else 0.0


def self_bleu(
    texts: list[str], n_sample: int = 200, n_ref: int = 100, seed: int = 0
) -> float:
    """Mean sentence-BLEU of sampled texts against other texts (0-1; lower = more diverse).

    Each hypothesis is scored as multi-reference BLEU against a sample of the *other*
    texts (Zhu et al. Self-BLEU). Sampled on both axes to bound the O(n^2) cost.
    """
    from sacrebleu import sentence_bleu

    rng = random.Random(seed)
    n = len(texts)
    if n < 2:
        return 0.0
    idx = range(n)
    hyp_idx = idx if n <= n_sample else rng.sample(idx, n_sample)
    scores = []
    for i in hyp_idx:
        # Sample refs from the other texts; over-sample by one so that dropping the
        # hypothesis itself (if drawn) still leaves up to n_ref references.
        refs = [texts[j] for j in rng.sample(idx, min(n_ref + 1, n)) if j != i][:n_ref]
        if refs:
            scores.append(sentence_bleu(texts[i], refs).score)  # sacrebleu is 0-100
    return float(np.mean(scores)) / 100.0 if scores else 0.0


def vendi_score(embeddings: np.ndarray) -> float:
    """Vendi score (effective number of distinct samples) from unit-norm embeddings."""
    from vendi_score import vendi

    return float(vendi.score_dual(np.asarray(embeddings, dtype=np.float64)))


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #


# Bounded cache: keep at most a couple of (model, device) embedders resident so a
# long-lived process (notebook / sweep) that switches models doesn't accumulate
# them in GPU memory the way an unbounded cache would.
@functools.lru_cache(maxsize=2)
def _load_embedder(model_name: str, device: str | None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def embed(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Unit-normalised sentence embeddings (model cached across calls)."""
    return _load_embedder(model_name, device).encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


# --------------------------------------------------------------------------- #
# reference-based (synthetic vs. real/Gen-0) — distance grows under collapse
# --------------------------------------------------------------------------- #


def unigram_kl(real: list[str], synth: list[str], eps: float = 1e-9) -> float:
    """KL(P_real || P_synth) over the union vocabulary (nats)."""
    cr, tr = _unigram_counts(real)
    cs, ts = _unigram_counts(synth)
    kl = 0.0
    for w in set(cr) | set(cs):
        p = cr.get(w, 0) / tr if tr else 0.0
        q = cs.get(w, 0) / ts if ts else 0.0
        if p > 0:
            kl += p * math.log((p + eps) / (q + eps))
    return float(kl)


def frechet_distance(
    real_emb: np.ndarray, synth_emb: np.ndarray, eps: float = 1e-6
) -> float:
    """Fréchet distance between embedding distributions (FID-style; lower = closer).

    Needs at least 2 samples per corpus for a covariance; returns NaN otherwise.
    With fewer samples than the embedding dimension the covariance is rank-deficient
    and ``sqrtm`` is unstable, so ``eps`` is added to the diagonals (standard FID
    stabilisation) to keep the result finite — values are still biased high in that
    regime, so compare only across runs with comparable corpus sizes.
    """
    from scipy.linalg import sqrtm

    if real_emb.shape[0] < 2 or synth_emb.shape[0] < 2:
        return float("nan")
    mu1, mu2 = real_emb.mean(0), synth_emb.mean(0)
    c1 = np.cov(real_emb, rowvar=False)
    c2 = np.cov(synth_emb, rowvar=False)
    offset = eps * np.eye(c1.shape[0])
    covmean = sqrtm((c1 + offset) @ (c2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1 + c2 - 2 * covmean))


def mmd_rbf(
    real_emb: np.ndarray,
    synth_emb: np.ndarray,
    gamma: float | None = None,
    block: int = 2048,
) -> float:
    """Squared RBF-kernel MMD between embedding sets (lower = closer).

    Kernel means are accumulated in row-blocks so peak memory is O(block * n)
    rather than the O(n^2) of a full pairwise matrix — safe for large corpora.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    if gamma is None:
        gamma = 1.0 / real_emb.shape[1]

    def _mean_kernel(a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        for i in range(0, a.shape[0], block):
            total += rbf_kernel(a[i : i + block], b, gamma=gamma).sum()
        return total / (a.shape[0] * b.shape[0])

    return float(
        _mean_kernel(real_emb, real_emb)
        + _mean_kernel(synth_emb, synth_emb)
        - 2 * _mean_kernel(real_emb, synth_emb)
    )


def prdc_metrics(
    real_emb: np.ndarray, synth_emb: np.ndarray, nearest_k: int = 5
) -> dict:
    """Precision / recall / density / coverage (Naeem et al.).

    recall & coverage are the collapse signals: they fall (synthetic stops covering
    the real distribution) while precision may stay high.
    """
    from prdc import compute_prdc

    return {
        k: float(v) for k, v in compute_prdc(real_emb, synth_emb, nearest_k).items()
    }


def mauve_score(
    real: list[str], synth: list[str], device_id: int = 0, max_text_length: int = 256
) -> float:
    """MAUVE: distribution similarity of synthetic vs. real text (higher = closer)."""
    import mauve

    out = mauve.compute_mauve(
        p_text=real,
        q_text=synth,
        device_id=device_id,
        max_text_length=max_text_length,
        verbose=False,
    )
    return float(out.mauve)


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #


def _mauve_device_id(device: str | None) -> int:
    """Translate a sentence-transformers device string to a mauve device_id.

    mauve uses an int: -1 for CPU, N for cuda:N. `None` (auto) maps to GPU 0, which
    mauve itself falls back to CPU on when no CUDA is present.
    """
    if device is None:
        return 0
    d = device.lower()
    if d.startswith("cpu"):
        return -1
    return int(d.split(":")[1]) if ":" in d else 0


def compute_panel(
    synth: list[str],
    real: list[str] | None = None,
    *,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
    with_mauve: bool = True,
) -> dict:
    """Full panel. Reference-based metrics are included only when `real` is given.

    Returns a flat dict of metric -> value (with a nested 'prdc' block).
    """
    panel: dict = {"n_synth": len(synth)}

    # reference-free (synthetic corpus)
    panel["distinct_1"] = distinct_n(synth, 1)
    panel["distinct_2"] = distinct_n(synth, 2)
    panel["self_bleu"] = self_bleu(synth)
    panel["vocab_size"] = vocabulary_size(synth)
    panel["tail_mass"] = tail_mass(synth)

    synth_emb = embed(synth, embed_model, device)
    panel["vendi"] = vendi_score(synth_emb)

    if real:
        panel["n_real"] = len(real)
        real_emb = embed(real, embed_model, device)
        panel["unigram_kl"] = unigram_kl(real, synth)
        panel["frechet"] = frechet_distance(real_emb, synth_emb)
        panel["mmd_rbf"] = mmd_rbf(real_emb, synth_emb)
        panel["prdc"] = prdc_metrics(real_emb, synth_emb)
        if with_mauve:
            panel["mauve"] = mauve_score(
                real, synth, device_id=_mauve_device_id(device)
            )

    return panel
