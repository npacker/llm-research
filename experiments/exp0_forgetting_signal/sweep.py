#!/usr/bin/env python
"""Experiment 0 sweep driver — domain-only (no rehearsal) hyperparameter calibration.

Two stages over a small grid (configs/sweep/exp0.yaml). Stage A is cheap and runs on every
point; Stage B is expensive and runs only on a Pareto shortlist:

  train      [Stage A, all points] one LoRA adapter per grid point (scripts/train.py; serial
             — single GPU), logging train/val loss + held-out domain ppl Δ (learning) and
             general ppl Δ (cheap forgetting proxy) to meta.json
  shortlist  [gate] Pareto front on (domain ppl Δ, general ppl Δ), capped at shortlist_k →
             runs/exp0/shortlist.json; the Stage-B phases honour it
  stageb     [Stage B, shortlist] serve the base + ALL shortlisted LoRAs from ONE vLLM init,
             then run the regression battery (lm-eval local-completions, vs. the base
             reference) AND the looping/incoherence verdict (coherence_check over HTTP)
             against that single engine. Resumable per point. (eval / coherence remain as
             separate phases that boot an in-process engine per point — for one-off debugging
             or `--no-serve`.)
  report     aggregate into runs/exp0/summary.{json,md} — learning, cheap forgetting proxy,
             Stage-B forgetting (IFEval/format vs gsm8k+gpqa/knowledge), coherence, wall-clock,
             and a best-case score (domain gain per unit knowledge forgetting, gated on a
             non-degenerate verdict)

Why two stages: the cheap proxies prune the grid before the costly battery + vLLM generation
run, so Stage B touches ~shortlist_k points instead of all of them. Stage B also validates the
cheap proxy — does general-ppl forgetting track task/knowledge forgetting?

vLLM init is the dominant Stage-B cost. Stage A pays none (training is transformers-only and
reads perplexity off the loss curve), so `stageb` amortises *all* of it: one `vllm serve`
holding the base + every shortlisted adapter, shared by eval (by LoRA-module name) and
coherence (HTTP) — one init total, not two per point. `--phase all` uses it by default;
`--no-serve` (or serve:false) reverts to the per-point in-process engines, and `--base-url`
reuses a server you launched yourself. `--phase plan --emit-serve-cmd` still prints the manual
`vllm serve` command if you want to drive it by hand.

Phases are independent and resumable; run `--phase all` end to end or one at a time.
Per-point artifacts land under <out_root>/<point-id>/. Confirm the chosen winner on the
FULL test sets afterwards (drop --limit; add configs/eval/full_local.yaml + medical.yaml).

Single GPU only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

from llm_core.evaluation import bucket_deltas, flatten_results
from llm_core.sweep import enumerate_grid, pareto_front
from llm_replay.forgetting import learning_per_forgetting

# This runner lives at experiments/<study>/sweep.py → repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/sweep/exp0.yaml")
    p.add_argument(
        "--phase",
        default="all",
        choices=[
            "plan",
            "train",
            "shortlist",
            "eval",
            "coherence",
            "stageb",
            "report",
            "all",
        ],
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print commands, do not run them"
    )
    p.add_argument(
        "--emit-serve-cmd",
        action="store_true",
        help="With --phase plan: print a multi-LoRA `vllm serve` command for amortized eval",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="Reuse an already-running vllm serve (http://host:port root) for Stage B "
        "instead of letting the sweep launch one",
    )
    p.add_argument(
        "--no-serve",
        action="store_true",
        help="Force the legacy per-point in-process engine for Stage B (one vLLM init "
        "per eval and per coherence call) instead of the shared served engine",
    )
    return p.parse_args()


_AXES = ("learning_rate", "num_train_epochs", "num_samples", "lora_rank")
_ID_FORMAT = "lr{learning_rate:g}_e{num_train_epochs:g}_n{num_samples}_r{lora_rank}"


def build_points(cfg: dict) -> list[dict]:
    """Enumerate grid points: main grid (lr×epochs×n at ranks_main) + a rank sub-sweep.

    Thin adapter over :func:`llm_core.sweep.enumerate_grid`: assemble this study's axes
    (the rank sub-sweep becomes explicit extra points at a fixed center) and let the shared
    primitive take the product, de-dup the shared center, and stamp each point id.
    """
    g = cfg["grid"]
    axes = {
        "learning_rate": g["learning_rate"],
        "num_train_epochs": g["num_train_epochs"],
        "num_samples": g["num_samples"],
        "lora_rank": cfg.get("ranks_main", [16]),
    }
    extra = []
    if sub := cfg.get("rank_subsweep"):
        extra = [{**sub["at"], "lora_rank": r} for r in sub["ranks"]]
    return enumerate_grid(axes, extra=extra, id_format=_ID_FORMAT)


def run(cmd: list[str], *, dry_run: bool) -> float:
    """Run a subprocess, returning wall-clock seconds (0.0 on dry-run)."""
    print("  $", " ".join(cmd))
    if dry_run:
        return 0.0
    t0 = time.monotonic()
    subprocess.run(cmd, check=True)
    return time.monotonic() - t0


def point_dir(cfg: dict, pid: str) -> Path:
    return REPO_ROOT / cfg.get("out_root", "runs/exp0") / pid


def phase_train(cfg: dict, points: list[dict], dry_run: bool) -> None:
    """Stage A: train every point + log the cheap proxies (domain & general ppl Δ)."""
    for pt in points:
        out = point_dir(cfg, pt["id"])
        print(f"[sweep:train] {pt['id']}")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/train.py"),
            "--config",
            str(REPO_ROOT / cfg["train_config"]),
            "--model",
            cfg["model"],
            "--output-dir",
            str(out),
            "--num-samples",
            str(pt["num_samples"]),
            "--lr",
            str(pt["learning_rate"]),
            "--epochs",
            str(pt["num_train_epochs"]),
            "--lora-rank",
            str(pt["lora_rank"]),
            "--val-frac",
            str(cfg.get("val_frac", 0.05)),
            "--early-stopping-patience",
            str(cfg.get("early_stopping_patience", 0)),
        ]
        if cfg.get("general_heldout"):
            cmd += [
                "--general-heldout",
                cfg["general_heldout"],
                "--general-heldout-n",
                str(cfg.get("general_heldout_n", 256)),
            ]
        secs = run(cmd, dry_run=dry_run)
        _record(out, {**pt, "train_seconds": secs})


def _deltas(cfg: dict, pid: str) -> tuple[float | None, float | None]:
    """(domain ppl Δ, general ppl Δ) for a point, from its meta.json heldout_perplexity."""
    hp = _load(point_dir(cfg, pid) / "meta.json").get("heldout_perplexity") or {}
    dom = hp.get("domain", {}).get("delta")
    gen = hp.get("general", {}).get("delta")
    return dom, gen


def phase_shortlist(cfg: dict, points: list[dict]) -> list[dict]:
    """Stage-A→B gate: Pareto front on (domain ppl Δ, general ppl Δ), capped at shortlist_k.

    Both axes are minimized — domain Δ low = more learning, general Δ low = less forgetting.
    Points that didn't learn (domain Δ ≥ 0, or no proxy yet) are excluded so the trivial
    'do-nothing' corner can't dominate. Writes shortlist.json; later phases honour it.
    """
    eligible = []
    for pt in points:
        dom, gen = _deltas(cfg, pt["id"])
        if dom is not None and gen is not None and dom < 0:
            eligible.append({**pt, "domain_delta": dom, "general_delta": gen})
    front = pareto_front(eligible, "domain_delta", "general_delta")
    front.sort(
        key=lambda p: learning_per_forgetting(p["domain_delta"], p["general_delta"]),
        reverse=True,
    )
    k = cfg.get("shortlist_k", 6)
    shortlist = front[:k]
    out_root = REPO_ROOT / cfg.get("out_root", "runs/exp0")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "shortlist.json").write_text(
        json.dumps([p["id"] for p in shortlist], indent=2)
    )
    print(
        f"[sweep:shortlist] {len(eligible)} learned / {len(front)} on Pareto front / "
        f"{len(shortlist)} shortlisted (cap {k}):"
    )
    for p in shortlist:
        print(
            f"  - {p['id']}  domainΔ={p['domain_delta']:+.3f}  generalΔ={p['general_delta']:+.3f}"
        )
    return shortlist


def _active_points(cfg: dict, points: list[dict]) -> list[dict]:
    """Stage-B points = the shortlist if it exists, else all (shortlist not yet run)."""
    sl_path = REPO_ROOT / cfg.get("out_root", "runs/exp0") / "shortlist.json"
    if not sl_path.exists():
        return points
    ids = set(json.loads(sl_path.read_text()))
    return [pt for pt in points if pt["id"] in ids]


def phase_eval(cfg: dict, points: list[dict], dry_run: bool) -> None:
    for pt in _active_points(cfg, points):
        out = point_dir(cfg, pt["id"])
        print(f"[sweep:eval] {pt['id']}")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/evaluate.py"),
            "--config",
            str(REPO_ROOT / cfg["eval_config"]),
            "--model",
            cfg["model"],
            "--backend",
            "vllm",
            "--lora",
            str(out / "adapter"),
            "--lora-rank",
            str(pt["lora_rank"]),
            "--generation",
            pt["id"],
            "--output-dir",
            str(out / "eval"),
        ]
        if cfg.get("eval_limit"):
            cmd += ["--limit", str(cfg["eval_limit"])]
        secs = run(cmd, dry_run=dry_run)
        _record(out, {"eval_seconds": secs})


def phase_coherence(cfg: dict, points: list[dict], dry_run: bool) -> None:
    for pt in _active_points(cfg, points):
        out = point_dir(cfg, pt["id"])
        print(f"[sweep:coherence] {pt['id']}")
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/coherence_check.py"),
                "--model",
                cfg["model"],
                "--lora",
                str(out / "adapter"),
                "--lora-rank",
                str(pt["lora_rank"]),
                "--num-samples",
                str(cfg.get("coherence_samples", 200)),
                "--output-dir",
                str(out),
            ],
            dry_run=dry_run,
        )


def _served_eval_cmd(cfg: dict, pt: dict, out: Path, completions_url: str) -> list[str]:
    """evaluate.py argv that hits the shared server by the point's LoRA-module name."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/evaluate.py"),
        "--config",
        str(REPO_ROOT / cfg["eval_config"]),
        "--model",
        pt["id"],  # served LoRA-module name
        "--backend",
        "local-completions",
        "--base-url",
        completions_url,
        "--tokenizer",
        cfg["model"],  # base id → lm-eval can tokenise the adapter name
        "--generation",
        pt["id"],
        "--output-dir",
        str(out / "eval"),
    ]
    if cfg.get("eval_limit"):
        cmd += ["--limit", str(cfg["eval_limit"])]
    return cmd


def _served_coherence_cmd(
    cfg: dict, pt: dict, out: Path, openai_base_url: str
) -> list[str]:
    """coherence_check.py argv that samples the point's LoRA off the shared server."""
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/coherence_check.py"),
        "--model",
        cfg["model"],
        "--base-url",
        openai_base_url,
        "--served-model",
        pt["id"],
        "--num-samples",
        str(cfg.get("coherence_samples", 200)),
        "--output-dir",
        str(out),
    ]


def phase_stage_b(
    cfg: dict, points: list[dict], dry_run: bool, base_url: str | None = None
) -> None:
    """Stage B over ONE vLLM init: serve base + all shortlisted LoRAs, then run the
    regression battery (lm-eval local-completions) and the coherence probe (HTTP) against
    that single engine. Resumable: a point whose results.json / coherence.json already
    exists is skipped. `base_url` reuses an externally-launched server (no launch here)."""
    from llm_core.serving import ServedVLLM

    active = _active_points(cfg, points)
    if not active:
        print("[sweep:stageb] no active points (run shortlist first)")
        return
    lora_modules = {
        pt["id"]: str(point_dir(cfg, pt["id"]) / "adapter") for pt in active
    }
    max_rank = max(pt["lora_rank"] for pt in active)
    server = ServedVLLM(
        cfg["model"],
        lora_modules,
        host=cfg.get("serve_host", "127.0.0.1"),
        port=cfg.get("serve_port", 8000),
        max_lora_rank=max_rank,
        max_loras=cfg.get("max_loras") or len(active),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
        max_model_len=cfg.get("max_model_len"),
        health_timeout_s=cfg.get("serve_health_timeout_s", 600),
        base_url=base_url,
        dry_run=dry_run,
    )
    with server:
        for pt in active:
            out = point_dir(cfg, pt["id"])
            if not dry_run and (out / "eval" / "results.json").exists():
                print(f"[sweep:stageb] eval {pt['id']} — already done, skip")
                continue
            print(f"[sweep:stageb] eval {pt['id']}")
            secs = run(
                _served_eval_cmd(cfg, pt, out, server.completions_url), dry_run=dry_run
            )
            _record(out, {"eval_seconds": secs})
        for pt in active:
            out = point_dir(cfg, pt["id"])
            if not dry_run and (out / "coherence.json").exists():
                print(f"[sweep:stageb] coherence {pt['id']} — already done, skip")
                continue
            print(f"[sweep:stageb] coherence {pt['id']}")
            run(
                _served_coherence_cmd(cfg, pt, out, server.openai_base_url),
                dry_run=dry_run,
            )


def _record(out: Path, extra: dict) -> None:
    """Merge `extra` into the point's sweep_point.json (params + timings)."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sweep_point.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(extra)
    path.write_text(json.dumps(data, indent=2, default=str))


def phase_report(cfg: dict, points: list[dict]) -> None:
    import pandas as pd

    ref_path = REPO_ROOT / cfg["base_canary_ref"]
    base = {}
    if ref_path.exists():
        base = flatten_results(ref_path)
    else:
        print(f"[sweep:report] WARNING: base reference {ref_path} missing — Δ omitted")

    shortlisted = {pt["id"] for pt in _active_points(cfg, points)}
    has_shortlist = (
        REPO_ROOT / cfg.get("out_root", "runs/exp0") / "shortlist.json"
    ).exists()
    fmt_names, know_names = cfg["format_tasks"], cfg["knowledge_tasks"]
    rows = []
    for pt in points:
        out = point_dir(cfg, pt["id"])
        meta = _load(out / "meta.json")
        coh = _load(out / "coherence.json")
        res = out / "eval" / "results.json"
        fmt_deltas, know_deltas = [], []
        if res.exists() and base:
            cur = flatten_results(res)
            if not (set(cur) & set(base)):
                print(
                    f"[sweep:report] WARNING: {pt['id']} eval metric keys don't overlap the "
                    f"base reference (lm-eval version/label skew?) — knowledge/format Δ omitted"
                )
            grouped = bucket_deltas(
                cur, base, {"fmt": fmt_names, "knowledge": know_names}
            )
            fmt_deltas, know_deltas = grouped["fmt"], grouped["knowledge"]
        dom_delta, gen_delta = _deltas(cfg, pt["id"])  # Stage-A proxies (cheap)
        know_delta = statistics.mean(know_deltas) if know_deltas else None
        degenerate = bool(coh.get("degenerate")) if coh else None
        learned = dom_delta is not None and dom_delta < 0
        # Two ranking scores, each in a *single* unit (never mixed within one column):
        #  - proxy_score: cheap Stage-A view — domain vs general *perplexity* Δ (all learned
        #    points), the pre-battery ordering.
        #  - best_case_score: Stage-B view — domain ppl gain per unit *knowledge-accuracy*
        #    forgetting (shortlist only, gated non-degenerate). None where unavailable so it
        #    sorts last; +inf = learned with no measurable forgetting (Pareto-best).
        proxy_score = (
            learning_per_forgetting(dom_delta, gen_delta)
            if learned and gen_delta is not None
            else None
        )
        best_case_score = (
            learning_per_forgetting(dom_delta, -know_delta)
            if learned and know_delta is not None and not degenerate
            else None
        )
        rows.append(
            {
                "id": pt["id"],
                **{a: pt[a] for a in _AXES},
                "train_loss": meta.get("train_loss"),
                "val_perplexity": meta.get("val_perplexity"),
                "domain_ppl_delta": dom_delta,
                "general_ppl_delta": gen_delta,
                "shortlist": (pt["id"] in shortlisted) if has_shortlist else None,
                "fmt_delta": statistics.mean(fmt_deltas) if fmt_deltas else None,
                "knowledge_delta": know_delta,
                "degenerate": degenerate,
                "proxy_score": proxy_score,
                "best_case_score": best_case_score,
            }
        )

    df = pd.DataFrame(rows).sort_values(
        ["best_case_score", "proxy_score"], ascending=False, na_position="last"
    )
    out_root = REPO_ROOT / cfg.get("out_root", "runs/exp0")
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_json(out_root / "summary.json", orient="records", indent=2)

    note = (
        "\n_Stage A (all points): domain_ppl_delta < 0 = learned, general_ppl_delta > 0 = "
        "forgetting (cheap proxies); proxy_score = domain gain per unit general-ppl forgetting. "
        "Stage B (shortlist only): *_delta on the battery < 0 = forgetting; best_case_score = "
        "domain ppl gain per unit *knowledge-accuracy* forgetting (blank if degenerate/missing; "
        "inf = learned with no measurable forgetting). The two scores are in different units and "
        "are never combined — sorted by best_case_score then proxy_score, best first. Confirm the "
        "top non-degenerate point on full test sets._\n"
    )
    (out_root / "summary.md").write_text(df.to_markdown(index=False) + "\n" + note)
    print(f"[sweep:report] wrote {out_root / 'summary.md'} ({len(df)} points)")
    print(df.to_markdown(index=False))


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def emit_serve_cmd(cfg: dict, points: list[dict]) -> None:
    max_rank = max(pt["lora_rank"] for pt in points)
    modules = " ".join(
        f"{pt['id']}={point_dir(cfg, pt['id']) / 'adapter'}" for pt in points
    )
    limit_clause = f" --limit {cfg['eval_limit']}" if cfg.get("eval_limit") else ""
    print(
        "\n[sweep:plan] amortized-eval server (one vLLM init for all points):\n"
        f"  vllm serve {cfg['model']} --enable-lora --max-lora-rank {max_rank} "
        f"--max-loras 4 --lora-modules {modules}\n"
        "  then per point: scripts/evaluate.py --backend local-completions "
        "--base-url http://localhost:8000/v1/completions --model <point-id> "
        f"--config {cfg['eval_config']}{limit_clause}\n"
        "  (note: local-completions sets tokenizer=<point-id>; pass the base model id as the\n"
        "   tokenizer if your lm-eval version can't resolve the adapter name.)\n"
    )


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    points = build_points(cfg)
    print(
        f"[sweep] {len(points)} grid points (out_root={cfg.get('out_root', 'runs/exp0')})"
    )

    if args.phase in ("plan", "all"):
        for pt in points:
            print(f"  - {pt['id']}")
        if args.emit_serve_cmd:
            emit_serve_cmd(cfg, points)
        if args.phase == "plan":
            return
    if args.phase in ("train", "all"):
        phase_train(cfg, points, args.dry_run)
    if args.phase in ("shortlist", "all"):
        phase_shortlist(cfg, points)

    # Stage B (eval + coherence). Served mode (default) shares ONE vLLM init across both;
    # --no-serve / serve:false falls back to a fresh in-process engine per call.
    serve_on = cfg.get("serve", True) and not args.no_serve
    if args.phase == "stageb" or (args.phase == "all" and serve_on):
        phase_stage_b(cfg, points, args.dry_run, base_url=args.base_url)
    else:
        if args.phase in ("eval", "all"):
            phase_eval(cfg, points, args.dry_run)
        if args.phase in ("coherence", "all"):
            phase_coherence(cfg, points, args.dry_run)

    if args.phase in ("report", "all"):
        phase_report(cfg, points)


if __name__ == "__main__":
    main()
