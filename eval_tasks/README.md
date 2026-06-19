# `eval_tasks/` — custom lm-eval tasks

Benchmarks **not shipped** with lm-evaluation-harness, packaged as custom task YAMLs and loaded
by [`../scripts/evaluate.py`](../scripts/evaluate.py) via `--include-path`:

```sh
python scripts/evaluate.py --config configs/eval/canary.yaml --model <id> --backend vllm \
    --tasks supergpqa --include-path eval_tasks/supergpqa
```

| Task | Dir | What | Validation (Qwen3.5-0.8B) |
|------|-----|------|---------------------------|
| **SuperGPQA** | `supergpqa/` | ~26K graduate-level MC across 285 disciplines; zero-shot generative + answer-letter extraction | 17.6% vs published 16.9 (non-thinking) |
| **IFBench** | `ifbench/` | Instruction-following on 57 new verifiable constraints; verifier functions **vendored** from Ai2 (Apache-2.0) | prompt-strict ~17–19% (published 21.0 is thinking-only) |

## Conventions for custom tasks here

- A task dir holds a `<task>.yaml` (`task:` field = the name passed to `--tasks`) and a `utils.py`
  with `process_docs` / `process_results`.
- lm-eval loads `utils.py` by file path (via `!function`) **without** adding the dir to
  `sys.path`, so if `utils.py` imports sibling modules (IFBench does), it must prepend its own
  dir to `sys.path` first — see `ifbench/utils.py`.
- Vendored third-party verifier code is kept **verbatim** with its upstream `LICENSE` for
  provenance (see `ifbench/`); don't reformat it.
- Any task data caches (e.g. IFBench's nltk corpora) must route to the HF cache volume, not the
  workspace — `.nltk_data/` is gitignored as a backstop.

See `ifbench/README.md` for the IFBench specifics (vendoring, deps, thinking-mode caveat).
