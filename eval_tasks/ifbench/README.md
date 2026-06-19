# IFBench — custom lm-eval task (vendored verifiers)

Runs [IFBench](https://github.com/allenai/IFBench) (Ai2, "Generalizing Verifiable
Instruction Following", NeurIPS 2025) under our lm-eval runner. IFBench tests
instruction-following on 57 **new** verifiable, out-of-domain constraints — its
premise is that models overfit IFEval, so high IFEval ≠ high IFBench, which makes
it a harder-to-game instruction-following signal.

## Files
- `instructions.py`, `instructions_registry.py`, `instructions_util.py` — **vendored
  verbatim** from `github.com/allenai/IFBench` (Apache-2.0, see `LICENSE`). These hold
  the constraint verifier functions. Kept unmodified for provenance; do not reformat.
- `utils.py` — ours. Mirrors lm-eval's `ifeval` `process_results` (strict/loose
  scoring) but imports the vendored IFBench registry. Adds the task dir to `sys.path`
  so the flat sibling imports resolve when lm-eval loads it via `!function`.
- `ifbench.yaml` — task config; dataset `allenai/IFBench_test` (data: ODC-BY-1.0).

## Run
```sh
python scripts/evaluate.py --config configs/eval/canary.yaml \
  --model <id> --backend vllm --tasks ifbench --include-path eval_tasks/ifbench
```

## Deps
Verifiers need `emoji` + `syllapy` (in `pyproject.toml`) and `nltk` data
(`punkt`, `stopwords`, POS tagger), fetched on first use by `utils.py`.

## Caveat
Published Qwen3.5-0.8B IFBench (21.0) is **thinking-mode**; there is no non-thinking
0.8B cell, so a non-thinking run validates the integration but isn't a direct
comparison. For thinking mode, set `enable_thinking: true` + `think_end_token: "</think>"`.
