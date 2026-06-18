# `configs/` — experiment configuration

One YAML file per experiment **condition**. The studies are large factorial / condition sweeps
(e.g. Area 4's 16-cell prefix × temperature design; Areas 2 & 5's 5–8 temperature strategies),
so keeping each condition as a small declarative config makes the "N conditions × M samples"
tables in the research doc reproducible. Runners in [`experiments/`](../experiments/) stay thin
and just load a config.

Suggested layout:

```
configs/
├── base.yaml                 # shared defaults (model id, seeds, output paths)
├── area1_token_vs_seq/       # one YAML per condition (A/B/C/D)
├── area2_curriculum/
└── ...
```

**Single-GPU defaults** (this repo targets one RTX PRO 6000 Blackwell, 96 GB — see
[`../CLAUDE.md`](../CLAUDE.md)) belong in `base.yaml`:

- `tensor_parallel_size: 1` (never raise — single GPU)
- `kv_cache_dtype: fp8` (longer context)
- LoRA for fine-tuning; prefer fp8/nvfp4 quantization
- `num_proc <= 16` for `datasets`, dataloader `num_workers ~ 8`
