# Benchmark Plan

Measured GPU1 text-only results are recorded in:

- `docs/2026-07-02_reproduction_p40_gpu1_text_only.md`
- `reports/2026-07-02_p40_llamacpp_qwen_longctx.md`
- `reports/2026-08-30_p40_phase2_long_context.md` (Phase 2 extension)

- Phase 1: **done** (2026-07-02, GPU1/USB4 rather than OCuLink). Environment, model metadata, and single-GPU context sweeps at 8K, 16K, and 32K.
- Phase 2: **done** (2026-08-30). Extended to 64K, 128K, and 160K. 192K OOMs and is recorded as data; the single-P40 ceiling at `q8_0/q8_0` KV is 163840.
- Phase 3: **done** (2026-07-02). `llama-server` throughput at concurrency 1, 2, 4, 8, 12, and 16.
- Phase 4: **done**. Figures and reports comparing P40 results to the RTX PRO 6000 reference values from the handoff memo.
- Remaining: OCuLink configs (`f16kv`, `q8kv`, `q8k_q4v`) and the dual P40 layer split are still unmeasured. Reaching 192K needs the dual-GPU split, which requires GPU0.

Comparison rows (filled 2026-08-30; see `reports/2026-08-30_p40_phase2_long_context.md`):

| Metric | RTX PRO 6000 reference | P40 measured | Ratio |
|---|---:|---:|---:|
| prefill peak | 5794 tok/s @ 16K | 964.54 tok/s @ 512 | 17% |
| decode short context | 74.1 tok/s | 50.05 tok/s | 67.5% |
| decode @ 32K | 70.7 tok/s | 37.52 tok/s | 53.1% |
| decode @ 64K | 68.5 tok/s | 30.49 tok/s | 44.5% |
| decode @ 128K | 65.0 tok/s | 21.36 tok/s | 32.9% |
| decode @ 192K | 58.6 tok/s | OOM | n/a |
| aggregate @ 8 concurrent | 322 tok/s | 50.35 tok/s | 16% |
| per-stream @ 8 concurrent | about 40 tok/s | 6.29 tok/s | 16% |
| max practical context | 256K | 163840 | 64% |
