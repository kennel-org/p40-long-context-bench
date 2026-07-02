# Benchmark Plan

Measured GPU1 text-only results are recorded in:

- `docs/2026-07-02_reproduction_p40_gpu1_text_only.md`
- `reports/2026-07-02_p40_llamacpp_qwen_longctx.md`

- Phase 1: Record environment, model metadata, and OCuLink P40 single-GPU context sweeps at 8K, 16K, and 32K.
- Phase 2: Extend to 64K, 128K, and 192K where VRAM permits. Record OOM as data.
- Phase 3: Measure `llama-server` throughput at concurrency 1, 2, 4, 8, 12, and 16.
- Phase 4: Generate figures and a short report comparing P40 results to the RTX PRO 6000 reference values from the handoff memo.

Default comparison rows:

| Metric | RTX PRO 6000 reference | P40 measured | Ratio |
|---|---:|---:|---:|
| prefill peak | 5700-6000 tok/s | TBD | TBD |
| decode short context | 74 tok/s | TBD | TBD |
| decode @ 32K | TBD | TBD | TBD |
| decode @ 64K | TBD | TBD | TBD |
| decode @ 128K | TBD | TBD | TBD |
| decode @ 192K | 59 tok/s | TBD | TBD |
| aggregate @ 8 concurrent | 322 tok/s | TBD | TBD |
| per-stream @ 8 concurrent | about 40 tok/s | TBD | TBD |
| max practical context | 256K | TBD | TBD |
