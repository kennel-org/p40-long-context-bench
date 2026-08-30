# Artifact Index

## Reproduction

- `docs/2026-07-02_reproduction_p40_gpu1_text_only.md`: canonical reproduction steps for the GPU1 text-only run.
- `docs/2026-07-02_p40_llamacpp_long_context_bench_handoff.md`: original planning and handoff notes.
- `docs/benchmark_plan.md`: benchmark phases and comparison axes.
- `docs/notes.md`: general measurement notes.

## Configs

- `configs/p40_usb4_q8kv.env`: measured GPU1 text-only q8 KV configuration.
- `configs/p40_usb4_q8kv_long.env`: measured GPU1 depth extension (64K/128K/160K/192K). 192K is expected to OOM and is kept in the depth list so a re-run reproduces that result.
- `configs/p40_oculink_f16kv.env`: planned OCuLink f16 KV configuration.
- `configs/p40_oculink_q8kv.env`: planned OCuLink q8 KV configuration.
- `configs/p40_oculink_q8k_q4v.env`: planned q8 K / q4 V configuration.
- `configs/p40_dual_layer_q8kv.env`: planned dual P40 layer split configuration.

## Scripts

- `scripts/collect_env.sh`: collect host, GPU, llama.cpp, and model metadata.
- `scripts/run_llama_server.sh`: start a benchmark-local llama-server from a config file.
- `scripts/bench_concurrency.py`: HTTP `/completion` concurrency benchmark with checkpointing.
- `scripts/bench_existing_server.py`: smoke benchmark against an already running server.
- `scripts/run_llama_bench_context.sh`: context-depth sweep through `llama-bench` with checkpointing.
- `scripts/plot_results.py`: build the three-panel comparison plot.

## Results

- `results/raw/p40_usb4_q8kv_context.csv`: `llama-bench` context sweep (512-32K, 2026-07-02).
- `results/raw/p40_usb4_q8kv_long_context.csv`: `llama-bench` depth extension (64K-160K, 2026-08-30). Plotted as one series with the file above.
- `results/raw/p40_usb4_q8kv_concurrency.csv`: concurrency sweep.
- `results/figures/p40_long_context_summary.png`: final plot.

Ignored local artifacts:

- `results/raw/env_20260702_initial.txt`: environment snapshot.
- `results/raw/p40_usb4_q8kv_server.log`: benchmark server log.
- `results/raw/p40_usb4_q8kv_context.log`: context sweep progress and stderr.
- `results/raw/p40_usb4_q8kv_long_context.log`: depth extension progress and the 192K abort.
- `results/processed/p40_usb4_q8kv_context_checkpoint.json`: context sweep checkpoint.
- `results/processed/p40_usb4_q8kv_long_context_checkpoint.json`: depth extension checkpoint.
- `results/processed/p40_usb4_q8kv_concurrency_checkpoint.json`: concurrency checkpoint.

## Reports

- `reports/2026-07-02_p40_llamacpp_qwen_longctx.md`: measured result summary (Phases 1 and 3).
- `reports/2026-08-30_p40_phase2_long_context.md`: Phase 2 depth extension, VRAM ceiling, and OOM analysis.

## License

- `LICENSE`: MIT License for this repository's code, scripts, configuration templates, documentation, and benchmark metadata.
- External dependencies, llama.cpp, model weights, projector files, drivers, and third-party reference results are not covered by this repository license.

## External Operational Memo

- `$HOME/projects/kennel-system-laboratory/docs/ops/2026-07-02_p40_gpu1_llamacpp_benchmark.md`: service stop/start operation memo with rollback.
