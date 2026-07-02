# P40 + llama.cpp Qwen Long Context Report

- Date: 2026-07-02
- Model: `Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf`
- Model SHA256: `ae2fb73ac0da875640269f1e65e9c7fb415b066c6d544c3eef9adb0d03f04792`
- Model type: `qwen35moe 35B.A3B Q4_K - Medium`
- Mode: text-only, no `--mmproj`
- llama.cpp commit: `bbce619adb409880fb6db850a1c5a5f36a4dc7b1`
- llama.cpp build: `9283 (bbce619ad)`
- GPU: Tesla P40, physical GPU1 via `CUDA_VISIBLE_DEVICES=1`
- Config: `configs/p40_usb4_q8kv.env`

## Summary

The measured GPU1 text-only run uses `N_GPU_LAYERS=41`, `CTX_SIZE=32768`, `PARALLEL=8`, `q8_0/q8_0` KV cache, `BATCH_SIZE=2048`, `UBATCH_SIZE=512`, and flash attention enabled.

At 32768 context, measured throughput was `427.15 tok/s` prefill and `37.52 tok/s` decode. The HTTP concurrency benchmark peaked around `51.98 tok/s` aggregate at 4 concurrent requests, with similar aggregate throughput through 8-12 concurrent requests but lower per-stream throughput.

## Comparison

| Metric | RTX PRO 6000 reference | P40 measured | Ratio |
|---|---:|---:|---:|
| prefill peak | 5700-6000 tok/s | 964.54 tok/s | 16-17% |
| decode short context | 74 tok/s | 50.05 tok/s | 68% |
| decode @ 32K | 70.7 tok/s | 37.52 tok/s | 53% |
| decode @ 64K | 68.5 tok/s | not measured | n/a |
| decode @ 128K | 65.0 tok/s | not measured | n/a |
| decode @ 192K | 58.6 tok/s | not measured | n/a |
| aggregate @ 8 concurrent | 322 tok/s | 50.35 tok/s | 16% |
| per-stream @ 8 concurrent | about 40 tok/s | 6.29 tok/s | 16% |
| max measured context | 256K ref | 32768 | n/a |

## Context Sweep

| context | prefill tok/s | decode tok/s |
|---:|---:|---:|
| 512 | 964.54 | 50.05 |
| 8192 | 729.59 | 45.38 |
| 16384 | 589.16 | 42.80 |
| 32768 | 427.15 | 37.52 |

## Concurrency Sweep

| concurrency | aggregate tok/s | per-stream tok/s |
|---:|---:|---:|
| 1 | 34.04 | 34.04 |
| 2 | 44.53 | 22.26 |
| 4 | 51.98 | 13.00 |
| 8 | 50.35 | 6.29 |
| 12 | 50.66 | 4.22 |
| 16 | 48.05 | 3.00 |

## Caveats

- Vision is disabled in the benchmark; the normal service uses `--mmproj`.
- The context sweep is a `llama-bench` measurement, not an HTTP endpoint measurement.
- The concurrency benchmark sets `cache_prompt: false`, but prompts share a common prefix and llama.cpp logs show slot/checkpoint reuse. Treat concurrency results as tuned server throughput, not strict cold-prompt throughput.

## Artifacts

- Raw CSV: `results/raw/`
- Processed checkpoints/tables: `results/processed/`
- Figures: `results/figures/`
- Reproduction: `docs/2026-07-02_reproduction_p40_gpu1_text_only.md`
