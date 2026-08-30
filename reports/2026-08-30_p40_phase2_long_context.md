# P40 Phase 2: Long Context Extension (64K / 128K / 160K / 192K)

- Date: 2026-08-30
- Scope: `benchmark_plan.md` Phase 2 — extend the GPU1 text-only sweep beyond 32K and record OOM as data
- Model: `Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf`
- Model SHA256: `ae2fb73ac0da875640269f1e65e9c7fb415b066c6d544c3eef9adb0d03f04792`
- Model type: `qwen35moe 35B.A3B Q4_K - Medium`
- Model `n_ctx_train`: `262144`
- Mode: text-only, no `--mmproj`
- llama.cpp commit: `bbce619adb409880fb6db850a1c5a5f36a4dc7b1`
- llama.cpp build: `9283 (bbce619ad)`
- GPU: Tesla P40, physical GPU1 via `CUDA_VISIBLE_DEVICES=1`, 22905 MiB visible
- Config: `configs/p40_usb4_q8kv_long.env`

The llama.cpp source tree and both binaries are byte-identical to the 2026-07-02 run
(same commit, same `build_number` 9283, clean worktree), so these numbers extend the
existing `p40_usb4_q8kv` series directly rather than forming a separate series.

## Headline

The single-P40 ceiling for this model at `q8_0/q8_0` KV and `-ngl 41` is **163840 tokens**.
`196608` fails, and it fails during prefill in the flash-attention scratch allocation —
not at KV cache allocation.

Decode throughput holds up far worse on the P40 than on the reference card as context grows:
the P40 retains 38% of its short-context decode rate at 160K, where the RTX PRO 6000
reference retains 79% at 192K.

## Extended Context Sweep

New measurements (`REPETITIONS=3`, `PROMPT_TOKENS=512`, `GEN_TOKENS=128`, `-b 2048 -ub 512 -fa 1`):

| context | prefill tok/s | stddev | decode tok/s | stddev | status |
|---:|---:|---:|---:|---:|---|
| 65536 | 274.78 | 0.568 | 30.49 | 0.036 | ok |
| 131072 | 160.28 | 0.677 | 21.36 | 0.031 | ok |
| 163840 | 132.62 | 0.086 | 19.16 | 0.029 | ok — last successful |
| 196608 | — | — | — | — | **OOM** |

Combined with the 2026-07-02 run, the full P40 series is now:

| context | prefill tok/s | decode tok/s |
|---:|---:|---:|
| 512 | 964.54 | 50.05 |
| 8192 | 729.59 | 45.38 |
| 16384 | 589.16 | 42.80 |
| 32768 | 427.15 | 37.52 |
| 65536 | 274.78 | 30.49 |
| 131072 | 160.28 | 21.36 |
| 163840 | 132.62 | 19.16 |

Retention relative to the 512-token measurement:

| context | prefill retained | decode retained |
|---:|---:|---:|
| 32768 | 44.3% | 75.0% |
| 65536 | 28.5% | 60.9% |
| 131072 | 16.6% | 42.7% |
| 163840 | 13.7% | 38.3% |

## OOM Detail (196608)

Reproduced twice, both aborting with `rc=134` at
`ggml/src/ggml-cuda/ggml-cuda.cu:102` (`ggml_cuda_error`). The backtrace is identical
in both runs:

```text
#3  ggml_cuda_error(...)
#4  ggml_cuda_pool_vmm::alloc(unsigned long, unsigned long*)
#5  launch_fattn<256, 4, 8>(...)
#6  ggml_cuda_flash_attn_ext_tile_case<256, 256>(...)
#7  ggml_backend_cuda_graph_compute(...)
...
#11 llama_context::decode(llama_batch const&)
#12 llama_decode()
#13 test_prompt(llama_context*, int, int, int)
```

This matters for interpretation: at 196608 the model weights and the KV cache **do**
allocate successfully. The process runs, reaches `test_prompt`, and then exhausts VRAM
in the CUDA VMM pool that backs the flash-attention kernel's scratch buffer. The failure
is a prefill-time working-set problem sitting on top of an already nearly-full card, not
a KV capacity problem.

Sampled VRAM during each run (`nvidia-smi`, 10-90s interval — sampled, not a true peak):

| context | observed VRAM | note |
|---:|---:|---|
| 65536 | ~21383 MiB | steady |
| 131072 | ~22037 MiB | steady |
| 163840 | 22461 → 22725 MiB | grows during run, completes |
| 196608 (`-ub 512`) | → 22901 MiB | OOM after ~68 s |
| 196608 (`-ub 256`) | → 22889 MiB | OOM after ~16 min |

In every long run the VMM pool grows monotonically toward the 22905 MiB limit as the
prefill proceeds.

### Does a smaller ubatch reach 192K?

No. Because the failing allocation is the flash-attention scratch buffer, `-ub 256` was
tested as a variant (all other parameters unchanged). It survives far longer — roughly
16 minutes versus 68 seconds — but aborts at the identical call site with the pool at
22889 MiB. Halving the micro-batch delays the exhaustion; it does not avoid it.

`196608` is therefore recorded as out of reach for a single P40 in this configuration,
not merely out of reach at `-ub 512`.

## Comparison vs RTX PRO 6000 Reference

Reference values are the handoff figures already encoded in `scripts/plot_results.py`.

| Metric | RTX PRO 6000 reference | P40 measured | Ratio |
|---|---:|---:|---:|
| prefill @ 512 | 4770 tok/s | 964.54 tok/s | 20.2% |
| prefill @ 32K | 5704 tok/s | 427.15 tok/s | 7.5% |
| prefill @ 64K | 5227 tok/s | 274.78 tok/s | 5.3% |
| prefill @ 128K | 4318 tok/s | 160.28 tok/s | 3.7% |
| decode @ 512 | 74.1 tok/s | 50.05 tok/s | 67.5% |
| decode @ 32K | 70.7 tok/s | 37.52 tok/s | 53.1% |
| decode @ 64K | 68.5 tok/s | 30.49 tok/s | 44.5% |
| decode @ 128K | 65.0 tok/s | 21.36 tok/s | 32.9% |
| decode @ 160K | not in reference set | 19.16 tok/s | n/a |
| decode @ 192K | 58.6 tok/s | OOM | n/a |
| aggregate @ 8 concurrent | 322 tok/s | 50.35 tok/s | 16% |
| per-stream @ 8 concurrent | about 40 tok/s | 6.29 tok/s | 16% |
| max measured context | 256K | 163840 | 64% |

The reference has no 163840 data point, so no ratio is given for that row rather than
interpolating one.

The gap widens with context on both axes, and it widens faster for prefill than for
decode. Prefill falls from a fifth of the reference at short context to under a
twenty-fifth at 128K.

## Thermals

Relevant because this run was the first after the GPU1 cooling fix. Sampled throughout
all long runs:

- idle before the run: 36 °C (GPU1), 39 °C (GPU0)
- sustained load: 62 → 84 °C, plateauing at 83-84 °C
- power draw under load: 190-220 W against the 250 W cap
- `clocks_throttle_reasons.active` was `0x0000000000000000` in **every** sample across
  all runs, including the ~16 minute 192K attempt

No thermal or power throttling occurred. The measurements above are not
cooling-limited.

## Wall Clock

| Run | Wall clock |
|---|---:|
| scripted sweep 65536 | 201 s |
| scripted sweep 131072 | 521 s |
| scripted sweep 196608 (OOM) | 68 s |
| scripted sweep total | 790 s |
| 163840 | ~749 s |
| 196608 `-ub 256` variant (OOM) | ~975 s |
| `llama-cpp-8080.service` downtime | 47 min 19 s (17:19:48 → 18:07:08 JST) |

## Method Note

`65536`, `131072`, and `196608` were produced by
`scripts/run_llama_bench_context.sh configs/p40_usb4_q8kv_long.env`.

`163840` was added afterwards as a bisection point between the last success (131072) and
the first failure (196608). It was run as a direct `llama-bench` invocation with
byte-identical arguments and its two CSV rows were appended to
`results/raw/p40_usb4_q8kv_long_context.csv`. The config file has since been updated to
include `163840`, so a single re-run of the config now reproduces the whole table.

`scripts/run_llama_bench_context.sh` truncates its output CSV at start and does not
resume from its checkpoint, so the extension was run under a separate `LABEL`
(`p40_usb4_q8kv_long`) to leave the 2026-07-02 CSV untouched. `scripts/plot_results.py`
strips the `_long` suffix so both files draw as one continuous series.

## Caveats

- Text-only. Vision projector costs are excluded; the normal service uses `--mmproj`.
- `llama-bench` measurements, not HTTP endpoint measurements.
- Concurrency rows in the comparison table are carried over unchanged from 2026-07-02
  and were not re-measured; they remain tuned-server throughput, not strict cold-prompt
  throughput.
- VRAM figures are `nvidia-smi` samples taken at 10-90 s intervals, so they under-report
  true peaks.
- The ceiling is bracketed, not resolved to the token: `163840` succeeds and `196608`
  fails. Nothing between the two was tested.

## Artifacts

| Artifact | Path |
|---|---|
| extended context CSV | `results/raw/p40_usb4_q8kv_long_context.csv` |
| extended context log | `results/raw/p40_usb4_q8kv_long_context.log` |
| extended context checkpoint | `results/processed/p40_usb4_q8kv_long_context_checkpoint.json` |
| config | `configs/p40_usb4_q8kv_long.env` |
| plot | `results/figures/p40_long_context_summary.png` |
| prior run report | `reports/2026-07-02_p40_llamacpp_qwen_longctx.md` |

## Follow-ups

- Phase 2 is complete for the single-GPU case. Reaching 192K or beyond requires the dual
  P40 layer split (`configs/p40_dual_layer_q8kv.env`), which needs GPU0 — currently held
  by ollama (about 19.7 GB).
- `configs/p40_oculink_f16kv.env`, `configs/p40_oculink_q8kv.env`, and
  `configs/p40_oculink_q8k_q4v.env` remain unmeasured. `q8_0` K with `q4_0` V would be
  the cheapest way to test whether a smaller V cache moves the ceiling.
- `scripts/run_llama_bench_context.sh` writes a checkpoint but never reads it, and
  truncates the output CSV on every run. Resume is not actually implemented.
