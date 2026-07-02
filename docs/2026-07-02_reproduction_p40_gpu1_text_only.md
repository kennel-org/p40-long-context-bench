# Reproduction: P40 GPU1 text-only llama.cpp benchmark

- Date: 2026-07-02
- Host: <host>
- Repository: `$HOME/projects/p40-long-context-bench`
- Scope: Tesla P40 GPU1, llama.cpp, text-only benchmark
- Result plot: `results/figures/p40_long_context_summary.png`

## Purpose

This run measures the text-only throughput of the tuned P40 + llama.cpp setup on the same axes as the provided reference image:

- prefill throughput vs context length
- decode throughput vs context length
- aggregate and per-stream throughput vs concurrency

It is not a vision benchmark. The benchmark commands do not pass `--mmproj`.

## Model

Text model:

```text
$HOME/models/youssofal_qwen36heretic/Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M/Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf
```

SHA256:

```text
ae2fb73ac0da875640269f1e65e9c7fb415b066c6d544c3eef9adb0d03f04792
```

Model metadata recorded by `llama-bench`:

```text
model_type: qwen35moe 35B.A3B Q4_K - Medium
model_size: 21155768832
model_n_params: 34660610688
```

Normal service projector, not used in this benchmark:

```text
$HOME/models/youssofal_qwen36heretic/mmproj-Qwen3.6-35B-A3B-Abliterated-Heretic.gguf
```

SHA256:

```text
42ef6d5f3564946163628d28ce1380b729ff3634dc37482506bb1eb951138f8c
```

## llama.cpp

```text
source: $HOME/src/llama.cpp
commit: bbce619adb409880fb6db850a1c5a5f36a4dc7b1
build_number: 9283
build_commit in CSV: bbce619ad
llama-server version: 9283 (bbce619ad)
```

## Hardware

`CUDA_VISIBLE_DEVICES=1` was used. Inside llama.cpp this appears as CUDA device 0 because only one physical GPU is exposed to the process.

```text
GPU: Tesla P40
visible VRAM: 22905 MiB
compute capability: 6.1
```

## Benchmark Configuration

Canonical config:

```bash
configs/p40_usb4_q8kv.env
```

Effective parameters:

```bash
LABEL=p40_usb4_q8kv
CUDA_VISIBLE_DEVICES=1
LLAMA_CPP_DIR=$HOME/src/llama.cpp
CTX_SIZE=32768
N_GPU_LAYERS=41
PARALLEL=8
PROMPT_TOKENS=512
GEN_TOKENS=128
BATCH_SIZE=2048
UBATCH_SIZE=512
CACHE_TYPE_K=q8_0
CACHE_TYPE_V=q8_0
FLASH_ATTN=1
REPETITIONS=3
```

Important differences from the normal service:

| Item | Benchmark | Normal `llama-cpp-8080.service` |
|---|---|---|
| vision projector | disabled, no `--mmproj` | enabled with `--mmproj` |
| exposed host | `127.0.0.1` | `0.0.0.0` |
| context | `32768` | `32768` |
| GPU layers | `41` | `41` |
| parallel | `8` | `1` |
| warmup | enabled | `--no-warmup` |
| metrics | enabled | not part of baseline service command |

## Commands

Stop the normal service before binding port 8080:

```bash
systemctl --user stop llama-cpp-8080.service
```

Start the benchmark server:

```bash
cd $HOME/projects/p40-long-context-bench
scripts/run_llama_server.sh configs/p40_usb4_q8kv.env \
  2>&1 | tee results/raw/p40_usb4_q8kv_server.log
```

In another terminal, run the concurrency benchmark:

```bash
python3 scripts/bench_concurrency.py \
  --out results/raw/p40_usb4_q8kv_concurrency.csv \
  --checkpoint results/processed/p40_usb4_q8kv_concurrency_checkpoint.json
```

Stop the foreground benchmark server with Ctrl-C, then run the context sweep:

```bash
scripts/run_llama_bench_context.sh configs/p40_usb4_q8kv.env
```

Restore the normal service:

```bash
systemctl --user start llama-cpp-8080.service
systemctl --user is-active llama-cpp-8080.service
curl -fsS http://127.0.0.1:8080/health
```

Generate the plot:

```bash
python3 scripts/plot_results.py \
  --include-reference \
  --title 'Tesla P40 GPU1 + llama.cpp Qwen3.6-35B-A3B-Abliterated-Heretic Q4_K_M q8 KV'
```

## Artifacts

| Artifact | Path |
|---|---|
| environment snapshot | `results/raw/env_20260702_initial.txt` |
| benchmark server log | `results/raw/p40_usb4_q8kv_server.log` |
| context CSV | `results/raw/p40_usb4_q8kv_context.csv` |
| context log | `results/raw/p40_usb4_q8kv_context.log` |
| context checkpoint | `results/processed/p40_usb4_q8kv_context_checkpoint.json` |
| concurrency CSV | `results/raw/p40_usb4_q8kv_concurrency.csv` |
| concurrency checkpoint | `results/processed/p40_usb4_q8kv_concurrency_checkpoint.json` |
| plot | `results/figures/p40_long_context_summary.png` |
| operational memo | `$HOME/projects/kennel-system-laboratory/docs/ops/2026-07-02_p40_gpu1_llamacpp_benchmark.md` |

## Results

Context sweep:

| context | prefill tok/s | decode tok/s |
|---:|---:|---:|
| 512 | 964.54 | 50.05 |
| 8192 | 729.59 | 45.38 |
| 16384 | 589.16 | 42.80 |
| 32768 | 427.15 | 37.52 |

Concurrency sweep:

| concurrency | aggregate tok/s | per-stream tok/s |
|---:|---:|---:|
| 1 | 34.04 | 34.04 |
| 2 | 44.53 | 22.26 |
| 4 | 51.98 | 13.00 |
| 8 | 50.35 | 6.29 |
| 12 | 50.66 | 4.22 |
| 16 | 48.05 | 3.00 |

## Interpretation Notes

- This is a text-only benchmark. Vision projector costs are excluded.
- The context sweep is from `llama-bench`, not the HTTP server.
- The concurrency benchmark uses `/completion` and `cache_prompt: false`, but llama.cpp may still reuse slot/checkpoint state for prompts with a common prefix.
- The server log contains `selected slot by LCP similarity` and `graphs reused`, so treat concurrency results as tuned server throughput, not strict cold-prompt throughput.
- For stricter cold-prompt measurements, disable prompt cache and randomize the prompt body:

```bash
EXTRA_ARGS="--cache-ram 0" scripts/run_llama_server.sh configs/p40_usb4_q8kv.env
```

## Rollback

```bash
# Confirm the current listener.
ss -lntp | rg ':8080|State'

# If a benchmark server is still running, confirm and stop only that PID.
pgrep -af 'llama-server.*--metrics.*--parallel 8'
kill <PID>

# Restore the normal service.
systemctl --user start llama-cpp-8080.service
systemctl --user is-active llama-cpp-8080.service
curl -fsS http://127.0.0.1:8080/health
```
