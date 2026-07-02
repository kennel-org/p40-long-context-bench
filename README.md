# p40-long-context-bench

Benchmarking long-context LLM inference on Tesla P40 + llama.cpp.

This repository measures practical limits of a tuned Pascal-generation P40 setup:

- prefill throughput vs context length
- decode throughput vs context length
- aggregate throughput vs concurrency
- KV cache quantization trade-offs
- OCuLink / USB4 / dual-GPU behavior

## Quick Start

Set a model path, collect the environment, then run a small context sweep:

```bash
export MODEL=$HOME/models/youssofal_qwen36heretic/Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M/Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf
scripts/collect_env.sh
scripts/run_llama_bench_context.sh configs/p40_oculink_q8kv.env
```

For concurrent throughput, start a server in one terminal and run the client in another:

```bash
scripts/run_llama_server.sh configs/p40_oculink_q8kv.env
python3 scripts/bench_concurrency.py --out results/raw/p40_oculink_q8kv_concurrency.csv
```

Outputs are written under `results/raw/`, `results/processed/`, and `results/figures/`.

Generate the three-panel plot:

```bash
python3 scripts/plot_results.py --include-reference
```

For a reference-only visual check:

```bash
python3 scripts/plot_results.py \
  --reference-only \
  --title "Qwen3.6-27B-NVFP4  ·  RTX PRO 6000 (sm_120, vLLM nightly / Marlin)"
```

## Documentation

- Reproduction: `docs/2026-07-02_reproduction_p40_gpu1_text_only.md`
- Artifact index: `docs/artifacts.md`
- Script guide: `scripts/README.md`
- Result report: `reports/2026-07-02_p40_llamacpp_qwen_longctx.md`

## Current GPU1 Text-Only Run

The `p40_usb4_q8kv` run is a text-only llama.cpp benchmark on physical GPU1:

```bash
CUDA_VISIBLE_DEVICES=1
MODEL=$HOME/models/youssofal_qwen36heretic/Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M/Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf
CTX_SIZE=32768
N_GPU_LAYERS=41
PARALLEL=8
BATCH_SIZE=2048
UBATCH_SIZE=512
CACHE_TYPE_K=q8_0
CACHE_TYPE_V=q8_0
FLASH_ATTN=1
```

Important scope notes:

- Vision is disabled for the benchmark. No `--mmproj` argument is passed by `scripts/run_llama_server.sh` or `scripts/run_llama_bench_context.sh`.
- The normal `llama-cpp-8080.service` is different: it uses the same text GGUF with `--mmproj $HOME/models/youssofal_qwen36heretic/mmproj-Qwen3.6-35B-A3B-Abliterated-Heretic.gguf`.
- The benchmark server uses `--parallel 8`; the normal service uses `--parallel 1`.
- The benchmark server enables metrics and uses warmup; the normal service is started with `--no-warmup`.
- `q8_0` V-cache requires flash attention for `llama-bench`, so context sweeps use `-fa 1`.

The generated plot for this run is:

```bash
results/figures/p40_long_context_summary.png
```

## Interpreting Throughput

`llama-bench` context sweeps are model-level text-only measurements and do not include image projector work.

The concurrency benchmark drives the OpenAI-compatible `/completion` endpoint. It sets `cache_prompt: false`, but llama.cpp can still reuse slot/checkpoint state when prompts share a long common prefix. The server log may show lines such as:

```text
selected slot by LCP similarity
graphs reused
```

Treat concurrency results as a tuned server-throughput measurement, not a strict cold-prompt benchmark. For stricter cold-prompt numbers, start the benchmark server with prompt cache disabled and use more randomized prompt bodies, for example:

```bash
EXTRA_ARGS="--cache-ram 0" scripts/run_llama_server.sh configs/p40_usb4_q8kv.env
```

## License

This repository's code, scripts, configuration templates, documentation, and benchmark metadata are licensed under the MIT License. See `LICENSE`.

This license does not grant rights to external dependencies, llama.cpp itself, model weights, projector files, drivers, or third-party reference results. Use those artifacts under their own licenses and terms.
