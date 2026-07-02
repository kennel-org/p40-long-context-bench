# Scripts

Run scripts from the repository root:

```bash
cd $HOME/projects/p40-long-context-bench
```

## Environment

```bash
scripts/collect_env.sh
```

Collects host, GPU, llama.cpp, and model metadata.

## Context Sweep

```bash
scripts/run_llama_bench_context.sh configs/p40_usb4_q8kv.env
```

Runs `llama-bench` across configured context depths and writes:

- `results/raw/<LABEL>_context.csv`
- `results/raw/<LABEL>_context.log`
- `results/processed/<LABEL>_context_checkpoint.json`

## Benchmark Server

```bash
scripts/run_llama_server.sh configs/p40_usb4_q8kv.env
```

Starts a local benchmark `llama-server` with metrics enabled. The GPU1 text-only benchmark config does not pass `--mmproj`.

## Concurrency

```bash
python3 scripts/bench_concurrency.py \
  --out results/raw/p40_usb4_q8kv_concurrency.csv \
  --checkpoint results/processed/p40_usb4_q8kv_concurrency_checkpoint.json
```

Runs concurrent `/completion` requests against the local benchmark server.

## Existing Server Smoke Test

```bash
python3 scripts/bench_existing_server.py
```

Runs a lightweight check against an already running server.

## Plot

```bash
python3 scripts/plot_results.py --include-reference
```

Generates:

```text
results/figures/p40_long_context_summary.png
```
