# Notes

- OOM is a benchmark result. Record the context length, KV type, GPU configuration, and the last successful context length.
- Record *where* an OOM happens, not just that it happened. On a nearly-full card the abort can come from the flash-attention scratch pool during prefill rather than from KV allocation, and those two have different workarounds. Capture the llama.cpp backtrace; `run_llama_bench_context.sh` only keeps the abort line.
- Use unique prompts or `cache_prompt=false` for concurrency testing.
- Treat dual P40 layer split primarily as a max-context / capacity experiment unless measurements show a throughput benefit.
- Keep raw outputs unchanged in `results/raw/`; put derived tables in `results/processed/`.

