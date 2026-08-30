# Notes

- OOM is a benchmark result. Record the context length, KV type, GPU configuration, and the last successful context length.
- Record *where* an OOM happens, not just that it happened. On a nearly-full card the abort can come from the flash-attention scratch pool during prefill rather than from KV allocation, and those two have different workarounds. Capture the llama.cpp backtrace; `run_llama_bench_context.sh` only keeps the abort line.
- On WSL hosts, exceeding VRAM does not raise a CUDA OOM. The driver silently places the excess in host RAM, so an over-committed `-ngl` still returns rc=0 while throughput collapses. Pick `-ngl` by measured throughput and wall clock, never by "it ran".
- Discard the first run after transferring a model. llama.cpp mmaps the weights, so a cold page cache is measured as low throughput, not as load time.
- Use unique prompts or `cache_prompt=false` for concurrency testing.
- Treat dual P40 layer split primarily as a max-context / capacity experiment unless measurements show a throughput benefit.
- Keep raw outputs unchanged in `results/raw/`; put derived tables in `results/processed/`.

