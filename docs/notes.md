# Notes

- OOM is a benchmark result. Record the context length, KV type, GPU configuration, and the last successful context length.
- Use unique prompts or `cache_prompt=false` for concurrency testing.
- Treat dual P40 layer split primarily as a max-context / capacity experiment unless measurements show a throughput benefit.
- Keep raw outputs unchanged in `results/raw/`; put derived tables in `results/processed/`.

