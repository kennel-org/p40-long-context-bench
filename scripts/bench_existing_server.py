#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import csv
import json
import time
import urllib.request
from pathlib import Path


def post_json(base, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def synthetic_prompt(target_words, salt):
    unit = (
        "This benchmark prompt is synthetic technical prose about local inference, "
        "GPU memory pressure, context length, prompt processing, and token generation. "
    )
    text = (unit * ((target_words // len(unit.split())) + 2)).split()
    return " ".join(text[:target_words]) + f"\nRequest id: {salt}\n"


def run_context(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    fields = [
        "target_words",
        "wall_s",
        "prompt_tokens",
        "pred_tokens",
        "prefill_tok_s",
        "decode_tok_s",
        "wall_pred_tok_s",
    ]
    start = time.perf_counter()
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, words in enumerate(contexts, 1):
            elapsed = time.perf_counter() - start
            eta = elapsed / max(i - 1, 1) * (len(contexts) - i + 1) if i > 1 else 0
            print(f"[{i}/{len(contexts)}] elapsed={elapsed:.0f}s eta={eta:.0f}s item=context_words_{words}", flush=True)
            prompt = synthetic_prompt(words, f"context-{words}-{time.time_ns()}")
            t0 = time.perf_counter()
            resp = post_json(args.base, {
                "prompt": prompt,
                "n_predict": args.n_predict,
                "temperature": 0,
                "cache_prompt": False,
                "stream": False,
            }, args.timeout)
            wall = time.perf_counter() - t0
            timings = resp.get("timings", {})
            pred = float(timings.get("predicted_n") or resp.get("tokens_predicted") or 0)
            writer.writerow({
                "target_words": words,
                "wall_s": f"{wall:.3f}",
                "prompt_tokens": int(timings.get("prompt_n") or resp.get("tokens_evaluated") or 0),
                "pred_tokens": int(pred),
                "prefill_tok_s": f"{float(timings.get('prompt_per_second') or 0):.2f}",
                "decode_tok_s": f"{float(timings.get('predicted_per_second') or 0):.2f}",
                "wall_pred_tok_s": f"{(pred / wall) if wall > 0 else 0:.2f}",
            })
            f.flush()


def run_concurrency(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    concurrencies = [int(x) for x in args.concurrency.split(",") if x.strip()]
    fields = [
        "concurrency",
        "wall_s",
        "prompt_tokens",
        "pred_tokens",
        "mean_prefill_tok_s",
        "mean_decode_tok_s",
        "aggregate_pred_tok_s_wall",
        "per_stream_pred_tok_s_wall",
    ]
    start = time.perf_counter()
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, c in enumerate(concurrencies, 1):
            elapsed = time.perf_counter() - start
            eta = elapsed / max(i - 1, 1) * (len(concurrencies) - i + 1) if i > 1 else 0
            print(f"[{i}/{len(concurrencies)}] elapsed={elapsed:.0f}s eta={eta:.0f}s item=concurrency_{c}", flush=True)

            def one(j):
                prompt = synthetic_prompt(args.words, f"concurrency-{c}-{j}-{time.time_ns()}")
                return post_json(args.base, {
                    "prompt": prompt,
                    "n_predict": args.n_predict,
                    "temperature": 0,
                    "cache_prompt": False,
                    "stream": False,
                }, args.timeout)

            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(max_workers=c) as ex:
                responses = list(ex.map(one, range(c)))
            wall = time.perf_counter() - t0
            timings = [r.get("timings", {}) for r in responses]
            prompt_tokens = sum(float(t.get("prompt_n") or 0) for t in timings)
            pred_tokens = sum(float(t.get("predicted_n") or r.get("tokens_predicted") or 0) for r, t in zip(responses, timings))
            prefill_rates = [float(t.get("prompt_per_second") or 0) for t in timings if t.get("prompt_per_second")]
            decode_rates = [float(t.get("predicted_per_second") or 0) for t in timings if t.get("predicted_per_second")]
            aggregate = pred_tokens / wall if wall > 0 else 0
            writer.writerow({
                "concurrency": c,
                "wall_s": f"{wall:.3f}",
                "prompt_tokens": int(prompt_tokens),
                "pred_tokens": int(pred_tokens),
                "mean_prefill_tok_s": f"{(sum(prefill_rates) / len(prefill_rates)) if prefill_rates else 0:.2f}",
                "mean_decode_tok_s": f"{(sum(decode_rates) / len(decode_rates)) if decode_rates else 0:.2f}",
                "aggregate_pred_tok_s_wall": f"{aggregate:.2f}",
                "per_stream_pred_tok_s_wall": f"{(aggregate / c) if c else 0:.2f}",
            })
            f.flush()


def main():
    ap = argparse.ArgumentParser(description="Benchmark an already running llama-server via /completion timings")
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--mode", choices=["context", "concurrency"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--contexts", default="512,2048,8192,16384")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--words", type=int, default=512)
    args = ap.parse_args()
    if args.mode == "context":
        run_context(args)
    else:
        run_concurrency(args)


if __name__ == "__main__":
    main()

