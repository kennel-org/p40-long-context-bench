#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import re
import time
import urllib.request
from pathlib import Path


BASE_PROMPT = (
    "You are benchmarking local inference. "
    "Summarize the following synthetic technical note in one paragraph. "
    + ("The quick brown fox jumps over the lazy dog. " * 90)
)


def post_json(base, path, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
        text = r.read().decode("utf-8")

    def metric(name):
        pat = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", re.M)
        m = pat.search(text)
        return float(m.group(1)) if m else 0.0

    return {
        "prompt_tokens": metric("llamacpp:prompt_tokens_total"),
        "prompt_seconds": metric("llamacpp:prompt_seconds_total"),
        "pred_tokens": metric("llamacpp:tokens_predicted_total"),
        "pred_seconds": metric("llamacpp:tokens_predicted_seconds_total"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--concurrency", default="1,2,4,8,12,16")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--out", default="results/raw/p40_oculink_q8kv_concurrency.csv")
    ap.add_argument("--checkpoint", default="results/processed/concurrency_checkpoint.json")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    concurrencies = [int(x) for x in args.concurrency.split(",") if x.strip()]
    out = Path(args.out)
    checkpoint = Path(args.checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "concurrency",
        "wall_s",
        "prompt_tokens",
        "pred_tokens",
        "prefill_tok_s_internal",
        "decode_tok_s_internal",
        "aggregate_pred_tok_s_wall",
        "per_stream_pred_tok_s_wall",
    ]

    start = time.perf_counter()
    rows = []
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, c in enumerate(concurrencies, 1):
            elapsed = time.perf_counter() - start
            eta = elapsed / max(idx - 1, 1) * (len(concurrencies) - idx + 1) if idx > 1 else 0
            print(f"[{idx}/{len(concurrencies)}] elapsed={elapsed:.0f}s eta={eta:.0f}s item=concurrency_{c}", flush=True)
            time.sleep(3)
            before = get_metrics(args.base)
            t0 = time.perf_counter()

            def one_request(i):
                prompt = BASE_PROMPT + f"\nRequest id: {c}-{i}-{time.time_ns()}\n"
                return post_json(args.base, "/completion", {
                    "prompt": prompt,
                    "n_predict": args.n_predict,
                    "temperature": 0,
                    "cache_prompt": False,
                    "stream": False,
                }, args.timeout)

            with cf.ThreadPoolExecutor(max_workers=c) as ex:
                list(ex.map(one_request, range(c)))

            wall = time.perf_counter() - t0
            after = get_metrics(args.base)
            dprompt = after["prompt_tokens"] - before["prompt_tokens"]
            dpred = after["pred_tokens"] - before["pred_tokens"]
            dprompt_s = after["prompt_seconds"] - before["prompt_seconds"]
            dpred_s = after["pred_seconds"] - before["pred_seconds"]
            row = {
                "concurrency": c,
                "wall_s": f"{wall:.3f}",
                "prompt_tokens": f"{dprompt:.0f}",
                "pred_tokens": f"{dpred:.0f}",
                "prefill_tok_s_internal": f"{(dprompt / dprompt_s) if dprompt_s > 0 else 0:.2f}",
                "decode_tok_s_internal": f"{(dpred / dpred_s) if dpred_s > 0 else 0:.2f}",
                "aggregate_pred_tok_s_wall": f"{(dpred / wall) if wall > 0 else 0:.2f}",
                "per_stream_pred_tok_s_wall": f"{(dpred / wall / c) if wall > 0 and c > 0 else 0:.2f}",
            }
            writer.writerow(row)
            f.flush()
            rows.append(row)
            elapsed_i = int(time.perf_counter() - start)
            checkpoint.write_text(json.dumps({
                "completed_index": idx,
                "total": len(concurrencies),
                "latest_item": c,
                "elapsed_seconds": elapsed_i,
                "wall_clock": str(dt.timedelta(seconds=elapsed_i)),
                "results": rows,
            }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

