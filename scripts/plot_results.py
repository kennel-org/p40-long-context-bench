#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


REFERENCE_PREFILL = [(512, 4770), (16384, 5794), (32768, 5704), (65536, 5227), (131072, 4318), (192000, 3216)]
REFERENCE_DECODE = [(512, 74.1), (8192, 73.6), (32768, 70.7), (65536, 68.5), (131072, 65.0), (192000, 58.6)]
REFERENCE_CONCURRENCY = [(1, 57, 74), (2, 113, 73), (4, 202, 72), (8, 322, 68), (12, 298, 58), (16, 333, 67)]


def find_col(df, candidates):
    lowered = {c.lower(): c for c in df.columns}
    for key in candidates:
        for low, original in lowered.items():
            if key in low:
                return original
    return None


def numeric_col(df, candidates):
    col = find_col(df, candidates)
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def normalize_context_csv(path):
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()

    test_col = "test" if "test" in df.columns else None
    speed_col = find_col(df, ["avg_ts", "tok/s", "t/s", "speed"])
    depth = numeric_col(df, ["n_depth", "depth"])
    prompt = numeric_col(df, ["n_prompt", "prompt"])
    n_gen = numeric_col(df, ["n_gen"])
    n_prompt = numeric_col(df, ["n_prompt"])
    if speed_col is None or depth is None:
        return pd.DataFrame()

    if prompt is None:
        prompt = n_prompt if n_prompt is not None else pd.Series([512] * len(df), index=df.index)
    prompt_for_x = prompt.mask(prompt <= 0, 512)
    x = depth.mask(depth <= 0, prompt_for_x)
    out = pd.DataFrame({
        "series": path.name.removesuffix("_context.csv"),
        "context": x,
        "speed": pd.to_numeric(df[speed_col], errors="coerce"),
        "kind": "",
    })

    if test_col is not None:
        tests = df[test_col].astype(str).str.lower()
        out.loc[tests.str.contains("pp|prompt"), "kind"] = "prefill"
        out.loc[tests.str.contains("tg|gen|generation"), "kind"] = "decode"
    elif n_prompt is not None and n_gen is not None:
        out.loc[(n_prompt > 0) & (n_gen <= 0), "kind"] = "prefill"
        out.loc[(n_prompt <= 0) & (n_gen > 0), "kind"] = "decode"
    else:
        pp = numeric_col(df, ["pp", "prompt"])
        tg = numeric_col(df, ["tg", "generation"])
        frames = []
        if pp is not None:
            frames.append(pd.DataFrame({"series": out["series"], "context": x, "speed": pp, "kind": "prefill"}))
        if tg is not None:
            frames.append(pd.DataFrame({"series": out["series"], "context": x, "speed": tg, "kind": "decode"}))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return out[out["kind"].isin(["prefill", "decode"])].dropna(subset=["context", "speed"])


def annotate_points(ax, xs, ys, fmt="{:.0f}"):
    for x, y in zip(xs, ys):
        if pd.notna(x) and pd.notna(y):
            ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)


def style_context_axis(ax, ylabel):
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("context length (prompt tokens)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=8)


def plot_reference_like(fig_dir, title, context_data, concurrency_data, include_reference):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    if include_reference:
        xs, ys = zip(*REFERENCE_PREFILL)
        ax.plot(xs, ys, marker="o", color="#2f6df6", label="RTX PRO 6000 ref")
        annotate_points(ax, xs, ys)
    for name, part in context_data[context_data["kind"] == "prefill"].groupby("series"):
        part = part.sort_values("context")
        ax.plot(part["context"], part["speed"], marker="o", linewidth=1.8, label=name)
        annotate_points(ax, part["context"], part["speed"])
    ax.set_title("Prefill speed vs context", fontsize=10)
    style_context_axis(ax, "prefill throughput (tok/s)")
    if ax.lines:
        ax.legend(fontsize=7)

    ax = axes[1]
    if include_reference:
        xs, ys = zip(*REFERENCE_DECODE)
        ax.plot(xs, ys, marker="o", color="#f05a24", label="RTX PRO 6000 ref")
        annotate_points(ax, xs, ys, "{:.1f}")
    for name, part in context_data[context_data["kind"] == "decode"].groupby("series"):
        part = part.sort_values("context")
        ax.plot(part["context"], part["speed"], marker="o", linewidth=1.8, label=name)
        annotate_points(ax, part["context"], part["speed"], "{:.1f}")
    ax.set_title("Decode speed vs context", fontsize=10)
    style_context_axis(ax, "decode throughput (tok/s)")
    if ax.lines:
        ax.legend(fontsize=7)

    ax = axes[2]
    if include_reference:
        ref = pd.DataFrame(REFERENCE_CONCURRENCY, columns=["concurrency", "aggregate", "per_stream"])
        ax.plot(ref["concurrency"], ref["aggregate"], marker="o", color="#0c9b6c", label="aggregate ref")
        ax.plot(ref["concurrency"], ref["per_stream"], marker="s", color="#2f6df6", label="per-stream ref")
        annotate_points(ax, ref["concurrency"], ref["aggregate"])
    if not concurrency_data.empty:
        for name, part in concurrency_data.groupby("series"):
            part = part.sort_values("concurrency")
            ax.plot(part["concurrency"], part["aggregate"], marker="o", linewidth=1.8, label=f"{name} aggregate")
            ax.plot(part["concurrency"], part["per_stream"], marker="s", linewidth=1.8, label=f"{name} per-stream")
            annotate_points(ax, part["concurrency"], part["aggregate"])
    ax.axvline(8, color="0.6", linestyle="--", linewidth=0.8)
    ax.text(8.15, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.07, "max_num_seqs=8", color="0.5", fontsize=7)
    ax.set_title("Throughput vs concurrency   (512in / 128out)", fontsize=10)
    ax.set_xlabel("concurrent requests", fontsize=8)
    ax.set_ylabel("throughput (tok/s)", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=8)
    if ax.lines:
        ax.legend(fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.5)
    out = fig_dir / "p40_long_context_summary.png"
    fig.savefig(out, dpi=180)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw")
    ap.add_argument("--fig-dir", default="results/figures")
    ap.add_argument("--title", default="P40 + llama.cpp long-context benchmark")
    ap.add_argument("--include-reference", action="store_true", help="overlay the RTX PRO 6000 values from the handoff image")
    ap.add_argument("--reference-only", action="store_true", help="draw only the reference image values")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    raw_dir = Path(args.raw_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    context_frames = [normalize_context_csv(path) for path in raw_dir.glob("*_context.csv")]
    context_frames = [df for df in context_frames if not df.empty]
    context_data = pd.concat(context_frames, ignore_index=True) if context_frames else pd.DataFrame(
        columns=["series", "context", "speed", "kind"]
    )

    concurrency_frames = []
    for path in raw_dir.glob("*_concurrency.csv"):
        df = pd.read_csv(path)
        if {"concurrency", "aggregate_pred_tok_s_wall", "per_stream_pred_tok_s_wall"}.issubset(df.columns):
            concurrency_frames.append(pd.DataFrame({
                "series": path.name.removesuffix("_concurrency.csv"),
                "concurrency": pd.to_numeric(df["concurrency"], errors="coerce"),
                "aggregate": pd.to_numeric(df["aggregate_pred_tok_s_wall"], errors="coerce"),
                "per_stream": pd.to_numeric(df["per_stream_pred_tok_s_wall"], errors="coerce"),
            }))
    concurrency_data = pd.concat(concurrency_frames, ignore_index=True) if concurrency_frames else pd.DataFrame(
        columns=["series", "concurrency", "aggregate", "per_stream"]
    )

    if args.reference_only:
        context_data = pd.DataFrame(columns=["series", "context", "speed", "kind"])
        concurrency_data = pd.DataFrame(columns=["series", "concurrency", "aggregate", "per_stream"])
        args.include_reference = True

    out = plot_reference_like(fig_dir, args.title, context_data, concurrency_data, args.include_reference)
    print(out)


if __name__ == "__main__":
    main()
