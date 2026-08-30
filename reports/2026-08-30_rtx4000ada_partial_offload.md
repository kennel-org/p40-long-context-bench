# RTX 4000 Ada: Partial-Offload Context Sweep

- Date: 2026-08-30
- Scope: add a second GPU per the README "Adding Another GPU" flow, measured independently of the P40
- Host: `pr36-wsl` (DELL Precision 3680, WSL2 `6.6.87.2-microsoft-standard-WSL2`, i7-14700, 78 GB RAM)
- GPU: NVIDIA RTX 4000 Ada Generation, 20475 MiB, compute capability 8.9
- Model: `Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf`
- Model SHA256: `ae2fb73ac0da875640269f1e65e9c7fb415b066c6d544c3eef9adb0d03f04792` — **verified identical to the x1ai copy**
- llama.cpp commit: `bbce619adb409880fb6db850a1c5a5f36a4dc7b1`, build `9283 (bbce619ad)` — **built to match x1ai exactly**
- Config: `configs/rtx4000ada_q8kv_ngl39.env`

## Headline

The RTX 4000 Ada cannot hold this model fully: weights are 20176 MiB against 20474 MiB
visible VRAM. At `-ngl 39` (2 layers on CPU) everything else fits and the card is
**1.8-2.5x the P40 on prefill and about 1.5x on decode**. Adding even one more layer
collapses it.

Decode lands surprisingly close to the RTX PRO 6000 reference — 82-96% of it — while
prefill is only 19-36% of the reference.

## The `-ngl` Cliff

Measured at depth 32768, `-b 2048 -ub 512 -fa 1`, `q8_0/q8_0`, `r=3`, warm page cache:

| `-ngl` | prefill tok/s | decode tok/s | run wall clock | state |
|---:|---:|---:|---:|---|
| **39** | **1390.95** | **58.30** | 34 s | fits in VRAM |
| 40 | 128.11 | 12.05 | 303 s | spilled |
| 41 (full) | 129.49 | 26.82 | 283 s | spilled |

One extra layer costs **10.9x prefill** and **4.8x decode**. `-ngl 39` is the operating
point, and it was chosen by measurement, not by arithmetic.

### WSL does not OOM — it spills

This is the part that makes the cliff dangerous. On a normal Linux host, exceeding VRAM
aborts with a CUDA out-of-memory error. Under WSL the NVIDIA driver instead places the
excess in host RAM as "shared GPU memory" and keeps running. Every access to that
portion then crosses PCIe.

So `-ngl 41` **returns rc=0 and produces plausible-looking numbers**. Nothing in the
llama.cpp output says anything is wrong. The only signals are the throughput collapse and
the wall clock (283 s versus 34 s). Any benchmark on this host has to verify the chosen
`-ngl` from throughput, because a silent success proves nothing.

## Context Sweep (`-ngl 39`)

`PROMPT_TOKENS=512`, `GEN_TOKENS=128`, `-b 2048 -ub 512 -fa 1`, `q8_0/q8_0`, `r=3`:

| context | prefill tok/s | decode tok/s | status |
|---:|---:|---:|---|
| 512 | 1716.21 | 71.14 | ok |
| 8192 | 1619.02 | 68.26 | ok |
| 16384 | — | — | **abort (rc=134)** |
| 32768 | 1069.95 | 57.87 | ok |

### The depth-16384 abort

Depth 16384 aborts at `-ub 512` while both 8192 and 32768 succeed. This is not noise: it
reproduced **three times out of three**, and it fails in the same allocator as the P40's
192K ceiling — `ggml_cuda_pool_vmm::alloc` inside `launch_fattn`, reached from
`ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 8>` during prefill.

Note the kernel variant differs from the P40's (`_mma_f16_case<256,256,8,8>` here versus
`_tile_case<256,256>` on the P40, and `launch_fattn<256,8,8>` versus `<256,4,8>`).
llama.cpp selects a flash-attention kernel per shape and compute capability, and the
variant chosen at this depth asks for a larger scratch block than the one chosen at
32768. That is why the failure is non-monotonic in context length.

**This was fixed upstream.** On llama.cpp build 10703 the same depth completes at
`-ub 512` — see "Update: llama.cpp build 10703" below. The `-ub` workaround below applies
to build 9283 only.

Lowering the micro-batch avoids it:

| depth | `-ub` | prefill tok/s | decode tok/s | status |
|---:|---:|---:|---:|---|
| 16384 | 512 | — | — | abort |
| 16384 | 256 | 1055.77 | 64.36 | ok |
| 16384 | 128 | 676.97 | 64.72 | ok |

Decode is essentially unaffected by `-ub` (64.36 vs 64.72); prefill drops 36% going from
256 to 128. `-ub 256` is the cheapest way to close the hole, at the cost of no longer
matching the P40's benchmark shape at that one depth.

## Comparison

### vs Tesla P40 (same model, same llama.cpp build, same KV type)

| context | P40 prefill | Ada prefill | ratio | P40 decode | Ada decode | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 964.54 | 1716.21 | **1.78x** | 50.05 | 71.14 | **1.42x** |
| 8192 | 729.59 | 1619.02 | **2.22x** | 45.38 | 68.26 | **1.50x** |
| 16384 | 589.16 | 1055.77 (`-ub 256`) | 1.79x | 42.80 | 64.36 | 1.50x |
| 32768 | 427.15 | 1069.95 | **2.50x** | 37.52 | 57.87 | **1.54x** |

The P40 runs fully offloaded (`-ngl 41`), the Ada with 2 layers on CPU, so this is not a
pure silicon comparison — it is what each card actually delivers on this model.

The Ada's advantage widens with context on prefill (1.78x → 2.50x) and stays flat on
decode (~1.5x). The P40 also reaches far longer contexts: 163840 versus 32768 tested
here, purely because it has 2.4 GB more VRAM to spend on KV after weights.

### vs RTX PRO 6000 reference

| context | reference prefill | Ada | share | reference decode | Ada | share |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 4770 | 1716.21 | 36.0% | 74.1 | 71.14 | **96.0%** |
| 8192 | — | 1619.02 | n/a | 73.6 | 68.26 | **92.7%** |
| 32768 | 5704 | 1069.95 | 18.8% | 70.7 | 57.87 | **81.9%** |

Decode comes within a few percent of the reference at short context. This is a 3B-active
MoE, so decode moves far less weight per token than the parameter count suggests, which
compresses the gap between cards. Prefill, which is compute-bound, does not compress.

## Update: llama.cpp build 10703 (same day)

`pr36` was subsequently moved to the current llama.cpp master to verify the latest
version on that host:

| | before | after |
|---|---|---|
| commit | `bbce619ad` | `0b5be7e4a` |
| build | 9283 | 10703 |
| version | — | 0.3.0-dev |

Re-measured at the identical shape (`-ngl 39`, `-b 2048 -ub 512 -fa 1`, `q8_0/q8_0`, `r=3`):

| context | prefill tok/s | stddev | decode tok/s | stddev | build 9283 |
|---:|---:|---:|---:|---:|---|
| 8192 | 1574.29 | 51.37 | 69.04 | 0.78 | 1619.02 / 68.26 |
| 16384 | 1540.64 | 38.04 | 65.13 | 0.85 | **abort** |
| 32768 | 1423.85 | 59.15 | 59.44 | 0.56 | 1069.95 / 57.87 |

**Depth 16384 now completes at `-ub 512`**, where build 9283 aborted three times out of
three. The flash-attention scratch allocation described above is no longer reached, so
the `-ub 256` workaround is not needed on this build.

The throughput differences are *not* attributable to the version bump. 32768 prefill
reads 1423.85 against 1069.95, but build 9283 itself produced 1070-1391 across three runs
of the same configuration on this interactively-used host, so the new figure sits inside
the old spread. 8192 went slightly *down* (1619.02 → 1574.29), also within that spread.
Decode rose a little at all three depths and is the more stable of the two axes. Nothing
here separates a real version effect from host noise; only the 16384 fix is a firm
conclusion.

### Rebuild note

The incremental rebuild across 1420 commits failed once, in `scripts/ui-assets.cmake:291`
while embedding the server web UI. The build directory still held the old asset set
(`bundle.css`, `bundle.js`, `index.html`, `loading.html`, `checksums.txt`) while the new
tree additionally requires `manifest.webmanifest`, `sw.js`, `build.json`, `version.json`,
and `workbox[hash].js`. Deleting the generated `build/tools/ui/dist` directory and
rebuilding resolved it. Only generated output was removed; `build/` is gitignored.

### Reproducing the build 9283 numbers

`pr36`'s `~/src/llama.cpp` is now on `master`. The earlier measurements need:

```bash
cd ~/src/llama.cpp && git checkout bbce619adb409880fb6db850a1c5a5f36a4dc7b1
```

The objects are already local, so no fetch is required. x1ai's `~/src/llama.cpp` was not
touched and remains at build 9283, so the P40 series is unaffected.

## Caveats

- **Prefill numbers vary run to run.** Three separate `r=3` runs at `-ngl 39` / depth
  32768 gave 1286.19, 1390.95, and 1069.95 tok/s — a spread of ±13%. `pr36` is kennel's
  interactive machine and had ~1 GB of VRAM in use by other processes throughout. Decode
  was stable across the same runs (57.87-58.59). Treat prefill here as ±15%, not as a
  precise figure.
- **An early `-ngl` probe produced invalid numbers.** The first probe ran immediately
  after the 21 GB download, with a cold page cache. Because llama.cpp mmaps the weights,
  the CPU-resident layers were paging in from disk during the measurement and prefill
  read as 94 tok/s at depth 32768 — 15x below the warm value. Those numbers are discarded.
  Any first run after a model transfer on this host must be treated as a warm-up.
- `n_threads` defaulted to 14 (physical cores), not 28.
- Text-only, no `--mmproj`, matching the P40 runs.
- No concurrency sweep was run on this host.
- 64K and beyond were not attempted here. With 2 layers already on CPU at 32K there is
  little KV headroom left.

## Wall Clock

| Step | Wall clock |
|---|---:|
| llama.cpp clone + fetch to `bbce619ad` | ~4 min |
| CUDA build (`-j 28`, arch 89) | ~13 min |
| model download from HuggingFace (21.2 GB, ~6.9 MB/s) | ~56 min |
| SHA256 verification | 46 s |
| `-ngl` probes + final sweep | ~25 min |

Downloading from HuggingFace directly to `pr36` rather than relaying the file from x1ai
was worth it: the x1ai → pr36 Tailscale path measured 3.45 MB/s (about 1.7 h for this
file) against 6.9 MB/s from HuggingFace.

## Reproduction

On `pr36-wsl`:

```bash
cd ~/bench
scripts/run_llama_bench_context.sh configs/rtx4000ada_q8kv_ngl39.env
```

The build was produced with:

```bash
git clone ~/projects/llama.cpp ~/src/llama.cpp
cd ~/src/llama.cpp
git fetch origin bbce619adb409880fb6db850a1c5a5f36a4dc7b1
git checkout bbce619adb409880fb6db850a1c5a5f36a4dc7b1
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build --target llama-bench llama-server -j 28
```

The existing `~/projects/llama.cpp` working tree on that host was left untouched.

## Artifacts

| Artifact | Path |
|---|---|
| context sweep CSV | `results/raw/rtx4000ada_q8kv_ngl39_context.csv` |
| depth-16384 ubatch variants | `results/raw/rtx4000ada_q8kv_ngl39_d16384_ubatch.csv` |
| build 10703 verification | `results/raw/rtx4000ada_q8kv_ngl39_build10703_verify.csv` |
| config | `configs/rtx4000ada_q8kv_ngl39.env` |
| plot | `results/figures/p40_long_context_summary.png` |
| P40 baseline | `reports/2026-07-02_p40_llamacpp_qwen_longctx.md` |
| P40 long-context extension | `reports/2026-08-30_p40_phase2_long_context.md` |

## Follow-ups

- Re-measure prefill when `pr36` is idle, to replace the ±13% spread with a clean figure.
- The `-ub 256` question is closed: build 10703 covers 16384 at `-ub 512`. If the P40
  series is ever rebuilt on a current llama.cpp, re-test its 192K ceiling too — that
  failure shares the same allocator.
- A concurrency sweep on this host would complete the third panel for the Ada.
