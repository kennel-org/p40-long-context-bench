# P40 + llama.cpp 長コンテキスト性能ベンチ 引継ぎメモ

実測済みの GPU1 text-only 再現手順と結果は以下を参照。

- `docs/2026-07-02_reproduction_p40_gpu1_text_only.md`
- `reports/2026-07-02_p40_llamacpp_qwen_longctx.md`

## Metadata

- date: `2026-07-02`
- owner: `<owner>`
- target: `P40 + llama.cpp + Qwen系GGUFモデル`
- comparison target: `Qwen3.6-27B-NVFP4 / RTX PRO 6000 / vLLM nightly / Marlin / 256K context`
- purpose: RTX PRO 6000 / vLLM の絶対性能を再現することではなく、**現在速度チューニング済みの P40 + llama.cpp 構成が、同じ評価軸でどこまで迫れるかを可視化する**。

---

## 1. 推奨リポジトリ名

### 第一候補

`p40-long-context-bench`

理由:

- 目的が明確。
- P40 世代の実力測定であることが名前から分かる。
- llama.cpp 固有に閉じすぎず、将来 vLLM / exllama / TensorRT-LLM などを比較対象に追加しやすい。
- RTX PRO 6000 / Blackwell との比較表を入れても違和感がない。

### 代替候補

| 候補 | ニュアンス |
|---|---|
| `pascal-llm-bench` | P40 = Pascal世代であることを前面に出す。P100等も後で比較しやすい。 |
| `p40-llamacpp-bench` | llama.cpp 固定の検証リポジトリとして分かりやすい。 |
| `longctx-llamacpp-lab` | 長コンテキスト実験全般に広げやすい。 |
| `qwen-p40-bench` | Qwen系モデルに強く寄せる場合。 |
| `domestic-datacenter-bench` | ネタ寄り。家庭内小規模データセンター感を出す名前。 |
| `pascal-vs-blackwell-llm` | RTX PRO 6000画像との比較を強く打ち出す場合。 |

現時点では **`p40-long-context-bench`** を推奨。

---

## 2. 背景

提示された比較対象ベンチは以下の構成。

```text
Qwen3.6-27B-NVFP4
RTX PRO 6000
sm_120
vLLM nightly
Marlin
256K context
```

画像上の代表値:

```text
prefill: 約 5.7k tok/s、ピーク約 6k tok/s
decode: 約 74 tok/s、192K contextでも約 59 tok/s
throughput: 512 input / 128 output, 8 concurrent で aggregate 約 322 tok/s
```

今回の手元構成は以下。

```text
P40 + llama.cpp
Qwen系GGUFモデル
速度チューニング済み llama.cpp
P40接続は OCuLink / USB4 構成を想定
```

重要な前提:

- 比較対象の RTX PRO 6000 は Blackwell / sm_120 / NVFP4 / Marlin / vLLM 系。
- P40 は Pascal 世代で、低精度カーネル・メモリ帯域・世代差が大きい。
- よって **同じ条件の再現ではなく、同じ評価軸での実測比較** とする。
- 見るべき主眼は絶対値ではなく、以下の3点。
  1. prefill がどこまで出るか
  2. decode が context length 増加でどれだけ落ちるか
  3. concurrent request 時の aggregate / per-stream throughput がどこで頭打ちになるか

---

## 3. ベンチの目的

### 主目的

現在の P40 + llama.cpp + Qwen系GGUF 構成について、長コンテキスト性能を以下の形式で可視化する。

1. `Prefill speed vs context length`
2. `Decode speed vs context length`
3. `Throughput vs concurrency`

### 副目的

- OCuLink接続P40とUSB4接続P40の差を確認する。
- dual P40 layer split が速度に効くのか、単にVRAM容量確保にしか効かないのかを確認する。
- KV cache 型 `f16`, `q8_0`, `q4_0` の速度・context上限・品質劣化を確認する。
- 実用上の上限、つまり「このモデルなら何K contextまで実用的か」を決める。

---

## 4. 評価指標

| 指標 | 意味 | 対応する測定方法 |
|---|---|---|
| prefill tok/s | 入力プロンプト処理速度 | `llama-bench` の prompt processing / `pp` |
| decode tok/s | 1 stream の生成速度 | `llama-bench` の text generation / `tg` |
| decode degradation | context長増加に対するdecode低下 | `-d` depth別の `tg` |
| aggregate throughput | 並列時の総生成 tok/s | `llama-server` + concurrent client |
| per-stream throughput | 並列時の1リクエストあたり tok/s | aggregate / concurrency |
| VRAM上限 | OOMしない最大 context | 各depthでの成否 |
| quality sanity | KV量子化時の出力破綻確認 | 固定promptによる簡易目視 |

---

## 5. 想定ディレクトリ構成

```text
p40-long-context-bench/
├── README.md
├── docs/
│   ├── handoff.md
│   ├── benchmark_plan.md
│   └── notes.md
├── scripts/
│   ├── run_llama_bench_context.sh
│   ├── run_llama_server.sh
│   ├── bench_concurrency.py
│   ├── collect_env.sh
│   └── plot_results.py
├── configs/
│   ├── p40_oculink_f16kv.env
│   ├── p40_oculink_q8kv.env
│   ├── p40_usb4_q8kv.env
│   └── p40_dual_layer_q8kv.env
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
└── reports/
    └── 2026-07-02_p40_llamacpp_qwen_longctx.md
```

---

## 6. 実験対象

### 最小構成

まずは現在動いている Qwen系GGUFモデルをそのまま使う。

```bash
MODEL=/path/to/current-qwen-model.gguf
```

モデル名、量子化、ファイルサイズ、llama.cpp の commit hash は必ず記録する。

```bash
./build/bin/llama-cli --version || true
git -C /path/to/llama.cpp rev-parse HEAD
ls -lh "$MODEL"
nvidia-smi
```

### 推奨する測定系列

| 系列 | GPU | KV cache | 目的 |
|---|---|---|---|
| A | OCuLink側 P40 単体 | `f16/f16` | 基準性能 |
| B | OCuLink側 P40 単体 | `q8_0/q8_0` | 長context化 |
| C | OCuLink側 P40 単体 | `q8_0/q4_0` | さらに長context化 |
| D | USB4側 P40 単体 | `q8_0/q8_0` | USB4側の速度確認 |
| E | dual P40 layer split | `q8_0/q8_0` | 容量拡張と速度影響確認 |

---

## 7. Context別 prefill / decode 測定

画像左・中央に相当する測定。

### 基準: OCuLink P40 + f16 KV

```bash
cd /path/to/llama.cpp

MODEL=/path/to/current-qwen-model.gguf
OUT=~/bench_p40_qwen
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-bench \
  -m "$MODEL" \
  -ngl -1 \
  -p 512 \
  -n 128 \
  -d 0,4096,8192,16384,32768,65536,131072,192000 \
  -b 2048 \
  -ub 512 \
  -fa auto \
  -ctk f16 \
  -ctv f16 \
  -r 3 \
  -o csv \
  > "$OUT/p40_oculink_f16kv_context.csv"
```

### 長context: OCuLink P40 + q8 KV

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-bench \
  -m "$MODEL" \
  -ngl -1 \
  -p 512 \
  -n 128 \
  -d 0,4096,8192,16384,32768,65536,131072,192000 \
  -b 2048 \
  -ub 512 \
  -fa auto \
  -ctk q8_0 \
  -ctv q8_0 \
  -r 3 \
  -o csv \
  > "$OUT/p40_oculink_q8kv_context.csv"
```

### さらに長context: OCuLink P40 + q8 K / q4 V

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-bench \
  -m "$MODEL" \
  -ngl -1 \
  -p 512 \
  -n 128 \
  -d 0,4096,8192,16384,32768,65536,131072,192000 \
  -b 2048 \
  -ub 512 \
  -fa auto \
  -ctk q8_0 \
  -ctv q4_0 \
  -r 3 \
  -o csv \
  > "$OUT/p40_oculink_q8k_q4v_context.csv"
```

### USB4側 P40 単体

```bash
CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-bench \
  -m "$MODEL" \
  -ngl -1 \
  -p 512 \
  -n 128 \
  -d 0,4096,8192,16384,32768,65536 \
  -b 2048 \
  -ub 512 \
  -fa auto \
  -ctk q8_0 \
  -ctv q8_0 \
  -r 3 \
  -o csv \
  > "$OUT/p40_usb4_q8kv_context.csv"
```

### dual P40 layer split

```bash
CUDA_VISIBLE_DEVICES=0,1 ./build/bin/llama-bench \
  -m "$MODEL" \
  -ngl -1 \
  -sm layer \
  -ts 1/1 \
  -p 512 \
  -n 128 \
  -d 0,4096,8192,16384,32768,65536,131072 \
  -b 2048 \
  -ub 512 \
  -fa auto \
  -ctk q8_0 \
  -ctv q8_0 \
  -r 3 \
  -o csv \
  > "$OUT/p40_dual_layer_q8kv_context.csv"
```

注意:

- `131072` や `192000` で OOM する可能性は高い。
- OOMは失敗ではなく、測定結果として記録する。
- P40単体で無理なcontextは、dual P40やKV量子化で再挑戦する。

---

## 8. 並列 throughput 測定

画像右に相当する測定。

条件をなるべく合わせる。

```text
input: 約512 tokens
output: 128 tokens
concurrency: 1, 2, 4, 8, 12, 16
server parallel slots: 8
```

### llama-server 起動例

```bash
MODEL=/path/to/current-qwen-model.gguf

CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port 8080 \
  -ngl -1 \
  -c 32768 \
  --parallel 8 \
  -b 2048 \
  -ub 512 \
  -fa auto \
  -ctk q8_0 \
  -ctv q8_0 \
  --metrics
```

### concurrent client

`scripts/bench_concurrency.py` として保存する。

```python
#!/usr/bin/env python3
import concurrent.futures as cf
import json
import re
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
CONCURRENCIES = [1, 2, 4, 8, 12, 16]

BASE_PROMPT = (
    "You are benchmarking local inference. "
    "Summarize the following synthetic technical note in one paragraph. "
    + ("The quick brown fox jumps over the lazy dog. " * 90)
)

def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))

def get_metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
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

def one_request(i):
    prompt = BASE_PROMPT + f"\nRequest id: {i}\n"
    return post_json("/completion", {
        "prompt": prompt,
        "n_predict": 128,
        "temperature": 0,
        "cache_prompt": False,
        "stream": False,
    })

print("concurrency,wall_s,prompt_tokens,pred_tokens,prefill_tok_s_internal,decode_tok_s_internal,aggregate_pred_tok_s_wall,per_stream_pred_tok_s_wall")

for c in CONCURRENCIES:
    time.sleep(3)
    before = get_metrics()
    t0 = time.time()

    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        list(ex.map(one_request, range(c)))

    wall = time.time() - t0
    after = get_metrics()

    dprompt = after["prompt_tokens"] - before["prompt_tokens"]
    dpred = after["pred_tokens"] - before["pred_tokens"]
    dprompt_s = after["prompt_seconds"] - before["prompt_seconds"]
    dpred_s = after["pred_seconds"] - before["pred_seconds"]

    prefill_internal = dprompt / dprompt_s if dprompt_s > 0 else 0
    decode_internal = dpred / dpred_s if dpred_s > 0 else 0
    aggregate_wall = dpred / wall if wall > 0 else 0
    per_stream_wall = aggregate_wall / c if c > 0 else 0

    print(f"{c},{wall:.3f},{dprompt:.0f},{dpred:.0f},{prefill_internal:.2f},{decode_internal:.2f},{aggregate_wall:.2f},{per_stream_wall:.2f}")
```

実行例:

```bash
python3 scripts/bench_concurrency.py \
  | tee results/raw/p40_oculink_q8kv_concurrency.csv
```

---

## 9. 比較表フォーマット

最終レポートでは以下の表を作る。

| 指標 | RTX PRO 6000画像 | P40実測 | 比率 |
|---|---:|---:|---:|
| prefill peak | 5700〜6000 tok/s | TBD | TBD |
| decode short context | 74 tok/s | TBD | TBD |
| decode @ 32K | TBD | TBD | TBD |
| decode @ 64K | TBD | TBD | TBD |
| decode @ 128K | TBD | TBD | TBD |
| decode @ 192K | 59 tok/s | TBD | TBD |
| aggregate @ 8 concurrent | 322 tok/s | TBD | TBD |
| per-stream @ 8 concurrent | 約40 tok/s | TBD | TBD |
| max practical context | 256K | TBD | TBD |

補足:

- RTX PRO 6000側の値は画像から読み取った値なので、厳密なソース値ではなく参考値として扱う。
- P40側はCSVから集計した実測値を入れる。
- `TBD` は測定後に埋める。

---

## 10. グラフ作成方針

元画像に合わせて、以下の3グラフを作る。

1. `Prefill speed vs context`
   - x軸: context length / prompt tokens
   - y軸: prefill throughput tok/s
   - 系列: f16KV, q8KV, q8K/q4V, dual P40 など

2. `Decode speed vs context`
   - x軸: context length / prompt tokens
   - y軸: decode throughput tok/s
   - 長contextでの落ち方を見る。

3. `Throughput vs concurrency`
   - x軸: concurrent requests
   - y軸: throughput tok/s
   - aggregate と per-stream の2系列。
   - `--parallel 8` の線を入れる。

---

## 11. collect_env.sh 案

`scripts/collect_env.sh` として保存する。

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-results/raw/env_$(date +%Y%m%d_%H%M%S).txt}
mkdir -p "$(dirname "$OUT")"

{
  echo "# date"
  date -Is
  echo

  echo "# hostname"
  hostnamectl || hostname
  echo

  echo "# uname"
  uname -a
  echo

  echo "# nvidia-smi"
  nvidia-smi || true
  echo

  echo "# CUDA_VISIBLE_DEVICES"
  echo "${CUDA_VISIBLE_DEVICES:-unset}"
  echo

  echo "# llama.cpp git"
  git rev-parse HEAD || true
  git status --short || true
  echo

  echo "# llama binaries"
  ./build/bin/llama-cli --version || true
  ./build/bin/llama-bench --help | head -40 || true
  echo

  echo "# model"
  echo "MODEL=${MODEL:-unset}"
  if [ -n "${MODEL:-}" ] && [ -f "$MODEL" ]; then
    ls -lh "$MODEL"
    sha256sum "$MODEL" | cut -c1-32
  fi
} | tee "$OUT"
```

---

## 12. 実施順序

### Phase 1: 最小測定

1. 現在動いているモデル名・量子化・llama.cpp commitを記録。
2. OCuLink側 P40単体で `8K / 16K / 32K` を測る。
3. `f16KV` と `q8KV` を比較。
4. まずはOOMしない範囲でCSVを確保。

### Phase 2: 長context挑戦

1. `64K / 128K / 192K` を追加。
2. OOMした点を記録。
3. `q8_0/q4_0` KVで再挑戦。
4. dual P40 layer split で最大contextを確認。

### Phase 3: 並列throughput

1. `-c 32768 --parallel 8` で server 起動。
2. 1,2,4,8,12,16 concurrent を測る。
3. aggregate / per-stream を算出。
4. 必要に応じて `--parallel 4`, `--parallel 16` も比較。

### Phase 4: レポート化

1. CSVを `results/raw/` に保存。
2. 集計値を `results/processed/` に保存。
3. グラフを `results/figures/` に保存。
4. `reports/YYYY-MM-DD_p40_llamacpp_qwen_longctx.md` に結論を書く。

---

## 13. 注意点・落とし穴

### prompt cache

`llama-server` の並列測定で同一promptを使うと、prompt cacheにより prefill が不自然に速くなる可能性がある。

対策:

```json
"cache_prompt": false
```

または requestごとにprompt末尾を変える。

### OOMの扱い

OOMは失敗ではなく測定結果。

記録すべき内容:

- context length
- KV cache型
- GPU構成
- エラーメッセージ
- その直前に成功した最大context

### USB4側P40

USB4側は、単体測定とdual測定で意味が違う。

- 単体測定: USB4接続P40そのものの実力確認。
- dual測定: VRAM容量拡張と通信オーバーヘッドのトレードオフ確認。

速度目的ならOCuLink側単体が勝つ可能性がある。dual P40は長contextや大きいモデルを載せるための手段として評価する。

### layer split

`-sm layer` はGPU間通信が少ないため、P40×2ではまずこれを試す。
Tensor split / row split 系は帯域が苦しい接続では不利になる可能性がある。

### 比較対象との違い

RTX PRO 6000画像側は NVFP4 / Marlin / vLLM。こちらは GGUF / llama.cpp。したがって、数値差は以下が混ざる。

- GPU世代差
- VRAM帯域差
- 量子化方式差
- カーネル差
- サービング実装差
- モデル実体差
- context / KV cache 実装差

よって最終レポートでは、**単純な勝ち負けではなく、実用境界の可視化**として書く。

---

## 14. 完了条件

最低限の完了条件:

- [ ] llama.cpp commit hashを記録した。
- [ ] モデル名・GGUF量子化・ファイルサイズを記録した。
- [ ] OCuLink P40単体で context別 prefill/decode CSVを取得した。
- [ ] 少なくとも `f16KV` と `q8KV` を比較した。
- [ ] 並列 1/2/4/8/12/16 の throughput CSVを取得した。
- [ ] 元画像と同じ3グラフを作成した。
- [ ] RTX PRO 6000画像との比較表を作成した。
- [ ] 実用上の推奨設定を1つ以上提示した。

望ましい完了条件:

- [ ] USB4側P40単体の測定を追加した。
- [ ] dual P40 layer split の測定を追加した。
- [ ] `q8_0/q4_0` KVの品質sanity checkを行った。
- [ ] OOM上限を一覧化した。
- [ ] `--parallel` 値を 4/8/16 で比較した。

---

## 15. 最終的に欲しい結論の形

最終レポートでは、以下のような結論を出す。

```text
このQwen系GGUFモデルでは、P40単体OCuLink構成において、
実用contextは XXK まで。

prefill peak は RTX PRO 6000画像比で約 X%。
decode short context は約 X%。
decode long context では XXK時点で約 X tok/s まで低下。

並列では --parallel 8 のとき aggregate が X tok/s で頭打ち。
per-stream は concurrency 8 で X tok/s。

KV cache は q8_0/q8_0 が速度・VRAM・品質のバランスが最もよい。
dual P40 は速度改善よりも最大context拡張に意味がある。
```

---

## 16. README冒頭案

```markdown
# p40-long-context-bench

Benchmarking long-context LLM inference on Tesla P40 + llama.cpp.

This repository measures how far a tuned Pascal-generation P40 setup can go against modern long-context serving results such as Qwen3.6-27B-NVFP4 on RTX PRO 6000 / vLLM / Marlin.

The goal is not to reproduce Blackwell-class FP4 performance, but to quantify practical limits of a local P40-based setup:

- prefill throughput vs context length
- decode throughput vs context length
- aggregate throughput vs concurrency
- KV cache quantization trade-offs
- OCuLink / USB4 / dual-GPU behavior
```
