#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: scripts/run_llama_bench_context.sh configs/<name>.env}
set -a
# shellcheck disable=SC1090
. "$CONFIG"
set +a

LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}
LABEL=${LABEL:-$(basename "$CONFIG" .env)}
MODEL=${MODEL:?MODEL is required}
CONTEXT_DEPTHS=${CONTEXT_DEPTHS:-0,8192,16384,32768}
PROMPT_TOKENS=${PROMPT_TOKENS:-512}
GEN_TOKENS=${GEN_TOKENS:-128}
N_GPU_LAYERS=${N_GPU_LAYERS:-41}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-512}
CACHE_TYPE_K=${CACHE_TYPE_K:-q8_0}
CACHE_TYPE_V=${CACHE_TYPE_V:-q8_0}
FLASH_ATTN=${FLASH_ATTN:-0}
REPETITIONS=${REPETITIONS:-3}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p results/raw results/processed
OUT="results/raw/${LABEL}_context.csv"
LOG="results/raw/${LABEL}_context.log"
CHECKPOINT="results/processed/${LABEL}_context_checkpoint.json"

IFS=',' read -r -a DEPTHS <<< "$CONTEXT_DEPTHS"
total=${#DEPTHS[@]}
start=$(date +%s)
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

: > "$OUT"
: > "$LOG"

for i in "${!DEPTHS[@]}"; do
  depth=${DEPTHS[$i]}
  item_index=$((i + 1))
  elapsed=$(( $(date +%s) - start ))
  if [ "$item_index" -gt 1 ]; then
    eta=$(( elapsed * (total - item_index + 1) / (item_index - 1) ))
  else
    eta=0
  fi
  printf '[%d/%d] elapsed=%ss eta=%ss item=context_%s\n' "$item_index" "$total" "$elapsed" "$eta" "$depth" | tee -a "$LOG"

  set +e
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} "$LLAMA_CPP_DIR/build/bin/llama-bench" \
    -m "$MODEL" \
    -ngl "$N_GPU_LAYERS" \
    -p "$PROMPT_TOKENS" \
    -n "$GEN_TOKENS" \
    -d "$depth" \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    -fa "$FLASH_ATTN" \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    -r "$REPETITIONS" \
    -o csv \
    $EXTRA_ARGS > "$tmp" 2>> "$LOG"
  rc=$?
  set -e

  if [ "$rc" -eq 0 ]; then
    if [ ! -s "$OUT" ]; then
      cat "$tmp" >> "$OUT"
    else
      tail -n +2 "$tmp" >> "$OUT"
    fi
    status=ok
  else
    status=failed
    printf 'context %s failed with rc=%s\n' "$depth" "$rc" | tee -a "$LOG"
  fi

  elapsed=$(( $(date +%s) - start ))
  python3 - "$CHECKPOINT" "$item_index" "$total" "$depth" "$status" "$elapsed" <<'PY'
import datetime, json, sys
path, idx, total, depth, status, elapsed = sys.argv[1:]
elapsed_i = int(elapsed)
data = {
    "completed_index": int(idx),
    "total": int(total),
    "latest_item": depth,
    "latest_status": status,
    "elapsed_seconds": elapsed_i,
    "wall_clock": str(datetime.timedelta(seconds=elapsed_i)),
}
open(path, "w", encoding="utf-8").write(json.dumps(data, indent=2) + "\n")
PY
done

elapsed=$(( $(date +%s) - start ))
printf 'done elapsed=%ss output=%s checkpoint=%s\n' "$elapsed" "$OUT" "$CHECKPOINT" | tee -a "$LOG"
