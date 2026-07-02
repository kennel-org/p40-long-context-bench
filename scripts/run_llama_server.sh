#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: scripts/run_llama_server.sh configs/<name>.env}
set -a
# shellcheck disable=SC1090
. "$CONFIG"
set +a

LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}
MODEL=${MODEL:?MODEL is required}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CTX_SIZE=${CTX_SIZE:-32768}
N_GPU_LAYERS=${N_GPU_LAYERS:-41}
PARALLEL=${PARALLEL:-8}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-512}
CACHE_TYPE_K=${CACHE_TYPE_K:-q8_0}
CACHE_TYPE_V=${CACHE_TYPE_V:-q8_0}
EXTRA_ARGS=${EXTRA_ARGS:-}

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" "$LLAMA_CPP_DIR/build/bin/llama-server" \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -ngl "$N_GPU_LAYERS" \
  -c "$CTX_SIZE" \
  --parallel "$PARALLEL" \
  -b "$BATCH_SIZE" \
  -ub "$UBATCH_SIZE" \
  -fa auto \
  -ctk "$CACHE_TYPE_K" \
  -ctv "$CACHE_TYPE_V" \
  --metrics \
  $EXTRA_ARGS
