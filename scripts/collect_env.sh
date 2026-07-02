#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-results/raw/env_$(date +%Y%m%d_%H%M%S).txt}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}
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

  echo "# llama.cpp"
  git -C "$LLAMA_CPP_DIR" rev-parse HEAD || true
  git -C "$LLAMA_CPP_DIR" status --short || true
  echo

  echo "# llama binaries"
  "$LLAMA_CPP_DIR/build/bin/llama-cli" --version || true
  "$LLAMA_CPP_DIR/build/bin/llama-bench" --help | head -40 || true
  echo

  echo "# model"
  echo "MODEL=${MODEL:-unset}"
  if [ -n "${MODEL:-}" ] && [ -f "$MODEL" ]; then
    ls -lh "$MODEL"
    sha256sum "$MODEL" | cut -c1-32
  fi
} | tee "$OUT"

