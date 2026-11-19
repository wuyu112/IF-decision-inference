#!/usr/bin/env bash
set -Eeuo pipefail
MODEL=""
PORT=18080
VLLM_PORT=18081
GPU=0
while (($#)); do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --vllm-port) VLLM_PORT="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$MODEL" ]] || { echo "--model is required" >&2; exit 2; }
export STARTLUX_VLLM_MAX_NUM_SEQS=8
exec startlux-decide serve \
  --model "$MODEL" \
  --name IF-decision-27b-v1 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --launch-vllm \
  --gpus "$GPU" \
  --vllm-base-port "$VLLM_PORT" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --startup-timeout 1200
