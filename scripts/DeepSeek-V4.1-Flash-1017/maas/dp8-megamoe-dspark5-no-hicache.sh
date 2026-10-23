#!/usr/bin/env bash
set -euo pipefail

export HOST_IP=$(hostname -i | awk '{print $1}')
export MASTER_ADDR="${MASTER_ADDR:-${HOST_IP}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
# Keep Engram tables on host to leave room for FP8 expert conversion and KV cache.
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
printf 'MASTER_ADDR: %s\n' "$MASTER_ADDR"

cd /root/Deploy-Env
source .venv/bin/activate

# H200 / SM90 MegaMoE requires block-FP8 experts and runner=auto.
# Text-only deployment with DSPARK block size 5; Engram remains on host.
export SGLANG_DSV4_FP4_DEQUANT=1
unset SGLANG_HUMMING_INPUT_QUANT_CONFIG
# 32768 global prefill tokens / DP8 = 4096 tokens per rank.
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
LOG_DIR="${LOG_DIR:-/root/Deploy-Env/logs/DeepSeek-V4.1-Flash-1017/megamoe-ep8-dspark5-no-hicache/$(TZ=Asia/Shanghai date '+%Y-%m-%d_%H-%M-%S')}"
mkdir -p "$LOG_DIR"
# HiCache (host/storage KV tiers) disabled; GPU prefix cache remains enabled.
printf 'SGLang log: %s/sglang.log\n' "$LOG_DIR"

NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
sglang serve \
    --trust-remote-code \
    --model-path "${MODEL_PATH:-/mnt/models2/DeepSeek-V4.1-Flash-1017}" \
    --served-model-name DeepSeek-V4.1-Flash-1017 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --tp 8 --ep 8 \
    --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
    --load-balance-method total_requests \
    --enable-prefill-delayer \
    --prefill-delayer-max-delay-passes 16 \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --host 0.0.0.0 \
    --port "${PORT:-5050}" \
    --mem-fraction-static 0.80 \
    --chunked-prefill-size 32768 \
    --max-running-requests 256 \
    --cuda-graph-max-bs-decode 64 \
    --watchdog-timeout 3600 \
    --enable-metrics \
    --enable-cache-report \
    "$@" \
    --json-model-override-args '{"vision_n_layers":0,"language_model_only":true}' \
    --moe-a2a-backend megamoe \
    --moe-runner-backend auto > "$LOG_DIR/sglang.log" 2>&1
