#!/usr/bin/env bash
set -euo pipefail

export HOST_IP=$(hostname -i | awk '{print $1}')
export MASTER_ADDR="${MASTER_ADDR:-${HOST_IP}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
# Keep TP-sharded Engram tables in GPU memory.
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
printf 'MASTER_ADDR: %s\n' "$MASTER_ADDR"

cd /root/Deploy-Env
source .venv/bin/activate

# H200 / SM90 MegaMoE requires block-FP8 experts and runner=auto.
# Text-only deployment; DSPARK is intentionally disabled for this baseline.
export SGLANG_DSV4_FP4_DEQUANT=1
unset SGLANG_HUMMING_INPUT_QUANT_CONFIG
# 32768 global prefill tokens / DP8 = 4096 tokens per rank.
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
# Dedicated Mooncake ports for this deployment.
MOONCAKE_HTTP_PORT="${MOONCAKE_HTTP_PORT:-31001}"
MOONCAKE_RPC_PORT="${MOONCAKE_RPC_PORT:-31002}"
MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-9103}"

LOG_DIR="${LOG_DIR:-/root/Deploy-Env/logs/DeepSeek-V4.1-Flash-1017/megamoe-ep8-gpu-engram/$(TZ=Asia/Shanghai date '+%Y-%m-%d_%H-%M-%S')}"
mkdir -p "$LOG_DIR"
printf 'Mooncake log: %s/mooncake.log\nSGLang log: %s/sglang.log\n' "$LOG_DIR" "$LOG_DIR"

mooncake_master \
    --eviction_high_watermark_ratio=0.80 \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
    --metrics_port="$MOONCAKE_METRICS_PORT" \
    --rpc_port="$MOONCAKE_RPC_PORT" > "$LOG_DIR/mooncake.log" 2>&1 &
MOONCAKE_PID=$!
trap 'kill "$MOONCAKE_PID" 2>/dev/null || true' EXIT

NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE" \
MOONCAKE_MASTER="127.0.0.1:${MOONCAKE_RPC_PORT}" \
MOONCAKE_LOCAL_HOSTNAME="${HOST_IP}" \
MOONCAKE_PROTOCOL="tcp" \
MC_FORCE_TCP=1 \
MOONCAKE_GLOBAL_SEGMENT_SIZE="100gb" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_ENABLE_STORAGE_METRICS=1 \
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
    --host 0.0.0.0 \
    --port "${PORT:-5050}" \
    --mem-fraction-static 0.80 \
    --chunked-prefill-size 32768 \
    --max-running-requests 256 \
    --cuda-graph-max-bs-decode 64 \
    --watchdog-timeout 3600 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --hicache-write-policy write_through \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend mooncake \
    "$@" \
    --json-model-override-args '{"vision_n_layers":0,"language_model_only":true}' \
    --moe-a2a-backend megamoe \
    --moe-runner-backend auto > "$LOG_DIR/sglang.log" 2>&1
