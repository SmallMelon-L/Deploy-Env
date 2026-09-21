#!/usr/bin/env bash

export HOST_IP=$(hostname -i)
export MASTER_ADDR="${MASTER_ADDR:-${HOST_IP}}"
printf 'MASTER_ADDR: %s\n' "$MASTER_ADDR"

cd /root/Deploy-Env
source .venv/bin/activate

LOG_DIR="${LOG_DIR:-/root/Deploy-Env/logs/DeepSeek-V4-Flash/dp4-mooncake/${MASTER_ADDR-:$(TZ=Asia/Shanghai date '+%Y-%m-%d_%H-%M-%S')}}"
mkdir -p "$LOG_DIR" || exit 1
printf 'Mooncake log: %s/mooncake.log\nSGLang log: %s/sglang.log\n' "$LOG_DIR" "$LOG_DIR"

mooncake_master \
    --eviction_high_watermark_ratio=0.80 \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=30001 \
    --rpc_port=30002 > "$LOG_DIR/mooncake.log" 2>&1 &

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE" \
MOONCAKE_MASTER="127.0.0.1:30002" \
MOONCAKE_LOCAL_HOSTNAME="${HOST_IP}" \
MOONCAKE_GLOBAL_SEGMENT_SIZE="100gb" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_ENABLE_STORAGE_METRICS=1 \
sglang serve \
    --trust-remote-code \
    --model-path /mnt/models/DeepSeek-V4-Flash \
    --served-model-name DeepSeek-V4-Flash \
    --reasoning-parser deepseek-v4 \
    --tool-call-parser deepseekv4 \
    --tp 4 \
    --dp-size 4 --enable-dp-attention --enable-dp-lm-head \
    --ep 4 \
    --load-balance-method total_requests \
    --enable-prefill-delayer \
    --prefill-delayer-max-delay-passes 16 \
    --moe-runner-backend flashinfer_mxfp4 \
    --flashinfer-mxfp4-moe-precision fp8 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path /mnt/models/DeepSeek-V4-Flash-DSpark \
    --speculative-dspark-block-size 7 \
    --host 0.0.0.0 \
    --port 5050 \
    --max-running-requests 256 \
    --cuda-graph-max-bs-decode 64 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 1 \
    --hicache-write-policy write_through \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend=mooncake > "$LOG_DIR/sglang.log" 2>&1
