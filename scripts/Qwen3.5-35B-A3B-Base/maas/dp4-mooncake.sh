#!/usr/bin/env bash

export HOST_IP=$(hostname -i)
export MASTER_ADDR="${MASTER_ADDR:-${HOST_IP}}"
printf 'MASTER_ADDR: %s\n' "$MASTER_ADDR"

cd /root/Deploy-Env
source .venv/bin/activate

LOG_DIR="${LOG_DIR:-/root/Deploy-Env/logs/Qwen3.5-35B-A3B-Base/dp4-mooncake/${MASTER_ADDR-:$(TZ=Asia/Shanghai date '+%Y-%m-%d_%H-%M-%S')}}"
mkdir -p "$LOG_DIR" || exit 1
printf 'Mooncake log: %s/mooncake.log\nSGLang log: %s/sglang.log\n' "$LOG_DIR" "$LOG_DIR"

mooncake_master \
    --eviction_high_watermark_ratio=0.80 \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=30001 \
    --rpc_port=30002 > "$LOG_DIR/mooncake.log" 2>&1 &

NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE" \
MOONCAKE_MASTER="127.0.0.1:30002" \
MOONCAKE_LOCAL_HOSTNAME="${HOST_IP}" \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" \
MOONCAKE_GLOBAL_SEGMENT_SIZE="100gb" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_ENABLE_STORAGE_METRICS=1 \
sglang serve \
    --trust-remote-code \
    --model-path /mnt/models/Qwen3.5-35B-A3B-Base \
    --served-model-name Qwen3.5-35B-A3B-Base \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --tp-size 2 \
    --dp-size 4 \
    --load-balance-method total_requests \
    --allow-auto-truncate \
    --host 0.0.0.0 \
    --port 5050 \
    --max-running-requests 64 \
    --max-mamba-cache-size 320 \
    --mamba-radix-cache-strategy extra_buffer \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 1 \
    --hicache-write-policy write_through \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend=mooncake > "$LOG_DIR/sglang.log" 2>&1
