#!/usr/bin/env bash
set -euo pipefail

export HOST_IP="${HOST_IP:-$(hostname -i | awk '{print $1}')}"
export MASTER_ADDR="${MASTER_ADDR:-${HOST_IP}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SGLANG_DIST_ADDR="${SGLANG_DIST_ADDR:-${MASTER_ADDR}:20000}"
export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
printf 'MASTER_ADDR: %s\n' "$MASTER_ADDR"

cd /root/Deploy-Env
source .venv/bin/activate

if [ ! -d /dev/infiniband ]; then
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
fi
if [ "${MOONCAKE_PROTOCOL:-rdma}" = tcp ]; then
    export MC_FORCE_TCP=1
fi

# SPECULATIVE=none selects the non-speculative baseline.
SPEC_ARGS=()
case "${SPECULATIVE:-dspark}" in
    none) ;;
    dspark)
        python scripts/GLM-5.3-Flash/maas/prepare-dspark-config.py
        # Dense draft KV shares target slots. Budget device and host caches together.
        SPEC_ARGS=(--speculative-algorithm DSPARK
            --speculative-draft-model-path "${DRAFT_MODEL_PATH:-/mnt/models2/GLM-5.3-Flash-speculator.dspark-sglang}"
            --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE:-3}"
            --speculative-draft-kv-cache-dtype fp8_e4m3
            --max-total-tokens 1048576
            # Keep 80 state slots/rank for 8 active requests plus cached prefixes.
            --max-mamba-cache-size 640
            --hicache-size 16
            --enable-linear-replayssm-spec)
        ;;
    *) printf 'Unknown SPECULATIVE: %s\n' "$SPECULATIVE" >&2; exit 1 ;;
esac

LOG_DIR="${LOG_DIR:-/root/Deploy-Env/logs/GLM-5.3-Flash/dp8-mooncake/${MASTER_ADDR}/$(TZ=Asia/Shanghai date '+%Y-%m-%d_%H-%M-%S')}"
mkdir -p "$LOG_DIR"
printf 'Mooncake log: %s/mooncake.log\nSGLang log: %s/sglang.log\n' "$LOG_DIR" "$LOG_DIR"

mooncake_master \
    --eviction_high_watermark_ratio=0.80 \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=30001 \
    --rpc_port=30002 > "$LOG_DIR/mooncake.log" 2>&1 &
MOONCAKE_PID=$!
trap 'kill "$MOONCAKE_PID" 2>/dev/null || true' EXIT

NCCL_IB_HCA="${NCCL_IB_HCA:-=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE" \
MOONCAKE_MASTER="${MASTER_ADDR}:30002" \
MOONCAKE_LOCAL_HOSTNAME="${HOST_IP}" \
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}" \
MOONCAKE_DEVICE="${MOONCAKE_DEVICE:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}" \
MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-32gb}" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_ENABLE_STORAGE_METRICS=1 \
sglang serve \
    --model-path "${MODEL_PATH:-/mnt/models2/GLM-5.3-Flash}" \
    --served-model-name GLM-5.3-Flash \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --trust-remote-code \
    --tp-size 8 \
    --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
    --ep-size 8 \
    --moe-runner-backend deep_gemm \
    --moe-a2a-backend none \
    --dsa-prefill-backend tilelang \
    --dsa-decode-backend tilelang \
    --kv-cache-dtype bfloat16 \
    --load-balance-method total_requests \
    --enable-prefill-delayer \
    --prefill-delayer-max-delay-passes 16 \
    --dist-init-addr "$SGLANG_DIST_ADDR" \
    --host 0.0.0.0 \
    --port 5050 \
    --mem-fraction-static 0.80 \
    --chunked-prefill-size 8192 \
    --max-running-requests 64 \
    --cuda-graph-max-bs-decode 8 \
    --watchdog-timeout 3600 \
    --dist-timeout 3600 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-size 32 \
    --hicache-write-policy write_through \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend mooncake \
    "${SPEC_ARGS[@]}" \
    "$@" > "$LOG_DIR/sglang.log" 2>&1
