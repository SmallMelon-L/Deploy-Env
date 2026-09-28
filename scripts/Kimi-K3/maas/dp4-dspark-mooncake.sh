export MASTER_ADDR="${MASTER_ADDR:-${LWS_LEADER_ADDRESS}}"
export NNODES="${NNODES:-${LWS_GROUP_SIZE}}"
export NODE_RANK="${NODE_RANK:-${LWS_WORKER_INDEX:-0}}"
export SGLANG_DIST_ADDR="${SGLANG_DIST_ADDR:-${MASTER_ADDR}:20000}"
export MOONCAKE_MASTER_ADDR="${MASTER_ADDR}:30002"
export HOST_IP=$(hostname -i)
echo "MASTER: ${SGLANG_DIST_ADDR}, RANK: ${NODE_RANK}, MOONCAKE: ${MOONCAKE_MASTER_ADDR}, IP: ${HOST_IP}"

cd /root/Deploy-Env
source .venv/bin/activate

LOG_DIR="${LOG_DIR:-/root/Deploy-Env/logs/Kimi-K3/dp4-mooncake/${MASTER_ADDR-:$(TZ=Asia/Shanghai date '+%Y-%m-%d_%H-%M-%S')}}"
mkdir -p "$LOG_DIR" || exit 1
printf 'Mooncake log: %s/mooncake.log\nSGLang log: %s/sglang.log\n' "$LOG_DIR" "$LOG_DIR"

# start mooncake-master (only on rank 0)
if [ "${NODE_RANK}" = "0" ]; then
    mooncake_master \
        --eviction_high_watermark_ratio=0.80 \
        --enable_http_metadata_server=true \
        --http_metadata_server_host=0.0.0.0 \
        --http_metadata_server_port=30001 \
        --rpc_port=30002 > "$LOG_DIR/mooncake_master.log" 2>&1 &
fi

# start sglang
NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE" \
MOONCAKE_MASTER="${MOONCAKE_MASTER_ADDR}" \
MOONCAKE_LOCAL_HOSTNAME="${HOST_IP}" \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" \
MOONCAKE_GLOBAL_SEGMENT_SIZE="400gb" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_ENABLE_STORAGE_METRICS=1 \
sglang serve \
    --model-path /mnt/models/Kimi-K3 \
    --served-model-name Kimi-K3 \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --trust-remote-code \
    --tp-size 32 \
    --dp-size 4 --enable-dp-attention --enable-dp-lm-head \
    --ep-size 32 \
    --moe-runner-backend marlin \
    --decode-attention-backend flashmla \
    --load-balance-method total_requests \
    --allow-auto-truncate \
    --enable-prefill-delayer \
    --prefill-delayer-max-delay-passes 16 \
    --dist-init-addr ${SGLANG_DIST_ADDR} \
    --nnodes 4 \
    --node-rank ${NODE_RANK} \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path /mnt/models/Kimi-K3-DSpark \
    --speculative-dspark-block-size 3 \
    --enable-linear-replayssm-spec \
    --host 0.0.0.0 \
    --port 5050 \
    --watchdog-timeout 3600 \
    --dist-timeout 3600 \
    --chunked-prefill-size 16384 \
    --max-running-requests 64 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --max-mamba-cache-size 256 \
    --cuda-graph-max-bs-decode 16 \
    --mem-fraction-static 0.90 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --hicache-write-policy write_through \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend=mooncake > "$LOG_DIR/sglang.log" 2>&1
