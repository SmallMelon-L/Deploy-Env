export MASTER_ADDR="${MASTER_ADDR:-${LWS_LEADER_ADDRESS}}"
export NNODES="${NNODES:-${LWS_GROUP_SIZE}}"
export NODE_RANK="${NODE_RANK:-${LWS_WORKER_INDEX:-0}}"
export SGLANG_DIST_ADDR="${SGLANG_DIST_ADDR:-${MASTER_ADDR}:20000}"

cd /root/Deploy-Env
source .venv/bin/activate

NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
SGLANG_PP_LAYER_PARTITION="24,24,24,21" \
sglang serve \
    --model-path /mnt/models/Kimi-K3 \
    --served-model-name kimi-k3 \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --trust-remote-code \
    --tp-size 8 \
    --pp-size 4 \
    --ep-size 8 \
    --moe-runner-backend marlin \
    --decode-attention-backend flashmla \
    --load-balance-method total_requests \
    --dist-init-addr ${SGLANG_DIST_ADDR} \
    --nnodes 4 \
    --node-rank ${NODE_RANK} \
    --host 0.0.0.0 \
    --port 5050 \
    --watchdog-timeout 3600 \
    --dist-timeout 3600 \
    --chunked-prefill-size 8192 \
    --max-running-requests 100 \
    --max-mamba-cache-size 400 \
    --cuda-graph-max-bs-decode 22 \
    --pp-async-batch-depth 1 \
    --pp-max-micro-batch-size 20 \
    --mem-fraction-static 0.90 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --hicache-write-policy write_through