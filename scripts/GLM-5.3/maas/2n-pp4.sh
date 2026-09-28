#!/usr/bin/env bash

export MASTER_ADDR="${MASTER_ADDR:-${LWS_LEADER_ADDRESS}}"
export NNODES="${NNODES:-${LWS_GROUP_SIZE}}"
export NODE_RANK="${NODE_RANK:-${LWS_WORKER_INDEX:-0}}"
export SGLANG_DIST_ADDR="${SGLANG_DIST_ADDR:-${MASTER_ADDR}:20000}"
echo "MASTER: ${SGLANG_DIST_ADDR}, RANK: ${NODE_RANK}"

cd /root/Deploy-Env
source .venv/bin/activate

LOG_DIR="${LOG_DIR:-/root/Deploy-Env/logs/GLM-5.3/2n-pp4/${MASTER_ADDR}/rank-${NODE_RANK}/$(TZ=Asia/Shanghai date '+%Y-%m-%d_%H-%M-%S')}"
mkdir -p "$LOG_DIR"
printf 'SGLang log: %s/sglang.log\n' "$LOG_DIR"

NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_PP_LAYER_PARTITION="20,20,19,19" \
sglang serve \
    --model-path /mnt/models2/GLM-5.3 \
    --served-model-name GLM-5.3 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --trust-remote-code \
    --tp-size 4 \
    --moe-dense-tp-size 1 \
    --ep-size 4 \
    --pp-size 4 \
    --dcp-size 4 \
    --dcp-replicate-q-proj \
    --dsa-prefill-backend fa3 \
    --dsa-decode-backend fa3 \
    --moe-runner-backend deep_gemm \
    --pp-prefill-delay-min-tokens 32768 \
    --pp-prefill-delay-max-passes 32 \
    --dist-init-addr "$SGLANG_DIST_ADDR" \
    --nnodes 2 \
    --node-rank "$NODE_RANK" \
    --host 0.0.0.0 \
    --port 5050 \
    --watchdog-timeout 3600 \
    --dist-timeout 3600 \
    --chunked-prefill-size 8192 \
    --max-running-requests 256 \
    --disable-prefill-cuda-graph \
    --cuda-graph-max-bs-decode 66 \
    --mem-fraction-static 0.88 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct \
    --hicache-write-policy write_back > "$LOG_DIR/sglang.log" 2>&1
