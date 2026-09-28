#!/usr/bin/env bash

export HOST_IP=$(hostname -i)
export MASTER_ADDR="${MASTER_ADDR:-${HOST_IP}}"
printf 'MASTER_ADDR: %s\n' "$MASTER_ADDR"

cd /root/Deploy-Env
source .venv/bin/activate
python scripts/GLM-5.3/maas/prepare-dspark-config.py || exit 1

LOG_DIR="${LOG_DIR:-/root/Deploy-Env/logs/GLM-5.3/1n-pp4-dspark/$(TZ=Asia/Shanghai date '+%Y-%m-%d_%H-%M-%S')}"
mkdir -p "$LOG_DIR"
printf 'SGLang log: %s/sglang.log\n' "$LOG_DIR"

NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_PP_LAYER_PARTITION="21,21,21,15" \
sglang serve \
    --model-path /mnt/models2/GLM-5.3 \
    --served-model-name GLM-5.3 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --trust-remote-code \
    --tp-size 2 \
    --moe-dense-tp-size 1 \
    --ep-size 2 \
    --pp-size 4 \
    --moe-runner-backend deep_gemm \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path /mnt/models2/GLM-5.3-speculator.dspark-sglang \
    --speculative-dspark-block-size 3 \
    --pp-prefill-delay-min-tokens 32768 \
    --pp-prefill-delay-max-passes 32 \
    --allow-auto-truncate \
    --host 0.0.0.0 \
    --port 5050 \
    --watchdog-timeout 3600 \
    --dist-timeout 3600 \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --disable-prefill-cuda-graph \
    --cuda-graph-max-bs-decode 10 \
    --mem-fraction-static 0.90 \
    --enable-metrics \
    --enable-cache-report \
    --enable-hierarchical-cache \
    --hicache-ratio 4 \
    --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct \
    --hicache-write-policy write_back > "$LOG_DIR/sglang.log" 2>&1
