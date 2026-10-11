#!/usr/bin/env bash
set -euo pipefail

export HOST_IP=$(hostname -i | awk '{print $1}')
export MASTER_ADDR="${MASTER_ADDR:-${HOST_IP}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
# Four H200s need HBM headroom for expert repacking and the KV cache.
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
printf 'MASTER_ADDR: %s\n' "$MASTER_ADDR"

cd /root/Deploy-Env
source .venv/bin/activate

# humming: native FP4 W4A8; megamoe: FP4 -> FP8.
# Native FP4 deep_gemm requires SM100/SM120 in the tested build, not H200.
MOE_BACKEND="${MOE_BACKEND:-humming}"
case "$MOE_BACKEND" in
    humming)
        export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'
        MOE_ARGS=(--moe-runner-backend humming)
        ;;
    deep_gemm)
        MOE_ARGS=(--moe-runner-backend deep_gemm)
        ;;
    megamoe)
        # Experimental: FP8 expert repacking OOMs on the tested 4 x H200.
        export SGLANG_DSV4_FP4_DEQUANT=1
        export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=1024
        # This SGLang build does not support V4.1 vision with MoE A2A.
        # Runtime FP4 -> FP8 requires runner=auto; SM90 MegaMoE uses DeepGEMM.
        MOE_ARGS=(--ep 4 --moe-a2a-backend megamoe --moe-runner-backend auto
            --json-model-override-args '{"vision_n_layers":0,"language_model_only":true}')
        ;;
    *) printf 'Unknown MOE_BACKEND: %s\n' "$MOE_BACKEND" >&2; exit 1 ;;
esac

LOG_DIR="${LOG_DIR:-/root/Deploy-Env/logs/DeepSeek-V4.1-Flash/${MOE_BACKEND}/${MASTER_ADDR-:$(TZ=Asia/Shanghai date '+%Y-%m-%d_%H-%M-%S')}}"
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

NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE" \
MOONCAKE_MASTER="127.0.0.1:30002" \
MOONCAKE_LOCAL_HOSTNAME="${HOST_IP}" \
MOONCAKE_PROTOCOL="tcp" \
MC_FORCE_TCP=1 \
MOONCAKE_GLOBAL_SEGMENT_SIZE="100gb" \
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
SGLANG_ENABLE_STORAGE_METRICS=1 \
sglang serve \
    --trust-remote-code \
    --model-path "${MODEL_PATH:-/mnt/models2/DeepSeek-V4.1-Flash}" \
    --served-model-name DeepSeek-V4.1-Flash \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --tp 4 \
    --dp-size 4 --enable-dp-attention --enable-dp-lm-head \
    --load-balance-method total_requests \
    --enable-prefill-delayer \
    --prefill-delayer-max-delay-passes 16 \
    "${MOE_ARGS[@]}" \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --host 0.0.0.0 \
    --port 5050 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 4096 \
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
    "$@" > "$LOG_DIR/sglang.log" 2>&1
