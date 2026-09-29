#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/Deploy-Env"

export TZ="${TZ:-Asia/Shanghai}"

cd "${REPO_ROOT}"
source .venv/bin/activate

ROUTER_LOG_DIR="${ROUTER_LOG_DIR:-${REPO_ROOT}/logs/router/Kimi-K3/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${ROUTER_LOG_DIR}"

python -m sglang_router.launch_router \
    --host 0.0.0.0 \
    --port 5050 \
    --policy cache_aware --cache-threshold 0.5 --balance-abs-threshold 32 --balance-rel-threshold 1.5 --eviction-interval-secs 120 \
    --log-dir "${ROUTER_LOG_DIR}" \
    --prometheus-host 0.0.0.0 --prometheus-port 29000 \
    --health-check-interval-secs 120 \
    --health-check-timeout-secs 60 \
    --health-success-threshold 1 \
    --health-failure-threshold 3 \
    --retry-max-retries 120 \
    --retry-initial-backoff-ms 5000 \
    --retry-max-backoff-ms 5000 \
    --retry-backoff-multiplier 1.0 \
    --request-timeout-secs 36000 \
    --enable-igw
