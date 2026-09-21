#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/Deploy-Env"

# 需要鉴权时设置 OPENAI_API_KEY，或追加 --api-key 参数。
# 默认不清缓存；需要时追加：
#   --flush-cache-url http://${ip}:${port}/flush_cache
# 下方为默认配置；命令行追加的同名参数会覆盖它们。
uv run scripts/benchmark/bench_multiturn_api.py \
  --base-url http://10.100.184.166:5050/v1 \
  --model DeepSeek-V4-Flash-0731 \
  --tokenizer /mnt/models/DeepSeek-V4-Flash-0731 \
  --input-len 4096 \
  --output-len 128 \
  --append-len 256 \
  --num-rounds 5 \
  --num-requests 128 \
  --max-concurrency 128 \
  "$@"
