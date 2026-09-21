#!/usr/bin/env bash
set -euo pipefail

cd /root/Deploy-Env

# 下方为默认配置；命令行追加的同名参数会覆盖它们。
uv run scripts/benchmark/bench_multiturn_api.py \
  --base-url http://10.68.84.252:80/v1 \
  --model m-20260919023219-7m7cf/DeepSeek-V4-Flash-0731 \
  --api-key "${MODEL_SERVER_API_KEY}" \
  --tokenizer /mnt/models/DeepSeek-V4-Flash-0731 \
  --input-len 4096 \
  --output-len 128 \
  --append-len 256 \
  --num-rounds 5 \
  --num-requests 128 \
  --max-concurrency 128 \
  "$@"
