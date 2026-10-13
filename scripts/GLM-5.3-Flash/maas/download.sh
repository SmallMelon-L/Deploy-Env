#!/usr/bin/env bash
set -euo pipefail

cd /root/Deploy-Env
source .venv/bin/activate

# The CN official repo has all 62 shards; zai-org currently resolves to the international mirror.
NO_PROXY="${NO_PROXY:+${NO_PROXY},}modelscope.cn,modelscope.com,aliyuncs.com" \
no_proxy="${no_proxy:+${no_proxy},}modelscope.cn,modelscope.com,aliyuncs.com" \
modelscope download --model "${MODEL_REPO:-ZhipuAI/GLM-5.3-Flash}" \
    --local-dir /mnt/models2/GLM-5.3-Flash --max-workers 16

if [ "${SPECULATIVE:-dspark}" = dspark ]; then
    hf download RedHatAI/GLM-5.3-Flash-speculator.dspark-preview \
        --local-dir /mnt/models2/GLM-5.3-Flash-speculator.dspark
    python scripts/GLM-5.3-Flash/maas/prepare-dspark-config.py
fi
