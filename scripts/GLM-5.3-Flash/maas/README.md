# GLM-5.3-Flash · 8 × H200

已在指定 GPU 工作空间验证：DP8 + DP attention + EP8、DeepGEMM、Mooncake L3 HiCache、prefill delay，以及 RedHatAI DSpark 草稿模型。SGLang 为本仓库的 0.5.21，使用 `/root/Deploy-Env/.venv`。

```bash
# 当前 CPU 工作区与 GPU 工作空间共享 /mnt/models2；GPU 容器无法直连下载站。
bash scripts/GLM-5.3-Flash/maas/download.sh

# 在 GPU 工作空间启动，默认开启 DSpark。
bash scripts/GLM-5.3-Flash/maas/dp8-mooncake.sh

# 非 speculative 基线。
SPECULATIVE=none bash scripts/GLM-5.3-Flash/maas/dp8-mooncake.sh
```

服务端口为 `5050`，模型名为 `GLM-5.3-Flash`；分布式初始化使用 `${MASTER_ADDR}:20000`，Mooncake 使用 `30001/30002`。`MASTER_ADDR` 默认取本机首个 IP。模型权重放在 `/mnt/models2/GLM-5.3-Flash`。

`--tp-size 8 --dp-size 8 --enable-dp-attention` 使用共 8 张卡，注意力 TP 为 1。MoE 使用 EP8 与 `deep_gemm`，通信后端为 `none`，兼容当前 SGLang 的 DSpark + DP attention 实现。

DSpark 使用 [RedHatAI/GLM-5.3-Flash-speculator.dspark-preview](https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview)。适配脚本展开配置、将捕获层号减一，并为原权重创建软链接。默认 block size 为 3；可通过 `DSPARK_BLOCK_SIZE` 调整，调整后需重新核对 KDA 状态缓存和并发上限。

DSpark 的草稿 KV 与目标模型共享 token 索引空间。实测 BF16 草稿 KV 会耗尽显存，因此草稿使用 FP8 KV，目标模型使用 BF16 KV；token 池为 1,048,576。全局并发上限为 64，即 DP8 下每 rank 8 个请求，decode CUDA Graph 最大 batch size 为每 rank 8。KDA 状态缓存保留全局 640 槽、每 rank 80 槽，供活跃请求与缓存前缀使用；该池大小也参与 HiCache 容量计算，因此不随并发上限缩小。L2 HiCache 为每 rank 16 GiB，草稿侧 host KV 约 25.5 GB；L3 使用 Mooncake。prefill delay 为 16 passes。

当前 GPU 容器没有 `/dev/infiniband`，脚本自动使用 TCP 并设置 `NCCL_IB_DISABLE=1`。有 RDMA 设备挂载时默认使用 RDMA，`NCCL_IB_HCA` 和 `MOONCAKE_DEVICE` 为 `mlx5_0` 到 `mlx5_7`，均可通过同名环境变量覆盖。单机 GPU 通信仍使用 NVLink。

已先执行用户指定的 `zai-org/GLM-5.3-Flash` 下载命令。该名字在 ModelScope 国内站不存在，客户端回退到国际镜像；最终使用国内官方 `ZhipuAI/GLM-5.3-Flash` 仓库下载全部 62 个分片（约 328 GB），配置一致，下载器的 SHA256 校验通过。下载脚本默认使用国内仓库；`MODEL_REPO` 可覆盖仓库名。

在这台机器上，固定本地文本以输入长度参数 512、输出 128 进行压测（服务端实际统计的输入平均约 496 tokens），seed=42，预热 8 次，每轮清空设备和 host 缓存；测量前保留该轮预热产生的 L3 数据：

| 配置 | 并发 | 成功请求 | 输出 tokens/s | 平均 TTFT | 平均 TPOT |
|---|---:|---:|---:|---:|---:|
| 基线 | 1 | 16/16 | 66.2 | 314.6 ms | 12.74 ms |
| DSpark block=3 | 1 | 16/16 | 133.5 | 360.9 ms | 4.69 ms |
| 基线 | 8 | 64/64 | 524.2 | 239.8 ms | 13.46 ms |
| DSpark block=3 | 8 | 64/64 | 665.1 | 516.6 ms | 7.66 ms |

DSpark 提升 decode 速度，同时增加 prefill 延迟。两种模式均通过单请求和 8 路并发生成测试；压测均记录到 448 个 token 的 Mooncake L3 命中。原始请求、响应和压测 JSONL 位于 `/root/Deploy-Env/logs/GLM-5.3-Flash/`；当前 DSpark 服务日志位于 `dspark-bs3-balanced/sglang.log`。

```bash
curl http://127.0.0.1:5050/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"请计算2+2，只输出答案。"}],"reasoning_effort":"low","temperature":0,"max_tokens":128}'
```

启动配置依据：[SGLang H200 配方](https://lmsysorg.mintlify.app/cookbook/autoregressive/GLM/GLM-5.3-Flash#hw=h200)。草稿配置转换依据：[Speculators 的 SGLang 导出说明](https://github.com/vllm-project/speculators/issues/753)。
