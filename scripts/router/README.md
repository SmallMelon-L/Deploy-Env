# 平台 worker 自动同步

在仓库根目录运行：

```bash
uv run scripts/router/sync_workers.py 172.21.120.82:5050 DeepSeek-V4-Flash
```

启动后立即同步，默认每 10 秒重新查询平台部署和 router 的 `/workers`：

- 精确匹配名称（忽略大小写）的所有易有云部署，从每个副本中选择
  rank0/leader IP，合并去重后注册为 `http://IP:5050`。
- 平台有、router 没有的 IP，先直连 `http://IP:5050/v1/chat/completions`
  发送 `{"temperature":1.0,"stream":true,"messages":[{"role":"user","content":"hello"}]}`。
  收到非空回答内容、正常结束标记和 `[DONE]` 后，才通过 `POST /workers` 注册。
  连接失败、超时、HTTP/流式错误或响应不完整时，本轮跳过该 IP，下一轮重试，
  继续处理其他 IP。已注册的 IP 不重复探测；`--dry-run` 会探测候选 IP，
  但不提交注册或移除。
- router 有、平台没有的 IP，通过 `DELETE /workers/{id}` 移除。
- DP-aware 的 `IP:5050@0` 等 rank 按 IP 合并比较，移除时删除整个 DP 分组。
- 管理的是指定 router 的**全部 worker**；因此应使用该部署专用的 router。

复用 `scripts/yicloud/server_ips.py` 的 SDK 和认证配置：凭证从环境变量
`YICLOUD_PUBLIC_KEY`（Access Key ID）、`YICLOUD_SECRET_KEY`（Access Key Secret）读取。
两个变量均须设置且非空，否则启动时报错；不再读取 `/.auth/` 下的凭证文件。
项目默认 `fdj-infra`，
可通过 `YICLOUD_PROJECT` 或 `--project` 指定。router 如需认证，设置
`SGLANG_ROUTER_API_KEY`。请求不使用系统代理。

平台查询支持分页。同名部署会全部纳入同步；任一部署查询失败、响应不完整或
IP 格式异常时，
本轮不会按错误的空集合移除 worker，常驻模式会在下一轮重试。
尚未分配 IP 的副本跳过；部署确实没有任何已分配的 worker IP 时，会清空 router。
同一个 IP 已存在时保留现有注册；新注册的端口默认 5050。
每轮只打印一条汇总：`platform`、`router`、`unready`、`add`、`remove`。
`unready` 为平台有、router 没有且探测失败的 IP；`add` 为其中探测通过的 IP。
不打印逐个探测、操作或异常日志；查询失败时跳过该轮并重试。
推理探测默认时间限制和 socket 超时均为 10 秒，可用 `--probe-timeout 30` 调整。
若流读取正阻塞，时间限制最迟在该次 socket 读取超时后生效。
各候选 IP 并发探测，最多同时 32 个；超过 32 个时，其余等待空闲名额。
本轮全部探测结束后打印一条汇总日志，再执行注册和移除。

注册/移除是异步操作，HTTP 202 仅表示已排队。脚本通过后续 `/workers`
查询检查实际结果，默认 120 秒内不重复提交尚未完成的同一操作，超时后重试。
DP 分组缺少原始注册 UUID 时，会先恢复 base UUID，等待该注册任务结束，
再提交分组删除，避免异步注册晚于删除完成而导致 worker 重新出现。
接口请求耗时超过 10 秒时，本轮完成后开始下一轮，不并发执行多个同步周期。
按 Ctrl+C 或发送 SIGTERM 可退出；已注册的 worker 保留。

```bash
# 只查询、打印差异，不修改 router
.venv/bin/python scripts/router/sync_workers.py 172.21.120.82:5050 DeepSeek-V4-Flash --dry-run

# 同步至 IP 集合一致后退出；默认 180 秒内未收敛则返回非零退出码
.venv/bin/python scripts/router/sync_workers.py 172.21.120.82:5050 DeepSeek-V4-Flash --once

# 查看其他参数
.venv/bin/python scripts/router/sync_workers.py --help

```

`--once` 要求 IP 集合一致；新增 worker 须先通过推理探测，尚未就绪时继续重试。
它不会重新验证已有 worker 的推理能力或每个 DP rank 的健康状态。
常驻运行需保持上述进程运行，可自行交给 tmux 或服务管理器托管。
