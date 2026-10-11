# 弹性 GPU 池

`elastic_pool.py` 默认每 60 秒检查项目 `fdj-infra`、可用区 `cn-shanghai-a`
中的 `fdj_post_train_task`，调整模型部署 `m-20260929133556-ljgv2`。
认证复用 `server_ips.py`，从环境变量 `YICLOUD_PUBLIC_KEY`、
`YICLOUD_SECRET_KEY` 读取。使用仓库的 `.venv` Python。

## 启动

默认只读，仅打印快照和拟执行的目标副本数：

```bash
cd /root/Deploy-Env
.venv/bin/python scripts/yicloud/elastic_pool.py --once
```

持续只读监控：

```bash
.venv/bin/python scripts/yicloud/elastic_pool.py
```

启用实际扩缩容，每 60 秒检查一次（前台运行，Ctrl-C 停止）：

```bash
.venv/bin/python scripts/yicloud/elastic_pool.py --apply
```

可选后台运行，凭证应事先在环境中配置：

```bash
mkdir -p logs
nohup .venv/bin/python scripts/yicloud/elastic_pool.py --apply > logs/elastic_pool.log 2>&1 &
```

`--interval` 调整检查周期；`--cooldown` 调整扩缩容完成后的等待时间，默认 120 秒。
支持 `--project`、`--quota-group`、`--deployment`、`--zone`，会校验部署所属分区、
可用区及每副本 4 台 × 8 卡的规格。不会修改平台自带自动扩缩容配置；若已启用则拒绝操作。

## 决策规则

- 只统计健康、非维护、未禁止调度的完整空闲 8 卡节点；`Used=0`、`Releasing=0`、`Usable=Total=8`。
- 排队范围是该分区该可用区当前身份可见的所有 `Inqueue,Pending` Job，`Self=False`，完整分页。
  按每个任务规格的 GPU 数乘待调度实例数累计；有状态计数时只计 `Pending`，避免重复释放已运行实例的资源。
  `Inqueue` 缺少状态计数时使用申请实例数；无法确定资源需求则整轮跳过。
- 有排队：缺口 = `max(0, 排队卡数 - 空闲整机数 × 8)`。
  缺口为零就等待调度，不扩容；否则目标 = `max(1, 当前副本 - ceil(缺口 / 32))`。
- 无排队且空闲整机 ≥4：增加 `floor(空闲整机数 / 4)` 个副本。
  先固定本次目标，再进入 5 分钟（300 秒）确认期，按监控周期（默认 60 秒）继续采样。
  每次都必须满足原有扩容条件：无排队 Job、部署稳定、无 GPU 正在释放、
  空闲整机足够支持最初计划新增的副本数。空闲机器增加时仍使用最初目标；
  减少后若仍足够支持该目标，确认继续，否则取消。本轮按当前状态重新决策，
  如仍可扩容则按新目标重新计时，如需缩容则立即走缩容流程。
  读取失败、部署变化或其他条件不满足也会取消确认；满 300 秒后的有效采样
  再次满足条件，并通过提交前部署复查，才真正提交扩容。缩容无需这段确认期。
  确认进度仅保存在进程内，重启后重新计时；只读模式也遵循确认流程，
  因此 `--once` 只读检查会打印确认开始，不会直接打印扩容的 `DRY RUN`。
- 至少保留 1 个副本；已被人工停到 0 时不自动启动。
- 部署 `Replicas` 是副本组数，`ReadyReplicas/TotalReplicas` 是实例数，应为前者的 4 倍。
  部署未 Running、实例未全就绪或分区仍有 GPU 正在释放时等待下一轮。
- 提交后等待目标副本完全就绪，再额外等待 120 秒，让调度器和资源统计收敛。

每轮日志包含空闲整机数、排队任务数、排队卡数、当前副本数、可释放卡数、目标及原因。
扩容确认日志包含 `Scale-up confirmation started`（开始）、
`Scale-up confirmation`（进度）和 `Scale-up confirmation cancelled`（取消）。
快照中的 `target` 是当前资源算出的目标；确认期实际计划以确认日志中的目标为准。
接口读取失败、分页不完整或部署在采样期间变化时跳过本轮，下一周期再查。
一次采样耗时超过周期时不会并发启动另一轮。

## 状态与限制

`logs/yicloud-elastic-pool/` 存放按项目和部署区分的锁与状态文件。
同一仓库内同一部署只允许一个控制器进程；部署到另一台主机时不要重复启动。
扩缩容意图先写入状态文件，再发送请求。即使请求超时或进程重启，也不会重复发起。
若手动把副本改为不同于旧目标和提交前副本数的新值，脚本会自动识别：旧请求已超过
120 秒，且两次至少相隔一个检查周期的采样中副本数、更新时间保持一致并全部就绪，
就清除旧请求，经过冷却期后按当前资源重新决策。
若改回提交前的副本数，无法区分人工修改和请求尚未生效，仍会等待。
请求失败或需要立即接管人工设置时，先 Ctrl-C 停止原进程，确认平台没有未完成的操作，再启动：

```bash
uv run scripts/yicloud/elastic_pool.py --apply --reset-pending
```

该选项在获取进程锁后清除旧请求和冷却状态，再从平台重新读取并决策；不会先把副本改回旧目标。
不要在正常扩缩容尚未完成时使用。此选项只需在恢复时加一次，以后正常使用 `--apply`。

整机空闲按 GPU 统计，不保证 CPU、内存、卡型、亲和性等条件也满足排队任务。
SDK 的节点与任务读取并非原子快照，也没有条件更新接口，因此外部同时提交任务或人工扩缩容仍可能产生竞态。
脚本会复查部署是否变化，但不能保证与外部调度器完全同步。
目前排队统计覆盖 Job 接口，不包含 Ray、开发机或其他模型部署的待调度资源。

离线测试：

```bash
.venv/bin/python -m unittest discover -s scripts/yicloud -p 'test_elastic_pool.py' -v
```
