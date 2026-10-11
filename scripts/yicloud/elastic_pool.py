#!/root/Deploy-Env/.venv/bin/python3
"""Use a distributed InferSet as a spare GPU pool. Read-only unless --apply."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import fcntl
import hashlib
import json
import logging
import math
from pathlib import Path
import signal
import threading
import time

from yicloud.base import msgs
from yicloud.services import inferset, job, mc
import server_ips as platform

LOG = logging.getLogger("elastic_pool")
MACHINES_PER_REPLICA = 4
GPUS_PER_NODE = 8
GPUS_PER_REPLICA = 32
SCALE_UP_CONFIRM_SECONDS = 300


def integer(value, field):
    if isinstance(value, bool):
        raise ValueError(f"Invalid {field}: {value!r}")
    try:
        number = int(str(value))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {field}: {value!r}") from None
    if number < 0:
        raise ValueError(f"Negative {field}")
    return number


def decide(replicas, idle_nodes, queued_jobs, queued_gpus):
    """Return target and reason; account for whole idle nodes before releasing."""
    if replicas < 1:
        return replicas, "deployment has no replicas; manual intervention required"
    if queued_jobs:
        shortage = max(0, queued_gpus - idle_nodes * GPUS_PER_NODE)
        if shortage == 0:
            return replicas, "idle machines cover queue; wait for scheduler"
        target = max(1, replicas - math.ceil(shortage / GPUS_PER_REPLICA))
        return target, "release queue shortfall" if target < replicas else "minimum replica retained"
    return replicas + idle_nodes // MACHINES_PER_REPLICA, "consume spare machines"


def idle_machine(node):
    """Count only healthy, schedulable, completely free eight-GPU nodes."""
    if (node.get("NodeStatus") != "True"
            or node.get("Unschedulable") is not False
            or node.get("MaintenanceStatus") not in ("", "False", False)):
        return False
    gpu = node["NodeQuota"]["GPU"]
    return (integer(gpu["Total"], "GPU.Total") == GPUS_PER_NODE
            and integer(gpu["Usable"], "GPU.Usable") == GPUS_PER_NODE
            and integer(gpu["Used"], "GPU.Used") == 0
            and integer(gpu["Releasing"], "GPU.Releasing") == 0)


def queue_demand(rows):
    total = 0
    for row in rows:
        if row.get("Phase") not in ("Inqueue", "Pending"):
            raise ValueError("Unexpected job phase in filtered response")
        specs = row.get("TaskSpecs")
        if not isinstance(specs, dict) or not specs:
            raise ValueError(f"Missing TaskSpecs for {row.get('JobId')}")
        for name, spec in specs.items():
            gpus = integer(spec.get("GPU"), "TaskSpec.GPU")
            replicas = integer(spec.get("Replicas"), "TaskSpec.Replicas")
            # Pending jobs can already have active tasks: count only remaining
            # pending instances, rather than freeing their full original request.
            counts = (row.get("TaskStatusCounts") or {}).get(name)
            if counts is not None:
                if not isinstance(counts, dict):
                    raise ValueError("Invalid task status counts")
                pending = integer(counts.get("Pending", 0), "TaskStatus.Pending")
                if pending > replicas:
                    raise ValueError("Pending task count exceeds requested replicas")
            elif row["Phase"] == "Inqueue":
                pending = replicas
            else:
                raise ValueError("Pending job lacks task status counts")
            total += gpus * pending
    return total


class API:
    def get(self, service, action, req):
        ctx, _ = msgs.new_meta_ctx()
        rsp = msgs.Rsp[dict]()
        inferset.client.base.get(ctx, f"/{service}/v1alpha1/{action}", req, rsp)
        if not isinstance(rsp.data, dict):
            raise ValueError(f"Invalid {action} response")
        return rsp.data

    def pages(self, service, action, req_type, id_key, **kwargs):
        rows, total = [], None
        while True:
            data = self.get(service, action, req_type(Limit=100, Offset=len(rows), **kwargs))
            count, page = data.get("Total"), data.get("Items")
            if type(count) is not int or count < 0 or not isinstance(page, list):
                raise ValueError(f"Invalid {action} pagination")
            if total is not None and count != total:
                raise ValueError(f"{action} changed during pagination; skipping cycle")
            total = count
            rows.extend(page)
            if len(rows) == total:
                ids = [r[id_key] for r in rows]
                if len(set(ids)) != len(ids):
                    raise ValueError(f"Duplicate {action} rows")
                return rows
            if not page or len(rows) > total:
                raise ValueError(f"Incomplete {action} response")

    def deployment(self, args):
        return self.get("inferset", "GetInferSet", inferset.GetInferSetReq(
            ID=args.deployment, Project=args.project))

    def scale(self, args, target):
        ctx, _ = msgs.new_meta_ctx()
        inferset.scale_inferset(ctx, inferset.ScaleInferSetReq(
            ID=args.deployment, Project=args.project, Replicas=target))


def validate_deployment(d, args):
    if (d.get("QuotaGroup") or {}).get("Name") != args.quota_group:
        raise ValueError("Deployment is in a different quota group")
    if d.get("ZoneId") != args.zone:
        raise ValueError("Deployment is in a different availability zone")
    if (d.get("AutoScalingConfig") or {}).get("Enabled"):
        raise ValueError("Disable platform autoscaling before using this controller")
    if d.get("PDSeparationEnabled"):
        raise ValueError("PD-separated deployments are not supported")
    if (integer((d.get("Topology") or {}).get("GroupSize"), "GroupSize") != 4
            or integer((d.get("InstanceType") or {}).get("GpuCount"), "GpuCount") != 8):
        raise ValueError("Deployment must use four eight-GPU machines per replica")
    return integer(d.get("Replicas"), "Replicas")


@dataclass
class Snapshot:
    replicas: int
    idle_nodes: int
    queued_jobs: int
    queued_gpus: int
    releasable_gpus: int
    ready_instances: int
    total_instances: int
    releasing_gpus: int
    stable: bool


def snapshot(api, args):
    before = api.deployment(args)
    replicas = validate_deployment(before, args)
    nodes = api.pages("mc", "ListNodes", mc.ListNodesReq, "NodeName",
                      Project=args.project, QuotaGroup=args.quota_group, ZoneId=args.zone)
    jobs = api.pages("job", "ListJobs", job.ListJobsReq, "JobId",
                     Project=args.project, QuotaGroup=args.quota_group,
                     ZoneId=args.zone, Phases="Inqueue,Pending", Self=False)
    for row in nodes + jobs:
        if row.get("QuotaGroup") != args.quota_group or row.get("ZoneId") != args.zone:
            raise ValueError("API returned a row outside the requested scope")
    after = api.deployment(args)
    validate_deployment(after, args)
    if any(before.get(k) != after.get(k) for k in
           ("Replicas", "UpdateTimestamp", "Status", "ReadyReplicas", "TotalReplicas")):
        raise ValueError("Deployment changed during sampling; skipping cycle")
    ready = integer(after.get("ReadyReplicas"), "ReadyReplicas")
    total = integer(after.get("TotalReplicas"), "TotalReplicas")
    releasing = sum(integer(n["NodeQuota"]["GPU"]["Releasing"], "Releasing") for n in nodes)
    stable = (after.get("Status") == "Running" and replicas > 0
              and ready == total == replicas * MACHINES_PER_REPLICA and releasing == 0)
    return Snapshot(replicas, sum(idle_machine(n) for n in nodes), len(jobs),
                    queue_demand(jobs), max(0, replicas - 1) * GPUS_PER_REPLICA,
                    ready, total, releasing, stable), after


class Controller:
    def __init__(self, args, api, state_path):
        self.args, self.api, self.state_path = args, api, state_path
        self.state = json.loads(state_path.read_text()) if state_path.exists() else {}
        self.override_candidate = None
        # Confirmation requires uninterrupted observations in this process;
        # restart always starts a fresh window, unlike submitted requests.
        self.scale_up_candidate = None

    def reset_pending(self):
        """Explicit operator recovery, called only while holding the process lock."""
        LOG.warning("Discarding saved controller state at operator request: %s", self.state)
        self.state = {}
        self.override_candidate = None
        self.scale_up_candidate = None
        self.save()

    def save(self):
        if self.args.apply:
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.state) + "\n")
            temporary.replace(self.state_path)

    def tick(self):
        try:
            self._tick()
        except Exception:
            # A failed sample cannot confirm that expansion remains safe.
            self.cancel_scale_up("monitoring cycle failed")
            raise

    def cancel_scale_up(self, reason):
        if self.scale_up_candidate is not None:
            LOG.info("Scale-up confirmation cancelled: %s", reason)
            self.scale_up_candidate = None

    def _tick(self):
        s, deployment = snapshot(self.api, self.args)
        target, reason = decide(s.replicas, s.idle_nodes, s.queued_jobs, s.queued_gpus)
        LOG.info("snapshot=%s target=%s reason=%s", json.dumps(asdict(s)), target, reason)
        pending = self.state.get("pending")
        cooling_down = time.time() < self.state.get("cooldown_until", 0)
        candidate = self.scale_up_candidate
        if candidate is not None and (
                pending or not s.stable or cooling_down or s.queued_jobs
                or s.replicas != candidate["previous"]
                or target < candidate["target"]):
            self.cancel_scale_up("expansion requirements no longer satisfied")
        if pending:
            if s.replicas == pending["target"] and s.stable:
                self.override_candidate = None
                self.state = {"cooldown_until": time.time() + self.args.cooldown}
                self.save()
                LOG.info("Previous scaling completed; waiting %ss for scheduling", self.args.cooldown)
            elif (s.stable and s.replicas not in (pending["target"], pending["previous"])
                  and time.time() - pending["requested_at"] >= 120):
                # A third desired count indicates an external override. Require
                # two stable observations separated by a full monitoring period
                # before adopting it; never automatically retry an ambiguous POST.
                candidate = (s.replicas, deployment.get("UpdateTimestamp"))
                if (self.override_candidate is not None
                        and self.override_candidate[0] == candidate
                        and time.monotonic() - self.override_candidate[1] >= self.args.interval):
                    LOG.warning("External replica change confirmed: saved target=%s, current=%s; "
                                "discarding old request and resuming after cooldown",
                                pending["target"], s.replicas)
                    self.state = {"cooldown_until": time.time() + self.args.cooldown}
                    self.override_candidate = None
                    self.save()
                else:
                    if self.override_candidate is None or self.override_candidate[0] != candidate:
                        self.override_candidate = (candidate, time.monotonic())
                    LOG.warning("Possible manual override: current=%s, saved target=%s; "
                                "waiting for another stable observation", s.replicas, pending["target"])
            else:
                self.override_candidate = None
                LOG.warning("Awaiting previous request target=%s; no further scaling. "
                            "If it failed or was overridden, inspect deployment and restart with "
                            "--apply --reset-pending. State: %s",
                            pending["target"], self.state_path)
            return
        if not s.stable:
            LOG.info("Deployment or GPU release is in progress; wait")
            return
        if cooling_down:
            LOG.info("Scheduling cooldown; wait")
            return
        if target == s.replicas:
            return
        if target > s.replicas:
            if self.scale_up_candidate is None:
                self.scale_up_candidate = {
                    "previous": s.replicas, "target": target,
                    "started_at": time.monotonic(),
                }
                LOG.info("Scale-up confirmation started: %s -> %s; require %ss of valid observations",
                         s.replicas, target, SCALE_UP_CONFIRM_SECONDS)
                return
            candidate = self.scale_up_candidate
            # More spare nodes are allowed, but do not enlarge this request.
            target = candidate["target"]
            elapsed = time.monotonic() - candidate["started_at"]
            if elapsed < SCALE_UP_CONFIRM_SECONDS:
                LOG.info("Scale-up confirmation: %s -> %s; %.0f/%ss elapsed",
                         s.replicas, target, elapsed, SCALE_UP_CONFIRM_SECONDS)
                return
        if not self.args.apply:
            LOG.info("DRY RUN: would scale %s -> %s", s.replicas, target)
            self.scale_up_candidate = None
            return
        latest = self.api.deployment(self.args)
        if latest != deployment:
            raise ValueError("Deployment changed before scaling; skipping cycle")
        # Persist intent before POST. A timeout may still mean the server applied
        # it; never blindly repeat a write, including after process restart.
        self.state = {"pending": {"previous": s.replicas, "target": target,
                                  "requested_at": time.time()}}
        self.save()
        self.scale_up_candidate = None
        self.api.scale(self.args, target)
        LOG.info("Submitted scale %s -> %s", s.replicas, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=platform.PROJECT)
    parser.add_argument("--quota-group", default="fdj_post_train_task")
    parser.add_argument("--deployment", default="m-20260929133556-ljgv2")
    parser.add_argument("--zone", default="cn-shanghai-a")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--cooldown", type=int, default=120,
                        help="Seconds to wait after observing completed scaling (default: 120)")
    parser.add_argument("--apply", action="store_true", help="Enable actual scaling")
    parser.add_argument("--once", action="store_true", help="Perform just one monitoring cycle")
    parser.add_argument("--reset-pending", action="store_true",
                        help="Discard saved pending request/cooldown on startup; requires --apply")
    args = parser.parse_args()
    if args.interval < 1 or args.cooldown < 0:
        parser.error("interval must be positive and cooldown nonnegative")
    if args.reset_pending and not args.apply:
        parser.error("--reset-pending requires --apply")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    state_dir = Path(__file__).resolve().parents[2] / "logs" / "yicloud-elastic-pool"
    state_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{platform.HOST}/{args.project}/{args.deployment}".encode()).hexdigest()[:20]
    # Shared lock even in dry-run, so two controllers cannot race on one deployment.
    with (state_dir / f"{key}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "Another controller is running for this deployment\n")
        platform.configure()
        # Avoid SDK retries for writes with an uncertain response.
        inferset.client.base.config.max_retries = 1
        controller = Controller(args, API(), state_dir / f"{key}.json")
        if args.reset_pending:
            controller.reset_pending()
        stop = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
        LOG.info("mode=%s interval=%ss state=%s", "APPLY" if args.apply else "DRY RUN",
                 args.interval, controller.state_path)
        while not stop.is_set():
            started = time.monotonic()
            try:
                controller.tick()
            except Exception as exc:
                LOG.error("Cycle failed; no additional action: %s", exc)
                if args.once:
                    return 1
            if args.once:
                return 0
            stop.wait(max(0, args.interval - (time.monotonic() - started)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
