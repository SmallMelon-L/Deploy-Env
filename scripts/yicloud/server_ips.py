#!/root/Deploy-Env/.venv/bin/python3
"""Print every replica's rank0 instance IP for matching model deployments.

    server_ips.py DeepSeek-V4-Flash-0731
    server_ips.py Kimi-K3

Deployment name may be any case; every exact name match is included. Project
defaults to fdj-infra (override with YICLOUD_PROJECT). One aligned
"replica  ip" line is printed per replica; if a replica has several instances,
the rank0 (leader) instance's IP is used.
Authentication reads YICLOUD_PUBLIC_KEY and YICLOUD_SECRET_KEY from the environment.
"""

from __future__ import annotations

import os
import sys

import requests
from yicloud import base
from yicloud.base import msgs
from yicloud.services import inferset

HOST = "https://gate.yicloud.com.cn"
PROJECT = os.environ.get("YICLOUD_PROJECT", "fdj-infra")


def configure() -> None:
    names = ("YICLOUD_PUBLIC_KEY", "YICLOUD_SECRET_KEY")
    credentials = [os.environ.get(name, "").strip() for name in names]
    missing = [name for name, value in zip(names, credentials) if not value]
    if missing:
        raise ValueError("Missing required environment variables: " + ", ".join(missing))
    cfg = base.new_config()
    cfg.host, cfg.region_id, cfg.timeout = HOST, "cn-shanghai", 60.0
    session = requests.Session()
    session.trust_env = False
    inferset.use_client(
        base.new_client(
            base.with_config(cfg),
            base.with_credential(
                base.auth.Credential(
                    credentials[0],
                    credentials[1],
                )
            ),
            base.with_http_client(session),
        )
    )


def call(path: str, req) -> dict:
    """Raw "Data" payload; the SDK's typed models for these list endpoints are
    aliased to an unrelated dataclass and come back empty."""
    ctx, _ = msgs.new_meta_ctx()
    rsp = msgs.Rsp[inferset.ListInferSetData]()
    inferset.client.base.get(ctx, f"/inferset/v1alpha1/{path}", req, rsp)
    return rsp.data or {}


def list_all(path: str, request_type, **kwargs) -> list[dict]:
    """Read every page and reject a partial or changing result set."""
    rows: list[dict] = []
    total: int | None = None
    while True:
        payload = call(path, request_type(Limit=1000, Offset=len(rows), **kwargs))
        page, count = payload.get("Data"), payload.get("Total")
        if not isinstance(page, list) or type(count) is not int or count < 0:
            raise ValueError(f"Invalid {path} response")
        if total is not None and count != total:
            raise ValueError(f"{path} changed during pagination; retry")
        total = count
        rows.extend(page)
        if len(rows) == total:
            ids = [row["Id"] for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate IDs in {path} response")
            return rows
        if not page or len(rows) > total:
            raise ValueError(f"Incomplete {path} response")


def find_deployments(name: str, project: str) -> list[dict]:
    """Return every deployment whose name is an exact case-insensitive match."""
    deployments = list_all(
        "ListInferSet",
        inferset.ListInferSetReq,
        Project=project,
        Keyword=name,
    )
    return sorted(
        (d for d in deployments if d["Name"].casefold() == name.casefold()),
        key=lambda d: d["Id"],
    )


def replica_groups(deployment_id: str, project: str) -> dict[str, list[dict]]:
    """Return all instances grouped by replica for one deployment."""
    groups: dict[str, list[dict]] = {}
    for instance in list_all(
        "ListInferSetReplica",
        inferset.ListInferSetReplicaReq,
        ID=deployment_id,
        Project=project,
    ):
        groups.setdefault(instance["Replica"], []).append(instance)
    return groups


def rank0_instance(replica: str, instances: list[dict]) -> dict:
    """Select exactly one rank0/leader without silently guessing."""
    leaders = [instance for instance in instances if instance["Id"] == replica]
    if not leaders:
        leaders = [
            instance for instance in instances
            if instance.get("ReplicaRole") == "leader"
        ]
    if not leaders and len(instances) == 1:
        leaders = instances
    if len(leaders) != 1:
        raise ValueError(f"Cannot determine rank0 for replica {replica}")
    return leaders[0]


def main(name: str) -> int:
    configure()
    deployments = find_deployments(name, PROJECT)
    if not deployments:
        sys.exit(f"no deployment named {name!r}")

    rows: list[tuple[str, dict]] = []
    summaries: list[tuple[dict, int]] = []
    for deployment in deployments:
        groups = replica_groups(deployment["Id"], PROJECT)
        summaries.append((deployment, len(groups)))
        rows.extend(
            (replica, rank0_instance(replica, instances))
            for replica, instances in groups.items()
        )

    group_width = max((len(replica) for replica, _ in rows), default=0)
    for replica, rank0 in sorted(rows):
        print(f"{replica:<{group_width}}  {rank0['Ip'] or '-'}")
    for deployment, group_count in summaries:
        print(
            f"{deployment['Name']} [{deployment['Id']}]: "
            f"status={deployment['Status']} "
            f"replicas={deployment.get('ReadyReplicas')}/"
            f"{deployment.get('TotalReplicas')} groups={group_count}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(__doc__ if len(sys.argv) < 2 else main(sys.argv[1]))
