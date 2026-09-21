#!/usr/bin/env python3
"""Reconcile one router with all Yicloud deployments sharing an exact name.

Run with the repository .venv Python. Defaults: 10-second interval, worker port
5050, project from YICLOUD_PROJECT (fdj-infra). See README.md in this directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ipaddress
import json
import logging
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
from urllib.parse import quote, urlsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yicloud"))
import server_ips as platform

LOG = logging.getLogger("sync_workers")
DP_SUFFIX = re.compile(r"@\d+$")


def worker_ready(base, timeout):
    """Require a completed streaming hello response before adding a new worker."""
    started = time.monotonic()
    # Keep router credentials out of requests to backend workers.
    with requests.Session() as session:
        session.trust_env = False
        try:
            with session.post(
                base + "/v1/chat/completions",
                json={"temperature": 1.0, "stream": True,
                      "messages": [{"role": "user", "content": "hello"}]},
                stream=True, timeout=timeout, allow_redirects=True,
            ) as response:
                response.raise_for_status()
                has_content = finished = False
                # A small chunk avoids buffering an idle/incomplete SSE stream.
                for line in response.iter_lines(chunk_size=1):
                    if time.monotonic() - started > timeout:
                        raise ValueError("inference probe exceeded time limit")
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        if not (has_content and finished):
                            raise ValueError("stream finished without a complete answer")
                        return True
                    event = json.loads(data)
                    if not isinstance(event, dict) or event.get("error"):
                        raise ValueError("inference stream returned an error")
                    for choice in event.get("choices", []):
                        content = choice.get("delta", {}).get("content")
                        if isinstance(content, str) and content.strip():
                            has_content = True
                        if choice.get("finish_reason") in ("stop", "length"):
                            finished = True
                raise ValueError("inference stream ended without [DONE]")
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            return False


def platform_ips(name, project):
    deployments = platform.find_deployments(name, project)
    if not deployments:
        raise ValueError(f"No deployment named {name!r}")
    ips = set()
    for deployment in deployments:
        groups = platform.replica_groups(deployment["Id"], project)
        for replica, instances in groups.items():
            ip = platform.rank0_instance(replica, instances)["Ip"]
            if not ip or ip == "-":
                continue
            ips.add(str(ipaddress.ip_address(ip)))
    return ips


def worker_groups(payload):
    workers = payload["workers"]
    if not isinstance(workers, list) or payload["total"] != len(workers):
        raise ValueError("Incomplete router worker response")
    groups = {}
    for worker in workers:
        base = DP_SUFFIX.sub("", worker["url"])
        parsed = urlsplit(base)
        if parsed.scheme not in ("http", "https") or not worker["id"]:
            raise ValueError(f"Invalid worker: {base}")
        ip = str(ipaddress.ip_address(parsed.hostname))
        groups.setdefault(ip, {}).setdefault(base, []).append(worker)
    return groups


class Router:
    def __init__(self, url, timeout):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False
        if key := os.environ.get("SGLANG_ROUTER_API_KEY"):
            self.session.headers["Authorization"] = f"Bearer {key}"
        self.base_ids = {}
        # Suppress repeated asynchronous submissions until observed or timed out.
        self.pending = {}

    def request(self, method, path, **kwargs):
        response = self.session.request(method, self.url + path, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

    def register(self, base):
        result = self.request("POST", "/workers", json={"url": base})
        self.base_ids[base] = result["worker_id"]
        return result["worker_id"]

    def remove(self, base, workers):
        if any(DP_SUFFIX.search(w["url"]) for w in workers):
            worker_id = self.base_ids.get(base)
            if not worker_id:
                # Reserve a base UUID; deleting a rank UUID cannot remove DP groups.
                self.register(base)
                return False
            path = "/workers/" + quote(worker_id, safe="")
            try:
                info = self.request("GET", path)
            except requests.HTTPError as exc:
                if exc.response.status_code != 404:
                    raise
                # A completed DP registration retains its base UUID mapping,
                # but GET returns 404 because only rank workers are listed.
            else:
                status = info.get("job_status") or {}
                if status.get("status") in ("pending", "processing"):
                    return False
            try:
                self.request("DELETE", path)
            except requests.HTTPError as exc:
                if exc.response.status_code != 404:
                    raise
                self.base_ids.pop(base, None)
                return False
        else:
            for worker in workers:
                self.request("DELETE", "/workers/" + quote(worker["id"], safe=""))
        return True

    def reconcile(self, desired, port, retry_after, dry_run=False, probe_timeout=10):
        groups = worker_groups(self.request("GET", "/workers"))
        actual = set(groups)
        missing, remove = desired - actual, actual - desired
        add, unready = set(), set()
        if missing:
            with ThreadPoolExecutor(max_workers=32) as executor:
                probes = {}
                for ip in sorted(missing):
                    host = f"[{ip}]" if ":" in ip else ip
                    future = executor.submit(worker_ready, f"http://{host}:{port}", probe_timeout)
                    probes[future] = ip
                for future in as_completed(probes):
                    target = add if future.result() else unready
                    target.add(probes[future])
        LOG.info("platform=%s router=%s unready=%s add=%s remove=%s",
                 sorted(desired), sorted(actual), sorted(unready), sorted(add), sorted(remove))
        now = time.monotonic()
        for key, started in list(self.pending.items()):
            action, ip = key
            # Keep unobserved jobs even if desired changes: an old job may still land.
            completed = ip in actual if action == "add" else ip not in actual
            if completed or now - started >= retry_after:
                del self.pending[key]
        if dry_run:
            return not missing and not remove
        for ip in sorted(add):
            if any((action, ip) in self.pending for action in ("add", "remove")):
                continue
            host = f"[{ip}]" if ":" in ip else ip
            base = f"http://{host}:{port}"
            self.register(base)
            self.pending[("add", ip)] = time.monotonic()
        for ip in sorted(remove):
            if any((action, ip) in self.pending for action in ("add", "remove")):
                continue
            submitted = [self.remove(base, workers) for base, workers in groups[ip].items()]
            if all(submitted):
                self.pending[("remove", ip)] = time.monotonic()
        return not missing and not remove and not self.pending


def positive(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("router", help="router IP:port or http(s)://IP:port")
    parser.add_argument(
        "deployment",
        help="exact deployment name; all case-insensitive matches are included",
    )
    parser.add_argument("--project", default=platform.PROJECT)
    parser.add_argument("--worker-port", type=int, default=5050)
    parser.add_argument("--interval", type=positive, default=10)
    parser.add_argument("--request-timeout", type=positive, default=15)
    parser.add_argument("--probe-timeout", type=positive, default=10,
                        help="inference probe time limit and socket timeout in seconds")
    parser.add_argument("--pending-timeout", type=positive, default=120,
                        help="seconds before retrying an unobserved async operation")
    parser.add_argument("--once", action="store_true", help="exit after verified convergence")
    parser.add_argument("--converge-timeout", type=positive, default=180,
                        help="maximum reconciliation time with --once")
    parser.add_argument("--dry-run", action="store_true", help="query once, log diff, do not mutate")
    args = parser.parse_args(argv)
    if not 1 <= args.worker_port <= 65535:
        parser.error("--worker-port must be between 1 and 65535")
    url = args.router if "://" in args.router else "http://" + args.router
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path not in ("", "/") or parsed.query or parsed.fragment or parsed.username:
        parser.error("router must be an http(s) origin without credentials, path or query")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    platform.configure()
    router = Router(url, args.request_timeout)
    started = time.monotonic()
    try:
        while not stop.is_set():
            cycle = time.monotonic()
            try:
                desired = platform_ips(args.deployment, args.project)
                converged = router.reconcile(desired, args.worker_port, args.pending_timeout,
                                             args.dry_run, args.probe_timeout)
                if args.dry_run:
                    return 0
                if converged and args.once:
                    return 0
            except Exception:
                if args.dry_run:
                    return 1
            delay = max(0, args.interval - (time.monotonic() - cycle))
            if args.once:
                remaining = args.converge_timeout - (time.monotonic() - started)
                if remaining <= 0:
                    return 1
                delay = min(delay, remaining)
            stop.wait(delay)
        return 130
    finally:
        router.session.close()


if __name__ == "__main__":
    sys.exit(main())
