"""Orchestration helpers: run a BRAND graph end-to-end from Python.

Used by the notebook and the integration tests. Mirrors the documented BRAND
session workflow: start `supervisor` (which starts redis-server itself), send
`startGraph` on `supervisor_ipstream`, watch the output stream, and send
`stopGraph` when done.
"""
import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import redis
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BRAND_DIR = REPO_ROOT / "third_party" / "brand"
SUPERVISOR = BRAND_DIR / "supervisor" / "supervisor.py"
# Portable redis config: BRAND's stock one needs root for /var/run paths.
REDIS_CONF = REPO_ROOT / "config" / "redis.supervisor.conf"


class SupervisorHandle:
    """A running supervisor + redis-server pair."""

    def __init__(self, proc: subprocess.Popen, host: str, port: int,
                 log_path: Path):
        self.proc = proc
        self.host = host
        self.port = port
        self.log_path = log_path
        self.redis = redis.Redis(host, port)

    def start_graph(self, graph: dict) -> None:
        self.redis.xadd("supervisor_ipstream", {
            "commands": "startGraph",
            "graph": json.dumps(graph),
        })

    def stop_graph(self) -> None:
        self.redis.xadd("supervisor_ipstream", {"commands": "stopGraph"})

    def graph_status(self) -> Optional[str]:
        entries = self.redis.xrevrange("graph_status", "+", "-", count=1)
        return entries[0][1][b"status"].decode() if entries else None

    def wait_for_status(self, status: str, timeout_s: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.graph_status() == status:
                return
            time.sleep(0.2)
        raise TimeoutError(
            f"graph status never reached {status!r} within {timeout_s:.0f}s "
            f"(last: {self.graph_status()!r}); see {self.log_path}")

    def shutdown(self) -> None:
        try:
            self.stop_graph()
            time.sleep(1.0)
        except redis.exceptions.ConnectionError:
            pass
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        # If the redis-server child outlived the supervisor (e.g. the
        # supervisor crashed), stop it too so the port is freed.
        try:
            self.redis.shutdown(nosave=True)
        except redis.exceptions.ConnectionError:
            pass  # already gone: the normal case


def start_supervisor(port: int = 28100, data_dir: Optional[Path] = None,
                     log_path: Optional[Path] = None,
                     timeout_s: float = 20.0) -> SupervisorHandle:
    """Launch supervisor from the BRAND core dir and wait for Redis."""
    data_dir = data_dir or (REPO_ROOT / "runs" / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path or (REPO_ROOT / "runs" / f"supervisor_{port}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Fail fast if the port is occupied (e.g. an orphaned redis-server from
    # an earlier run). Otherwise the readiness check below could read stale
    # state from the wrong server.
    probe = redis.Redis("127.0.0.1", port, socket_connect_timeout=0.5)
    try:
        probe.ping()
        raise RuntimeError(
            f"port {port} already in use; stop the old server first "
            f"(redis-cli -p {port} shutdown nosave)")
    except redis.exceptions.ConnectionError:
        pass  # port free, as expected

    # Run the supervisor with this interpreter's environment, and make sure
    # the node shebangs (#!/usr/bin/env python3) resolve to it too.
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, str(SUPERVISOR),
         "-i", "127.0.0.1", "-p", str(port), "-d", str(data_dir),
         "-c", str(REDIS_CONF)],
        cwd=BRAND_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    handle = SupervisorHandle(proc, "127.0.0.1", port, log_path)

    # The supervisor reads supervisor_ipstream starting from '$', so commands
    # sent before its listen loop begins are silently dropped. Wait for the
    # "Listening for commands" status, not merely for Redis to answer.
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"supervisor exited early (code {proc.returncode}); "
                f"see {log_path}")
        try:
            statuses = handle.redis.xrange("supervisor_status", "-", "+")
            if any(e[b"status"] == b"Listening for commands"
                   for _, e in statuses):
                return handle
        except redis.exceptions.ConnectionError:
            pass
        time.sleep(0.2)
    proc.kill()
    raise TimeoutError(f"supervisor not listening after {timeout_s:.0f}s; "
                       f"see {log_path}")


def load_graph(wav_file: str, graph_path: Optional[Path] = None,
               **overrides) -> dict:
    """Load the canonical graph YAML and override per-run parameters.

    `overrides` maps node nickname to a dict of parameter updates, e.g.
    `speech_to_text={"model": "small.en"}`.
    """
    graph_path = graph_path or (
        REPO_ROOT / "graphs" / "speech_pipeline" / "speech_pipeline.yaml")
    with open(graph_path) as f:
        graph = yaml.safe_load(f)
    graph = copy.deepcopy(graph)
    graph["graph_name"] = graph_path.stem
    for node in graph["nodes"]:
        if node["nickname"] == "audio_publisher":
            node["parameters"]["wav_file"] = str(wav_file)
        if node["nickname"] in overrides:
            node["parameters"].update(overrides[node["nickname"]])
    return graph


def decode_entry(entry: dict) -> dict:
    """Decode a Redis stream entry's bytes keys/values to str (data kept raw)."""
    out = {}
    for k, v in entry.items():
        key = k.decode()
        out[key] = v if key == "data" else v.decode()
    return out
