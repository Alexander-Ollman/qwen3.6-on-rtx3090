"""Thin wrapper over docker-py for the operations the orchestrator needs.

We deliberately avoid `docker compose` here because the qwen-* containers were
launched via `docker run` from launch-*.sh — they aren't part of a compose
project. We only need to stop and remove them; launching is delegated to the
launcher script (which we exec from inside this container with the host
docker socket mounted).
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import time
from typing import Iterable, Optional

import docker
from docker.errors import APIError, NotFound


log = logging.getLogger("qwen-control.docker")
_client: Optional[docker.DockerClient] = None


def client() -> docker.DockerClient:
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def container_exists(name: str) -> bool:
    try:
        client().containers.get(name)
        return True
    except NotFound:
        return False
    except APIError as e:
        log.warning("inspect %s failed: %s", name, e)
        return False


def container_running(name: str) -> bool:
    try:
        c = client().containers.get(name)
        return c.status == "running"
    except NotFound:
        return False
    except APIError:
        return False


async def stop_and_remove(names: Iterable[str], stop_timeout: int = 20) -> list[str]:
    """Stop and `rm -f` each container. Returns names that were touched."""
    touched: list[str] = []
    cli = client()
    loop = asyncio.get_running_loop()
    for n in names:
        try:
            c = await loop.run_in_executor(None, cli.containers.get, n)
        except NotFound:
            continue
        log.info("stopping %s (status=%s)", n, c.status)
        try:
            await loop.run_in_executor(None, lambda: c.stop(timeout=stop_timeout))
        except APIError as e:
            log.warning("stop %s: %s", n, e)
        try:
            await loop.run_in_executor(None, lambda: c.remove(force=True))
            touched.append(n)
        except APIError as e:
            log.warning("remove %s: %s", n, e)
    return touched


async def gpu_memory_used_mib() -> list[tuple[int, int]]:
    """Return [(gpu_index, used_mib), ...] for each visible GPU.

    Uses nvidia-smi from the host (inside the container if `docker.io`
    package was installed). Works because the qwen-control image installs
    docker.io which depends on… nothing useful for nvidia-smi. Fall back
    to running nvidia-smi via `docker run --rm --gpus all` if it's not
    available locally.
    """
    if shutil.which("nvidia-smi"):
        cmd = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    else:
        # docker.sock is mounted; we can ask the daemon for the info.
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "nvidia/cuda:12.8.0-base-ubuntu22.04",
            "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits",
        ]

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        return []

    result: list[tuple[int, int]] = []
    for line in out.decode().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                result.append((int(parts[0]), int(parts[1])))
            except ValueError:
                pass
    return result


async def run_launcher(launcher_path: str, env: Optional[dict] = None) -> tuple[int, str, str]:
    """Exec a launcher script. Returns (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/bash", launcher_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**__import__("os").environ, **(env or {})},
    )
    out, err = await proc.communicate()
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")
