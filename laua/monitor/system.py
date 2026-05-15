"""System monitor — psutil with self-PID and Ollama container PID filtering."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class SystemSnapshot:
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    top_processes: list[dict[str, Any]]


def _get_ollama_pids() -> set[int]:
    """Find PIDs associated with the Ollama Docker container process tree."""
    pids: set[int] = set()
    try:
        import docker
        client = docker.from_env()
        for container in client.containers.list():
            name_match = "ollama" in container.name.lower()
            tag_match = bool(container.image.tags) and "ollama" in container.image.tags[0].lower()
            if name_match or tag_match:
                top = container.top()
                for proc in top.get("Processes", []):
                    try:
                        pids.add(int(proc[1]))  # PID column
                    except (IndexError, ValueError):
                        pass
    except Exception:
        # Docker may not be available or accessible
        pass
    return pids


class SystemMonitor:
    def __init__(self, refresh_interval: float = 5.0) -> None:
        self._interval = refresh_interval
        self._own_pid = os.getpid()
        self._excluded_pids: set[int] = set()
        self._running = False

    async def refresh_excluded_pids(self) -> None:
        loop = asyncio.get_event_loop()
        self._excluded_pids = await loop.run_in_executor(None, _get_ollama_pids)
        self._excluded_pids.add(self._own_pid)

    def snapshot(self) -> SystemSnapshot:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        procs = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
            try:
                info = proc.info
                if info["pid"] in self._excluded_pids:
                    continue
                procs.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        procs.sort(key=lambda p: p.get("memory_percent") or 0, reverse=True)

        return SystemSnapshot(
            cpu_percent=cpu,
            memory_percent=mem.percent,
            memory_available_gb=round(mem.available / 1e9, 2),
            disk_percent=disk.percent,
            disk_free_gb=round(disk.free / 1e9, 2),
            top_processes=procs[:10],
        )
