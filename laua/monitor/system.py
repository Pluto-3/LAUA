"""System monitor — async-safe, self/Ollama PID filtering, GPU, thermals, anomaly detection."""

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
    gpu: dict[str, Any] | None = None
    thermals: dict[str, Any] | None = None


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
                top_data = container.top()
                titles = top_data.get("Titles", [])
                pid_idx = titles.index("PID") if "PID" in titles else 1
                for proc in top_data.get("Processes", []):
                    try:
                        pids.add(int(proc[pid_idx]))
                    except (IndexError, ValueError):
                        pass
    except Exception:
        pass
    return pids


def _snapshot_blocking(excluded_pids: set[int]) -> SystemSnapshot:
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    procs = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
        try:
            info = proc.info
            if info["pid"] in excluded_pids:
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


def _gpu_info_blocking() -> dict[str, Any] | None:
    """Try pynvml first, fall back to nvidia-smi subprocess."""
    try:
        import pynvml  # type: ignore[import-untyped]
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                "index": i,
                "utilization_percent": util.gpu,
                "memory_used_mb": round(mem.used / 1e6, 1),
                "memory_total_mb": round(mem.total / 1e6, 1),
            })
        return {"gpus": gpus}
    except Exception:
        pass

    # Fall back to nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpus = []
            for i, line in enumerate(result.stdout.strip().splitlines()):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "index": i,
                        "utilization_percent": float(parts[0]),
                        "memory_used_mb": float(parts[1]),
                        "memory_total_mb": float(parts[2]),
                    })
            return {"gpus": gpus}
    except Exception:
        pass

    return None


def _thermal_info_blocking() -> dict[str, Any] | None:
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        result: dict[str, list[dict[str, Any]]] = {}
        for sensor_name, entries in temps.items():
            result[sensor_name] = [
                {"label": e.label or "core", "current": e.current,
                 "high": e.high, "critical": e.critical}
                for e in entries
            ]
        return result
    except (AttributeError, Exception):
        return None


def detect_anomalies(
    snapshot: SystemSnapshot,
    cpu_threshold: float = 90.0,
    memory_threshold: float = 90.0,
    disk_threshold: float = 90.0,
) -> list[str]:
    """Return a list of natural-language warning strings for abnormal resource usage."""
    warnings: list[str] = []
    if snapshot.cpu_percent >= cpu_threshold:
        warnings.append(
            f"CPU usage is {snapshot.cpu_percent:.0f}% — consider killing heavy processes."
        )
    if snapshot.memory_percent >= memory_threshold:
        warnings.append(
            f"Memory usage is {snapshot.memory_percent:.0f}% "
            f"({snapshot.memory_available_gb:.1f} GB available) — consider freeing RAM."
        )
    if snapshot.disk_percent >= disk_threshold:
        warnings.append(
            f"Disk usage is {snapshot.disk_percent:.0f}% "
            f"({snapshot.disk_free_gb:.1f} GB free) — consider cleaning up large files."
        )
    if snapshot.gpu:
        for gpu in snapshot.gpu.get("gpus", []):
            if gpu.get("utilization_percent", 0) >= cpu_threshold:
                warnings.append(
                    f"GPU {gpu['index']} utilization is {gpu['utilization_percent']:.0f}%."
                )
    return warnings


class SystemMonitor:
    def __init__(
        self,
        refresh_interval: float = 5.0,
        gpu_enabled: bool = True,
        thermal_enabled: bool = True,
    ) -> None:
        self._interval = refresh_interval
        self._gpu_enabled = gpu_enabled
        self._thermal_enabled = thermal_enabled
        self._own_pid = os.getpid()
        self._excluded_pids: set[int] = set()

    async def refresh_excluded_pids(self) -> None:
        loop = asyncio.get_event_loop()
        self._excluded_pids = await loop.run_in_executor(None, _get_ollama_pids)
        self._excluded_pids.add(self._own_pid)

    async def snapshot(self) -> SystemSnapshot:
        """Return a full system snapshot. All blocking calls run in executor."""
        loop = asyncio.get_event_loop()
        excluded = set(self._excluded_pids)

        gpu_task = (
            loop.run_in_executor(None, _gpu_info_blocking)
            if self._gpu_enabled else asyncio.sleep(0)
        )
        thermal_task = (
            loop.run_in_executor(None, _thermal_info_blocking)
            if self._thermal_enabled else asyncio.sleep(0)
        )
        snap, gpu, thermals = await asyncio.gather(
            loop.run_in_executor(None, _snapshot_blocking, excluded),
            gpu_task,
            thermal_task,
        )

        snap.gpu = gpu if self._gpu_enabled else None
        snap.thermals = thermals if self._thermal_enabled else None
        return snap
