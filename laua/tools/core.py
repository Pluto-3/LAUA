"""Phase 1 core tools: run_command, get_system_info, read_file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psutil

from laua.executor.pty_session import PtySession
from laua.executor.safety import check_command
from laua.tools.registry import Tool, ToolRegistry


async def _run_command(
    args: list[str],
    session: PtySession,
    confirm_fn,
    audit_fn,
    timeout: int = 30,
) -> dict[str, Any]:
    verdict = check_command(args)
    if verdict.blocked:
        return {"error": f"Blocked: {verdict.reason}", "exit_code": -1}

    if verdict.requires_confirmation:
        approved = await confirm_fn(args, requires_sudo=verdict.requires_sudo)
        if not approved:
            return {"error": "User denied execution.", "exit_code": -1}

    result = await session.run(args, timeout=timeout)
    await audit_fn(
        args=args,
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        sudo_used=verdict.requires_sudo,
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "cwd": result.cwd,
        "timed_out": result.timed_out,
    }


async def _get_system_info(
    include: list[str] | None = None,
    own_pids: list[int] | None = None,
) -> dict[str, Any]:
    include_set = set(include or ["cpu", "memory", "disk", "processes"])
    own_pids_set = set(own_pids or [os.getpid()])
    info: dict[str, Any] = {}

    if "cpu" in include_set:
        info["cpu_percent"] = psutil.cpu_percent(interval=1)
        info["cpu_count"] = psutil.cpu_count()

    if "memory" in include_set:
        mem = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": round(mem.total / 1e9, 2),
            "available_gb": round(mem.available / 1e9, 2),
            "percent": mem.percent,
        }

    if "disk" in include_set:
        disk = psutil.disk_usage("/")
        info["disk"] = {
            "total_gb": round(disk.total / 1e9, 2),
            "used_gb": round(disk.used / 1e9, 2),
            "free_gb": round(disk.free / 1e9, 2),
            "percent": disk.percent,
        }

    if "processes" in include_set:
        procs = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info_dict = proc.info
                if info_dict["pid"] in own_pids_set:
                    continue
                procs.append(info_dict)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        procs.sort(key=lambda p: p.get("memory_percent") or 0, reverse=True)
        info["top_processes"] = procs[:15]

    return info


async def _read_file(path: str, max_bytes: int = 65536) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {"error": f"File not found: {p}"}
    if not p.is_file():
        return {"error": f"Not a file: {p}"}
    try:
        content = p.read_bytes()[:max_bytes]
        return {"path": str(p), "content": content.decode(errors="replace"), "truncated": len(content) == max_bytes}
    except PermissionError:
        return {"error": f"Permission denied: {p}"}


def register_core_tools(
    registry: ToolRegistry,
    session: PtySession,
    confirm_fn,
    audit_fn,
    own_pids: list[int] | None = None,
) -> None:
    registry.register(Tool(
        name="run_command",
        description="Execute a shell command via argument array. Returns stdout, stderr, and exit code.",
        parameters_schema={
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command and arguments as a list. Example: ['ls', '-la', '/tmp']",
                    "minItems": 1,
                },
                "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds."},
            },
            "required": ["args"],
        },
        handler=lambda args, timeout=30: _run_command(args, session, confirm_fn, audit_fn, timeout),
    ))

    registry.register(Tool(
        name="get_system_info",
        description="Get current system resource usage: CPU, memory, disk, and top processes.",
        parameters_schema={
            "type": "object",
            "properties": {
                "include": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["cpu", "memory", "disk", "processes"]},
                    "description": "Subset of metrics to include. Defaults to all.",
                },
            },
        },
        handler=lambda include=None: _get_system_info(include=include, own_pids=own_pids),
    ))

    registry.register(Tool(
        name="read_file",
        description="Read the contents of a file. Returns the text content.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or home-relative file path."},
                "max_bytes": {"type": "integer", "default": 65536},
            },
            "required": ["path"],
        },
        handler=_read_file,
    ))
