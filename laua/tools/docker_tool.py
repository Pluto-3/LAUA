"""Docker controller tool — list/start/stop/restart containers with Ollama guard."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from laua.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


class OllamaGuardError(Exception):
    """Raised when stopping an Ollama container is attempted without explicit confirmation."""


def _is_ollama_container(container: Any) -> bool:
    """Return True if the container looks like an Ollama container."""
    name_match = "ollama" in container.name.lower()
    tag_match = (
        bool(container.image.tags)
        and "ollama" in container.image.tags[0].lower()
    )
    return name_match or tag_match


def _container_stats_blocking(container: Any) -> dict[str, Any]:
    """Blocking call to container.stats(stream=False) — run in executor."""
    try:
        stats = container.stats(stream=False)
        # CPU percent calculation
        cpu_delta = (
            stats["cpu_stats"]["cpu_usage"]["total_usage"]
            - stats["precpu_stats"]["cpu_usage"]["total_usage"]
        )
        system_delta = (
            stats["cpu_stats"]["system_cpu_usage"]
            - stats["precpu_stats"].get("system_cpu_usage", 0)
        )
        num_cpus = stats["cpu_stats"].get("online_cpus") or len(
            stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1])
        )
        cpu_percent = (cpu_delta / system_delta) * num_cpus * 100.0 if system_delta > 0 else 0.0

        # Memory usage
        mem_usage = stats["memory_stats"].get("usage", 0)
        memory_mb = round(mem_usage / (1024 * 1024), 2)

        return {"cpu_percent": round(cpu_percent, 2), "memory_mb": memory_mb}
    except Exception as exc:
        logger.debug("Could not get container stats: %s", exc)
        return {"cpu_percent": 0.0, "memory_mb": 0.0}


async def _list_containers(include_stopped: bool = False) -> list[dict[str, Any]]:
    loop = asyncio.get_event_loop()
    try:
        import docker  # type: ignore[import-untyped]
        client = await loop.run_in_executor(None, docker.from_env)
    except Exception as exc:
        return [{"error": f"Docker unavailable: {exc}"}]


    def _fetch() -> list[dict[str, Any]]:
        containers = client.containers.list(all=bool(include_stopped))
        result = []
        for c in containers:
            stats = _container_stats_blocking(c) if c.status == "running" else {
                "cpu_percent": 0.0, "memory_mb": 0.0
            }
            result.append({
                "id": c.short_id,
                "name": c.name,
                "image": c.image.tags[0] if c.image.tags else c.image.short_id,
                "status": c.status,
                "cpu_percent": stats["cpu_percent"],
                "memory_mb": stats["memory_mb"],
            })
        return result

    return await loop.run_in_executor(None, _fetch)


async def _get_container_logs(container_id: str, lines: int = 50) -> dict[str, Any]:
    loop = asyncio.get_event_loop()
    try:
        import docker  # type: ignore[import-untyped]
        client = await loop.run_in_executor(None, docker.from_env)
    except Exception as exc:
        return {"error": f"Docker unavailable: {exc}"}

    def _fetch() -> dict[str, Any]:
        try:
            container = client.containers.get(container_id)
            logs = container.logs(tail=lines, stream=False)
            return {"container": container.name, "logs": logs.decode(errors="replace")}
        except Exception as exc:
            return {"error": str(exc)}

    return await loop.run_in_executor(None, _fetch)


async def _start_container(container_id: str, confirm_fn: Callable) -> dict[str, Any]:
    approved = await confirm_fn(["docker", "start", container_id])
    if not approved:
        return {"status": "blocked", "reason": "User denied start."}

    loop = asyncio.get_event_loop()
    try:
        import docker  # type: ignore[import-untyped]
        client = await loop.run_in_executor(None, docker.from_env)
    except Exception as exc:
        return {"error": f"Docker unavailable: {exc}"}

    def _do() -> dict[str, Any]:
        try:
            container = client.containers.get(container_id)
            container.start()
            return {"status": "started", "container": container.name}
        except Exception as exc:
            return {"error": str(exc)}

    return await loop.run_in_executor(None, _do)


async def _stop_container(container_id: str, confirm_fn: Callable) -> dict[str, Any]:
    loop = asyncio.get_event_loop()
    try:
        import docker  # type: ignore[import-untyped]
        client = await loop.run_in_executor(None, docker.from_env)
    except Exception as exc:
        return {"error": f"Docker unavailable: {exc}"}

    def _get_container() -> Any:
        return client.containers.get(container_id)

    try:
        container = await loop.run_in_executor(None, _get_container)
    except Exception as exc:
        return {"error": str(exc)}

    # Ollama guard: require explicit container name confirmation
    if _is_ollama_container(container):
        approved = await confirm_fn(
            ["STOP OLLAMA CONTAINER", container.name],
            requires_sudo=False,
        )
        if not approved:
            return {"status": "blocked", "reason": "Ollama container stop denied by user."}
    else:
        approved = await confirm_fn(["docker", "stop", container.name])
        if not approved:
            return {"status": "blocked", "reason": "User denied stop."}

    def _do() -> dict[str, Any]:
        try:
            container.stop()
            return {"status": "stopped", "container": container.name}
        except Exception as exc:
            return {"error": str(exc)}

    return await loop.run_in_executor(None, _do)


async def _restart_container(container_id: str, confirm_fn: Callable) -> dict[str, Any]:
    approved = await confirm_fn(["docker", "restart", container_id])
    if not approved:
        return {"status": "blocked", "reason": "User denied restart."}

    loop = asyncio.get_event_loop()
    try:
        import docker  # type: ignore[import-untyped]
        client = await loop.run_in_executor(None, docker.from_env)
    except Exception as exc:
        return {"error": f"Docker unavailable: {exc}"}

    def _do() -> dict[str, Any]:
        try:
            container = client.containers.get(container_id)
            container.restart()
            return {"status": "restarted", "container": container.name}
        except Exception as exc:
            return {"error": str(exc)}

    return await loop.run_in_executor(None, _do)


async def _list_ollama_models(ollama_client: Any) -> dict[str, Any]:
    """Query Ollama HTTP API for available models."""
    try:
        models = await ollama_client.list_models()
        return {
            "models": [
                {
                    "name": m.get("name", ""),
                    "size_gb": round(m.get("size", 0) / 1e9, 2),
                }
                for m in models
            ]
        }
    except Exception as exc:
        return {"error": str(exc)}


def register_docker_tools(
    registry: ToolRegistry,
    ollama_client: Any,
    confirm_fn: Callable,
) -> None:
    registry.register(Tool(
        name="list_containers",
        description="List Docker containers with CPU and memory stats.",
        parameters_schema={
            "type": "object",
            "properties": {
                "include_stopped": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, include stopped containers.",
                },
            },
            "additionalProperties": False,
        },
        handler=_list_containers,
    ))

    registry.register(Tool(
        name="get_container_logs",
        description="Retrieve the last N log lines from a Docker container.",
        parameters_schema={
            "type": "object",
            "properties": {
                "container_id": {"type": "string", "description": "Container ID or name."},
                "lines": {"type": "integer", "default": 50, "description": "Number of log lines."},
            },
            "required": ["container_id"],
            "additionalProperties": False,
        },
        handler=_get_container_logs,
    ))

    registry.register(Tool(
        name="start_container",
        description="Start a stopped Docker container. Requires confirmation.",
        parameters_schema={
            "type": "object",
            "properties": {
                "container_id": {"type": "string", "description": "Container ID or name."},
            },
            "required": ["container_id"],
            "additionalProperties": False,
        },
        handler=lambda container_id: _start_container(container_id, confirm_fn),
    ))

    registry.register(Tool(
        name="stop_container",
        description=(
            "Stop a running Docker container. Requires confirmation. "
            "Ollama containers require typing the container name explicitly."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "container_id": {"type": "string", "description": "Container ID or name."},
            },
            "required": ["container_id"],
            "additionalProperties": False,
        },
        handler=lambda container_id: _stop_container(container_id, confirm_fn),
    ))

    registry.register(Tool(
        name="restart_container",
        description="Restart a Docker container. Requires confirmation.",
        parameters_schema={
            "type": "object",
            "properties": {
                "container_id": {"type": "string", "description": "Container ID or name."},
            },
            "required": ["container_id"],
            "additionalProperties": False,
        },
        handler=lambda container_id: _restart_container(container_id, confirm_fn),
    ))

    registry.register(Tool(
        name="list_ollama_models",
        description="List models available in Ollama with their sizes.",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=lambda: _list_ollama_models(ollama_client),
    ))
