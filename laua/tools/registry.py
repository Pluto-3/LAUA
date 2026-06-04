"""Tool registry — validates JSON schema on every call before dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import jsonschema

logger = logging.getLogger(__name__)

_DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({
    "run_command", "write_file", "delete_file",
    "start_container", "stop_container", "restart_container",
})


def _dry_run_result(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "run_command":
        return {
            "dry_run": True,
            "would_run": arguments.get("args", []),
            "note": "DRY RUN — command was NOT executed. Say what it would do.",
        }
    if name == "write_file":
        content = arguments.get("content") or ""
        return {
            "dry_run": True,
            "would_write": arguments.get("path", ""),
            "bytes": len(content.encode()),
            "note": "DRY RUN — file was NOT written.",
        }
    if name == "delete_file":
        return {
            "dry_run": True,
            "would_delete": arguments.get("path", ""),
            "note": "DRY RUN — file was NOT deleted.",
        }
    if name in ("start_container", "stop_container", "restart_container"):
        _past = {"start": "started", "stop": "stopped", "restart": "restarted"}
        action_past = _past[name.split("_")[0]]
        return {
            "dry_run": True,
            "would_call": name,
            "container": arguments.get("container_id", ""),
            "note": f"DRY RUN — container was NOT {action_past}.",
        }
    return {"dry_run": True, "would_call": name, "note": "DRY RUN — action was NOT executed."}


@dataclass
class Tool:
    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]

    def to_ollama_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_schemas(self) -> list[dict[str, Any]]:
        return [t.to_ollama_schema() for t in self._tools.values()]

    async def dispatch(self, name: str, arguments: dict[str, Any], dry_run: bool = False) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name!r}")

        try:
            jsonschema.validate(arguments, tool.parameters_schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"Invalid arguments for {name!r}: {exc.message}") from exc

        if dry_run and name in _DESTRUCTIVE_TOOLS:
            return _dry_run_result(name, arguments)

        return await tool.handler(**arguments)
