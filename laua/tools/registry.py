"""Tool registry — validates JSON schema on every call before dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

import jsonschema

logger = logging.getLogger(__name__)


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

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name!r}")

        try:
            jsonschema.validate(arguments, tool.parameters_schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"Invalid arguments for {name!r}: {exc.message}") from exc

        return await tool.handler(**arguments)
