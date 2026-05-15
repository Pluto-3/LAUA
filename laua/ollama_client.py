"""HTTP client for Ollama running in Docker. Never calls the ollama CLI."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class OllamaUnavailableError(Exception):
    pass


class OllamaClient:
    def __init__(self, base_url: str, request_timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = request_timeout
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(f"Ollama unreachable: {exc}") from exc

    async def chat_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Single (non-streaming) chat call that expects a tool_call response."""
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }
        try:
            resp = await self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(f"Ollama request failed: {exc}") from exc

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Streaming plain-text chat for final response rendering."""
        payload = {"model": model, "messages": messages, "stream": True}
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload, timeout=self._timeout
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        if content := data.get("message", {}).get("content"):
                            yield content
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(f"Ollama stream failed: {exc}") from exc

    async def close(self) -> None:
        await self._client.aclose()
