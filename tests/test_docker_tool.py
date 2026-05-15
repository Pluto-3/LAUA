"""Tests for Docker tool — mock docker SDK, Ollama guard, container ops."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from laua.tools.docker_tool import (
    OllamaGuardError,
    _is_ollama_container,
    _list_containers,
    _get_container_logs,
    _start_container,
    _stop_container,
    _restart_container,
    _list_ollama_models,
    register_docker_tools,
)
from laua.tools.registry import ToolRegistry


def _make_container(name: str, image_tag: str = "ubuntu:22.04", status: str = "running") -> MagicMock:
    c = MagicMock()
    c.name = name
    c.short_id = name[:12]
    c.status = status
    c.image.tags = [image_tag]
    c.image.short_id = image_tag[:12]
    return c


# ── _is_ollama_container ──────────────────────────────────────────────────────

def test_is_ollama_container_by_name():
    c = _make_container("ollama-server")
    assert _is_ollama_container(c) is True


def test_is_ollama_container_by_image_tag():
    c = _make_container("mycontainer", image_tag="ollama/ollama:latest")
    assert _is_ollama_container(c) is True


def test_is_not_ollama_container():
    c = _make_container("nginx", image_tag="nginx:alpine")
    assert _is_ollama_container(c) is False


# ── list_containers ────────────────────────────────────────────────────────────

async def test_list_containers_returns_list():
    c1 = _make_container("web", "nginx:alpine")
    c1.stats.return_value = {
        "cpu_stats": {"cpu_usage": {"total_usage": 200}, "system_cpu_usage": 1000, "online_cpus": 2},
        "precpu_stats": {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 900},
        "memory_stats": {"usage": 52428800},
    }
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_client.containers.list.return_value = [c1]
        mock_docker.return_value = mock_client
        result = await _list_containers(include_stopped=False)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["name"] == "web"
    assert "cpu_percent" in result[0]
    assert "memory_mb" in result[0]


async def test_list_containers_docker_unavailable():
    with patch("docker.from_env", side_effect=Exception("Docker not found")):
        result = await _list_containers()
    assert len(result) == 1
    assert "error" in result[0]


# ── get_container_logs ─────────────────────────────────────────────────────────

async def test_get_container_logs_success():
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.name = "web"
        mock_container.logs.return_value = b"line1\nline2\n"
        mock_client.containers.get.return_value = mock_container
        mock_docker.return_value = mock_client
        result = await _get_container_logs("web", lines=10)
    assert result["container"] == "web"
    assert "line1" in result["logs"]


async def test_get_container_logs_not_found():
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_client.containers.get.side_effect = Exception("404 Not Found")
        mock_docker.return_value = mock_client
        result = await _get_container_logs("nonexistent")
    assert "error" in result


# ── start_container ────────────────────────────────────────────────────────────

async def test_start_container_approved():
    confirm_fn = AsyncMock(return_value=True)
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.name = "web"
        mock_client.containers.get.return_value = mock_container
        mock_docker.return_value = mock_client
        result = await _start_container("web", confirm_fn)
    assert result["status"] == "started"
    confirm_fn.assert_awaited_once()


async def test_start_container_denied():
    confirm_fn = AsyncMock(return_value=False)
    result = await _start_container("web", confirm_fn)
    assert result["status"] == "blocked"


# ── stop_container ────────────────────────────────────────────────────────────

async def test_stop_container_regular_approved():
    confirm_fn = AsyncMock(return_value=True)
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_container = _make_container("web", "nginx:alpine")
        mock_client.containers.get.return_value = mock_container
        mock_docker.return_value = mock_client
        result = await _stop_container("web", confirm_fn)
    assert result["status"] == "stopped"
    confirm_fn.assert_awaited_once()


async def test_stop_container_regular_denied():
    confirm_fn = AsyncMock(return_value=False)
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_container = _make_container("web", "nginx:alpine")
        mock_client.containers.get.return_value = mock_container
        mock_docker.return_value = mock_client
        result = await _stop_container("web", confirm_fn)
    assert result["status"] == "blocked"


async def test_stop_ollama_container_denied_blocks():
    """Ollama guard: if confirm_fn returns False, stop is blocked."""
    confirm_fn = AsyncMock(return_value=False)
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_container = _make_container("ollama-server", "ollama/ollama:latest")
        mock_client.containers.get.return_value = mock_container
        mock_docker.return_value = mock_client
        result = await _stop_container("ollama-server", confirm_fn)
    assert result["status"] == "blocked"
    # Confirm should have been called with prominent STOP OLLAMA args
    call_args = confirm_fn.call_args[0][0]
    assert "STOP OLLAMA CONTAINER" in call_args


async def test_stop_ollama_container_approved():
    """Ollama guard: explicit confirmation allows the stop."""
    confirm_fn = AsyncMock(return_value=True)
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_container = _make_container("ollama-server", "ollama/ollama:latest")
        mock_client.containers.get.return_value = mock_container
        mock_docker.return_value = mock_client
        result = await _stop_container("ollama-server", confirm_fn)
    assert result["status"] == "stopped"


# ── restart_container ─────────────────────────────────────────────────────────

async def test_restart_container_approved():
    confirm_fn = AsyncMock(return_value=True)
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.name = "web"
        mock_client.containers.get.return_value = mock_container
        mock_docker.return_value = mock_client
        result = await _restart_container("web", confirm_fn)
    assert result["status"] == "restarted"


async def test_restart_container_denied():
    confirm_fn = AsyncMock(return_value=False)
    result = await _restart_container("web", confirm_fn)
    assert result["status"] == "blocked"


# ── list_ollama_models ─────────────────────────────────────────────────────────

async def test_list_ollama_models_success():
    mock_ollama = AsyncMock()
    mock_ollama.list_models.return_value = [
        {"name": "qwen2.5:7b", "size": 4_000_000_000},
        {"name": "llama3:8b", "size": 6_000_000_000},
    ]
    result = await _list_ollama_models(mock_ollama)
    assert len(result["models"]) == 2
    assert result["models"][0]["name"] == "qwen2.5:7b"
    assert result["models"][0]["size_gb"] > 0


async def test_list_ollama_models_error():
    mock_ollama = AsyncMock()
    mock_ollama.list_models.side_effect = Exception("Ollama down")
    result = await _list_ollama_models(mock_ollama)
    assert "error" in result


# ── register_docker_tools ──────────────────────────────────────────────────────

def test_register_docker_tools_registers_all():
    registry = ToolRegistry()
    mock_ollama = MagicMock()
    confirm_fn = AsyncMock()
    register_docker_tools(registry, mock_ollama, confirm_fn)
    tool_names = [t.name for t in registry._tools.values()]
    assert "list_containers" in tool_names
    assert "get_container_logs" in tool_names
    assert "start_container" in tool_names
    assert "stop_container" in tool_names
    assert "restart_container" in tool_names
    assert "list_ollama_models" in tool_names


def test_docker_tool_schemas_have_additional_properties_false():
    registry = ToolRegistry()
    mock_ollama = MagicMock()
    confirm_fn = AsyncMock()
    register_docker_tools(registry, mock_ollama, confirm_fn)
    for tool in registry._tools.values():
        assert tool.parameters_schema.get("additionalProperties") is False, (
            f"Tool {tool.name} is missing additionalProperties: false"
        )
