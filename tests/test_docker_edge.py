"""Edge-case tests for Docker tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from laua.tools.docker_tool import (
    _container_stats_blocking,
    _is_ollama_container,
    _list_containers,
    _list_ollama_models,
    _get_container_logs,
    _stop_container,
)


def _make_container(
    name: str,
    image_tags: list[str] | None = None,
    status: str = "running",
) -> MagicMock:
    c = MagicMock()
    c.name = name
    c.short_id = name[:12]
    c.status = status
    c.image.tags = image_tags if image_tags is not None else ["ubuntu:22.04"]
    c.image.short_id = "abc123"
    return c


# ── _is_ollama_container ──────────────────────────────────────────────────────

def test_is_ollama_empty_image_tags_uses_name():
    """Container with no image tags: detection falls back to name only."""
    c = _make_container("ollama", image_tags=[])
    assert _is_ollama_container(c) is True


def test_is_ollama_empty_tags_non_ollama_name():
    c = _make_container("postgres", image_tags=[])
    assert _is_ollama_container(c) is False


def test_is_ollama_case_insensitive_name():
    c = _make_container("OLLAMA-SERVER")
    assert _is_ollama_container(c) is True


def test_is_ollama_case_insensitive_tag():
    c = _make_container("mycontainer", image_tags=["OLLAMA/OLLAMA:LATEST"])
    assert _is_ollama_container(c) is True


def test_is_ollama_partial_name_match():
    """A container named 'ollama-gpu' should still be detected."""
    c = _make_container("ollama-gpu-server", image_tags=["ubuntu:22.04"])
    assert _is_ollama_container(c) is True


# ── _container_stats_blocking ─────────────────────────────────────────────────

def test_stats_zero_system_delta_no_division_error():
    """system_delta=0 should return cpu_percent=0.0, not ZeroDivisionError."""
    c = MagicMock()
    c.stats.return_value = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 500, "percpu_usage": [250, 250]},
            "system_cpu_usage": 1000,
            "online_cpus": 2,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": 500},
            "system_cpu_usage": 1000,  # same → delta=0
        },
        "memory_stats": {"usage": 10_000_000},
    }
    result = _container_stats_blocking(c)
    assert result["cpu_percent"] == 0.0


def test_stats_missing_keys_returns_zeros():
    """Malformed stats dict should not crash."""
    c = MagicMock()
    c.stats.return_value = {}  # completely empty
    result = _container_stats_blocking(c)
    assert result == {"cpu_percent": 0.0, "memory_mb": 0.0}


def test_stats_missing_memory_usage_returns_zero():
    c = MagicMock()
    c.stats.return_value = {
        "cpu_stats": {"cpu_usage": {"total_usage": 200}, "system_cpu_usage": 1000, "online_cpus": 1},
        "precpu_stats": {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 900},
        "memory_stats": {},  # no 'usage' key
    }
    result = _container_stats_blocking(c)
    assert result["memory_mb"] == 0.0


def test_stats_exception_in_container_stats_returns_zeros():
    c = MagicMock()
    c.stats.side_effect = RuntimeError("container gone")
    result = _container_stats_blocking(c)
    assert result == {"cpu_percent": 0.0, "memory_mb": 0.0}


# ── _list_containers ──────────────────────────────────────────────────────────

async def test_list_containers_stopped_container_no_stats_call():
    """Stopped containers must NOT trigger container.stats() — it would block."""
    c = _make_container("stopped-svc", status="exited")
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_client.containers.list.return_value = [c]
        mock_docker.return_value = mock_client
        result = await _list_containers(include_stopped=True)
    c.stats.assert_not_called()
    assert result[0]["cpu_percent"] == 0.0
    assert result[0]["memory_mb"] == 0.0


async def test_list_containers_no_image_tags_uses_short_id():
    c = _make_container("untagged", image_tags=[])
    c.stats.return_value = {
        "cpu_stats": {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 1000, "online_cpus": 1},
        "precpu_stats": {"cpu_usage": {"total_usage": 50}, "system_cpu_usage": 900},
        "memory_stats": {"usage": 5_000_000},
    }
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_client.containers.list.return_value = [c]
        mock_docker.return_value = mock_client
        result = await _list_containers()
    assert result[0]["image"] == "abc123"  # falls back to short_id


async def test_list_containers_empty():
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_client.containers.list.return_value = []
        mock_docker.return_value = mock_client
        result = await _list_containers()
    assert result == []


# ── _get_container_logs ───────────────────────────────────────────────────────

async def test_get_container_logs_zero_lines():
    """lines=0 should not crash — returns whatever Docker SDK returns."""
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.name = "app"
        mock_container.logs.return_value = b""
        mock_client.containers.get.return_value = mock_container
        mock_docker.return_value = mock_client
        result = await _get_container_logs("app", lines=0)
    assert "error" not in result
    mock_container.logs.assert_called_once_with(tail=0, stream=False)


async def test_get_container_logs_binary_content_decoded():
    """Binary log output should be decoded with replacement chars, not crash."""
    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.name = "binapp"
        mock_container.logs.return_value = bytes(range(256))
        mock_client.containers.get.return_value = mock_container
        mock_docker.return_value = mock_client
        result = await _get_container_logs("binapp")
    assert isinstance(result["logs"], str)


# ── _list_ollama_models ───────────────────────────────────────────────────────

async def test_list_ollama_models_empty_list():
    mock_ollama = AsyncMock()
    mock_ollama.list_models.return_value = []
    result = await _list_ollama_models(mock_ollama)
    assert result == {"models": []}


async def test_list_ollama_models_missing_name_field():
    """Model dict missing 'name' → name defaults to empty string, no crash."""
    mock_ollama = AsyncMock()
    mock_ollama.list_models.return_value = [{"size": 4_000_000_000}]
    result = await _list_ollama_models(mock_ollama)
    assert result["models"][0]["name"] == ""
    assert result["models"][0]["size_gb"] > 0


async def test_list_ollama_models_missing_size_field():
    """Model dict missing 'size' → size_gb defaults to 0.0, no crash."""
    mock_ollama = AsyncMock()
    mock_ollama.list_models.return_value = [{"name": "llama3:8b"}]
    result = await _list_ollama_models(mock_ollama)
    assert result["models"][0]["name"] == "llama3:8b"
    assert result["models"][0]["size_gb"] == 0.0


async def test_list_ollama_models_size_zero():
    mock_ollama = AsyncMock()
    mock_ollama.list_models.return_value = [{"name": "tiny", "size": 0}]
    result = await _list_ollama_models(mock_ollama)
    assert result["models"][0]["size_gb"] == 0.0


# ── _stop_container: Ollama guard subtleties ──────────────────────────────────

async def test_stop_container_ollama_guard_shows_container_name():
    """The confirm args for Ollama guard must include the container's real name."""
    confirm_calls = []

    async def tracking_confirm(args, **kw):
        confirm_calls.append(args)
        return False

    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        c = _make_container("my-ollama", image_tags=["ollama/ollama:latest"])
        mock_client.containers.get.return_value = c
        mock_docker.return_value = mock_client
        await _stop_container("my-ollama", tracking_confirm)

    assert len(confirm_calls) == 1
    args = confirm_calls[0]
    assert "STOP OLLAMA CONTAINER" in args
    assert "my-ollama" in args


async def test_stop_container_non_ollama_does_not_use_guard():
    """Non-Ollama container stop must NOT use the Ollama guard prompt."""
    confirm_calls = []

    async def tracking_confirm(args, **kw):
        confirm_calls.append(args)
        return False

    with patch("docker.from_env") as mock_docker:
        mock_client = MagicMock()
        c = _make_container("nginx", image_tags=["nginx:alpine"])
        mock_client.containers.get.return_value = c
        mock_docker.return_value = mock_client
        await _stop_container("nginx", tracking_confirm)

    assert len(confirm_calls) == 1
    assert "STOP OLLAMA CONTAINER" not in confirm_calls[0]
