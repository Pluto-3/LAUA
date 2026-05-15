"""Edge-case tests for core tools (read_file, get_system_info)."""

import os
import stat
import tempfile
from pathlib import Path

import pytest

from laua.tools.core import _get_system_info, _read_file


# ── _read_file ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_read_file_nonexistent():
    result = await _read_file("/nonexistent_path_xyz_abc/file.txt")
    assert "error" in result


@pytest.mark.asyncio
async def test_read_file_directory():
    """Passing a directory path should return an error, not crash."""
    result = await _read_file("/tmp")
    assert "error" in result


@pytest.mark.asyncio
async def test_read_file_normal():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        path = f.name
    try:
        result = await _read_file(path)
        assert result["content"] == "hello world"
        assert result["truncated"] is False
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_read_file_truncation():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(b"a" * 200)
        path = f.name
    try:
        result = await _read_file(path, max_bytes=100)
        assert result["truncated"] is True
        assert len(result["content"]) == 100
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_read_file_exact_max_bytes_marked_truncated():
    """
    A file of exactly max_bytes bytes: truncated will be True even though
    nothing was dropped. This is a known edge case — document it.
    """
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(b"b" * 100)
        path = f.name
    try:
        result = await _read_file(path, max_bytes=100)
        # Currently returns True (false positive) — document behaviour
        assert result["truncated"] is True
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_read_file_empty_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        path = f.name
    try:
        result = await _read_file(path)
        assert result["content"] == ""
        assert result["truncated"] is False
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_read_file_binary_decoded():
    """Binary file decoded with replacement chars, not crash."""
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(bytes(range(256)))
        path = f.name
    try:
        result = await _read_file(path)
        assert isinstance(result["content"], str)
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_read_file_no_read_permission():
    """Root bypasses file permissions, so skip when running as root."""
    if os.getuid() == 0:
        pytest.skip("Running as root — file permission checks don't apply")
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("secret")
        path = f.name
    os.chmod(path, 0o000)
    try:
        result = await _read_file(path)
        assert "error" in result
    finally:
        os.chmod(path, 0o644)
        os.unlink(path)


@pytest.mark.asyncio
async def test_read_file_tilde_expansion():
    """~ in path should expand to home."""
    home_file = Path.home() / ".profile"
    if home_file.exists():
        result = await _read_file("~/.profile")
        assert "error" not in result
    else:
        pytest.skip("~/.profile not present")


# ── _get_system_info ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_system_info_all_keys():
    result = await _get_system_info()
    assert "cpu_percent" in result
    assert "memory" in result
    assert "disk" in result
    assert "top_processes" in result


@pytest.mark.asyncio
async def test_get_system_info_subset():
    result = await _get_system_info(include=["cpu"])
    assert "cpu_percent" in result
    assert "memory" not in result
    assert "disk" not in result


@pytest.mark.asyncio
async def test_get_system_info_empty_include():
    """Empty include list should return empty info dict (no keys)."""
    result = await _get_system_info(include=[])
    assert result == {}


@pytest.mark.asyncio
async def test_get_system_info_own_pid_filtered():
    """Our own PID must not appear in top_processes."""
    own_pid = os.getpid()
    result = await _get_system_info(include=["processes"], own_pids=[own_pid])
    pids = [p["pid"] for p in result.get("top_processes", [])]
    assert own_pid not in pids


@pytest.mark.asyncio
async def test_get_system_info_custom_pid_filtered():
    """A specific PID passed as own_pids must be excluded."""
    result = await _get_system_info(include=["processes"], own_pids=[1])
    pids = [p["pid"] for p in result.get("top_processes", [])]
    assert 1 not in pids


@pytest.mark.asyncio
async def test_get_system_info_unknown_include_key_ignored():
    """Unknown include keys should not crash — just produce no output for them."""
    result = await _get_system_info(include=["cpu", "bogus_key"])
    assert "cpu_percent" in result
    assert "bogus_key" not in result


@pytest.mark.asyncio
async def test_get_system_info_returns_sane_values():
    result = await _get_system_info(include=["cpu", "memory", "disk"])
    assert 0.0 <= result["cpu_percent"] <= 100.0
    assert 0.0 <= result["memory"]["percent"] <= 100.0
    assert 0.0 <= result["disk"]["percent"] <= 100.0
    assert result["memory"]["total_gb"] > 0
    assert result["disk"]["total_gb"] > 0
