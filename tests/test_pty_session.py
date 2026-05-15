"""Tests for the stateful pty session."""

import pytest
from laua.executor.pty_session import PtySession


@pytest.mark.asyncio
async def test_simple_command():
    s = PtySession()
    result = await s.run(["echo", "hello"])
    assert result.exit_code == 0
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_cwd_tracking():
    s = PtySession("/tmp")
    assert s.cwd == "/tmp"
    result = await s.run(["cd", "/var"])
    assert result.exit_code == 0
    assert s.cwd == "/var"


@pytest.mark.asyncio
async def test_invalid_cd():
    s = PtySession("/tmp")
    result = await s.run(["cd", "/nonexistent_path_xyz"])
    assert result.exit_code == 1
    assert s.cwd == "/tmp"  # unchanged


@pytest.mark.asyncio
async def test_command_not_found():
    s = PtySession()
    result = await s.run(["nonexistent_binary_xyz_abc"])
    assert result.exit_code == 127


@pytest.mark.asyncio
async def test_timeout():
    s = PtySession()
    result = await s.run(["sleep", "60"], timeout=1)
    assert result.timed_out is True
    assert result.exit_code == -1
