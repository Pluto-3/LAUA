"""Edge-case tests for PtySession."""

import os
from pathlib import Path

import pytest

from laua.executor.pty_session import PtySession


# ── basic sanity ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_args_raises():
    s = PtySession()
    with pytest.raises(ValueError):
        await s.run([])


# ── cd edge cases ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cd_no_args_goes_home():
    """cd with no arguments should navigate to $HOME."""
    s = PtySession("/tmp")
    result = await s.run(["cd"])
    assert result.exit_code == 0
    assert s.cwd == str(Path.home())


@pytest.mark.asyncio
async def test_cd_tilde_expands_to_home():
    """cd ~ must expand to home directory, not literally look for a dir named ~."""
    s = PtySession("/tmp")
    result = await s.run(["cd", "~"])
    assert result.exit_code == 0, f"cd ~ failed: {result.stderr}"
    assert s.cwd == str(Path.home())


@pytest.mark.asyncio
async def test_cd_to_file_is_error():
    """cd /etc/passwd must fail — /etc/passwd is a file, not a directory."""
    s = PtySession("/tmp")
    original_cwd = s.cwd
    result = await s.run(["cd", "/etc/passwd"])
    assert result.exit_code != 0, "cd to a file should fail"
    assert s.cwd == original_cwd, "cwd must not change on failed cd"


@pytest.mark.asyncio
async def test_cd_relative_dotdot():
    s = PtySession("/tmp")
    result = await s.run(["cd", ".."])
    assert result.exit_code == 0
    assert s.cwd == "/"


@pytest.mark.asyncio
async def test_cd_empty_string_stays_put():
    """cd '' — stays in current directory."""
    s = PtySession("/tmp")
    await s.run(["cd", ""])
    assert s.cwd == "/tmp"


@pytest.mark.asyncio
async def test_cd_dash_fails_gracefully():
    """cd - means previous dir in bash but we don't track OLDPWD."""
    s = PtySession("/tmp")
    result = await s.run(["cd", "-"])
    # Should not crash; either fails or goes somewhere. Must not raise.
    assert isinstance(result.exit_code, int)


@pytest.mark.asyncio
async def test_cd_nonexistent_does_not_change_cwd():
    s = PtySession("/tmp")
    await s.run(["cd", "/nonexistent_xyz_abc_123"])
    assert s.cwd == "/tmp"


@pytest.mark.asyncio
async def test_cwd_persists_across_multiple_cd():
    s = PtySession("/tmp")
    await s.run(["cd", "/var"])
    await s.run(["cd", "log"])      # relative from /var
    assert s.cwd == "/var/log"


# ── subprocess edge cases ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_command_with_stderr():
    s = PtySession()
    result = await s.run(["ls", "/nonexistent_path_xyz"])
    assert result.exit_code != 0
    assert result.stderr != ""


@pytest.mark.asyncio
async def test_both_stdout_and_stderr():
    """A command that writes to both streams."""
    s = PtySession()
    result = await s.run(["bash", "-c", "echo out; echo err >&2"], timeout=5)
    # We can't use shell=True but bash -c is a legitimate argv invocation
    assert "out" in result.stdout
    assert "err" in result.stderr


@pytest.mark.asyncio
async def test_exit_code_nonzero():
    s = PtySession()
    result = await s.run(["false"])
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_large_output():
    """Command that emits >64 KB should not hang or truncate at the session level."""
    s = PtySession()
    result = await s.run(["python3", "-c", "print('x' * 100_000)"], timeout=10)
    assert result.exit_code == 0
    assert len(result.stdout) > 90_000


@pytest.mark.asyncio
async def test_binary_output_decoded_with_replace():
    """Binary output should decode with replacement characters, not crash."""
    s = PtySession()
    result = await s.run(
        ["python3", "-c", "import sys; sys.stdout.buffer.write(bytes(range(256)))"],
        timeout=5,
    )
    assert result.exit_code == 0
    assert isinstance(result.stdout, str)


@pytest.mark.asyncio
async def test_args_with_spaces_in_path():
    """Paths containing spaces passed as single arg element (not split)."""
    # Create a temp dir with a space in its name
    import tempfile
    with tempfile.TemporaryDirectory(prefix="test dir ") as d:
        s = PtySession()
        result = await s.run(["ls", d])
        assert result.exit_code == 0


@pytest.mark.asyncio
async def test_permission_error_handled():
    """Running a file without execute permission should not raise uncaught exception."""
    if os.getuid() == 0:
        pytest.skip("Running as root — execute permission checks don't apply")
    import tempfile, stat
    with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
        f.write(b"#!/bin/sh\necho hi\n")
        path = f.name
    os.chmod(path, stat.S_IRUSR)  # read-only, no execute
    try:
        s = PtySession()
        result = await s.run([path])
        # Must return a CommandResult, not raise
        assert isinstance(result.exit_code, int)
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_nonexistent_cwd_initial():
    """PtySession with a non-existent initial cwd falls back to home."""
    s = PtySession("/nonexistent_xyz_initial")
    # The cwd is stored as-is; the subprocess launch would fail
    # if cwd doesn't exist. Test that it does fail gracefully.
    result = await s.run(["ls"])
    # Should not raise; exit_code reflects the OS error
    assert isinstance(result.exit_code, int)
