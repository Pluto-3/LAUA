"""Stateful pseudo-terminal session. Tracks $PWD across tool calls."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30


@dataclass
class CommandResult:
    args: list[str]
    stdout: str
    stderr: str
    exit_code: int
    cwd: str
    timed_out: bool = False


class PtySession:
    """
    Maintains a persistent working directory context.
    Each command is spawned with the tracked cwd so multi-step
    navigation works correctly across tool calls.

    shell=True is explicitly prohibited — all commands must arrive
    as argument arrays.
    """

    def __init__(self, initial_cwd: str | None = None) -> None:
        self.cwd = initial_cwd or str(Path.home())

    async def run(
        self,
        args: list[str],
        timeout: int = _DEFAULT_TIMEOUT,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        if not args:
            raise ValueError("args must be a non-empty list")

        # Detect cd specially so we can track PWD
        if args[0] == "cd":
            return self._handle_cd(args)

        proc_env = {**os.environ, **(env or {}), "PWD": self.cwd}


        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=proc_env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                exit_code = proc.returncode or 0
                timed_out = False
            except asyncio.TimeoutError:
                proc.kill()
                stdout_bytes, stderr_bytes = await proc.communicate()
                exit_code = -1
                timed_out = True
                logger.warning("Command %s timed out after %ds", args, timeout)
        except FileNotFoundError:
            return CommandResult(
                args=args,
                stdout="",
                stderr=f"Command not found: {args[0]}",
                exit_code=127,
                cwd=self.cwd,
            )
        except PermissionError:
            return CommandResult(
                args=args,
                stdout="",
                stderr=f"Permission denied: {args[0]}",
                exit_code=126,
                cwd=self.cwd,
            )

        return CommandResult(
            args=args,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace"),
            exit_code=exit_code,
            cwd=self.cwd,
            timed_out=timed_out,
        )

    def _handle_cd(self, args: list[str]) -> CommandResult:
        target = args[1] if len(args) > 1 else str(Path.home())
        # Expand ~ before any path resolution
        target = os.path.expanduser(target)
        new_path = Path(self.cwd) / target if not Path(target).is_absolute() else Path(target)
        try:
            resolved = new_path.resolve(strict=True)
            if not resolved.is_dir():
                return CommandResult(
                    args=args,
                    stdout="",
                    stderr=f"cd: {args[1] if len(args) > 1 else target}: Not a directory",
                    exit_code=1,
                    cwd=self.cwd,
                )
            self.cwd = str(resolved)
            return CommandResult(args=args, stdout="", stderr="", exit_code=0, cwd=self.cwd)
        except FileNotFoundError:
            return CommandResult(
                args=args,
                stdout="",
                stderr=f"cd: {args[1] if len(args) > 1 else target}: No such file or directory",
                exit_code=1,
                cwd=self.cwd,
            )
