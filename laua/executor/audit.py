"""Append-only audit log stored in SQLite."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    args        TEXT    NOT NULL,
    sudo_used   INTEGER NOT NULL DEFAULT 0,
    stdout      TEXT,
    stderr      TEXT,
    exit_code   INTEGER,
    request     TEXT
);
"""


class AuditLog:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path).expanduser()

    async def init(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    async def record(
        self,
        args: list[str],
        stdout: str,
        stderr: str,
        exit_code: int,
        sudo_used: bool = False,
        request: str | None = None,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT INTO audit_log (ts, args, sudo_used, stdout, stderr, exit_code, request)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    json.dumps(args),
                    int(sudo_used),
                    stdout[:4096],
                    stderr[:4096],
                    exit_code,
                    request,
                ),
            )
            await db.commit()
