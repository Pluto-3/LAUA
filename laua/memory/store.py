"""SQLite-backed memory store — session persistence, interaction history, preferences."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    started TEXT NOT NULL,
    ended   TEXT,
    active  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    ts         TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT,
    tool_calls TEXT,
    token_est  INTEGER
);

CREATE TABLE IF NOT EXISTS preferences (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    ts    TEXT NOT NULL
);
"""


class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path).expanduser()

    async def init(self) -> None:
        """Create tables (idempotent)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()
        logger.debug("MemoryStore initialised at %s", self._path)

    async def create_session(self) -> int:
        """Open a new session and return its id."""
        ts = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute(
                "INSERT INTO sessions (started, active) VALUES (?, 1)", (ts,)
            )
            await db.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    async def end_session(self, session_id: int) -> None:
        """Mark a session as ended."""
        ts = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "UPDATE sessions SET ended = ?, active = 0 WHERE id = ?",
                (ts, session_id),
            )
            await db.commit()

    async def get_active_session(self) -> int | None:
        """Return the id of the last active session (crash recovery)."""
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute(
                "SELECT id FROM sessions WHERE active = 1 ORDER BY id DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            return int(row[0]) if row else None

    async def add_message(
        self,
        session_id: int,
        role: str,
        content: str | None,
        tool_calls: Any = None,
        token_est: int | None = None,
    ) -> None:
        """Append a message to the history for a session."""
        ts = datetime.now(timezone.utc).isoformat()
        tool_calls_str = json.dumps(tool_calls) if tool_calls is not None else None
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT INTO messages (session_id, ts, role, content, tool_calls, token_est)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, ts, role, content, tool_calls_str, token_est),
            )
            await db.commit()

    async def get_history(
        self, session_id: int, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Return messages as a list of dicts compatible with the orchestrator."""
        async with aiosqlite.connect(self._path) as db:
            if limit is not None:
                cursor = await db.execute(
                    "SELECT role, content, tool_calls FROM messages"
                    " WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit),
                )
                rows = list(reversed(await cursor.fetchall()))
            else:
                cursor = await db.execute(
                    "SELECT role, content, tool_calls FROM messages"
                    " WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                )
                rows = await cursor.fetchall()

        result: list[dict[str, Any]] = []
        for role, content, tool_calls_str in rows:
            entry: dict[str, Any] = {"role": role, "content": content}
            if tool_calls_str:
                entry["tool_calls"] = json.loads(tool_calls_str)
            result.append(entry)
        return result

    async def set_preference(self, key: str, value: str) -> None:
        """Upsert a user preference."""
        ts = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT INTO preferences (key, value, ts) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value, ts = excluded.ts",
                (key, value, ts),
            )
            await db.commit()

    async def get_preference(self, key: str, default: str | None = None) -> str | None:
        """Retrieve a user preference value, or default if not set."""
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            )
            row = await cursor.fetchone()
            return str(row[0]) if row else default
