"""SQLite-backed schedule store — fires named workflows on a fixed interval."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL UNIQUE,
    workflow_name    TEXT NOT NULL,
    interval_seconds INTEGER NOT NULL,
    next_run         TEXT NOT NULL,
    last_run         TEXT,
    enabled          INTEGER NOT NULL DEFAULT 1,
    run_count        INTEGER NOT NULL DEFAULT 0,
    created          TEXT NOT NULL
);
"""


def compute_next_run(interval_seconds: int, from_time: datetime | None = None) -> str:
    base = from_time or datetime.now(timezone.utc)
    return (base + timedelta(seconds=interval_seconds)).isoformat()


class SchedulesStore:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path).expanduser()

    async def init(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()
        logger.debug("SchedulesStore initialised at %s", self._path)

    async def create(self, name: str, workflow_name: str, interval_seconds: int) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        next_run = compute_next_run(interval_seconds)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT INTO schedules"
                " (name, workflow_name, interval_seconds, next_run, last_run, enabled, run_count, created)"
                " VALUES (?, ?, ?, ?, NULL, 1, 0, ?)"
                " ON CONFLICT(name) DO UPDATE SET"
                " workflow_name = excluded.workflow_name,"
                " interval_seconds = excluded.interval_seconds,"
                " next_run = excluded.next_run,"
                " last_run = NULL, enabled = 1, run_count = 0, created = excluded.created",
                (name, workflow_name, interval_seconds, next_run, ts),
            )
            await db.commit()
        logger.debug("Created schedule %r -> workflow %r every %ss", name, workflow_name, interval_seconds)

    async def list_schedules(self) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute(
                "SELECT name, workflow_name, interval_seconds, next_run, last_run,"
                " enabled, run_count FROM schedules ORDER BY name"
            )
            rows = await cursor.fetchall()
        return [
            {
                "name": r[0],
                "workflow_name": r[1],
                "interval_seconds": r[2],
                "next_run": r[3],
                "last_run": r[4],
                "enabled": bool(r[5]),
                "run_count": r[6],
            }
            for r in rows
        ]

    async def due_schedules(self, now_iso: str | None = None) -> list[dict[str, Any]]:
        now = now_iso or datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute(
                "SELECT name, workflow_name, interval_seconds, next_run, last_run,"
                " enabled, run_count FROM schedules"
                " WHERE enabled = 1 AND next_run <= ? ORDER BY name",
                (now,),
            )
            rows = await cursor.fetchall()
        return [
            {
                "name": r[0],
                "workflow_name": r[1],
                "interval_seconds": r[2],
                "next_run": r[3],
                "last_run": r[4],
                "enabled": bool(r[5]),
                "run_count": r[6],
            }
            for r in rows
        ]

    async def mark_run(self, name: str, interval_seconds: int) -> None:
        now = datetime.now(timezone.utc)
        next_run = compute_next_run(interval_seconds, from_time=now)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "UPDATE schedules SET last_run = ?, next_run = ?, run_count = run_count + 1"
                " WHERE name = ?",
                (now.isoformat(), next_run, name),
            )
            await db.commit()

    async def set_enabled(self, name: str, enabled: bool) -> bool:
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute(
                "UPDATE schedules SET enabled = ? WHERE name = ?", (int(enabled), name)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def delete(self, name: str) -> bool:
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute("DELETE FROM schedules WHERE name = ?", (name,))
            await db.commit()
            return cursor.rowcount > 0
