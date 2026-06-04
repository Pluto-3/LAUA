"""SQLite-backed workflow recorder — capture and replay named tool-call sequences."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflows (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL UNIQUE,
    steps     TEXT NOT NULL,
    created   TEXT NOT NULL,
    run_count INTEGER NOT NULL DEFAULT 0
);
"""


class WorkflowStore:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path).expanduser()

    async def init(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()
        logger.debug("WorkflowStore initialised at %s", self._path)

    async def save(self, name: str, steps: list[dict[str, Any]]) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT INTO workflows (name, steps, created) VALUES (?, ?, ?)"
                " ON CONFLICT(name) DO UPDATE SET steps = excluded.steps,"
                " created = excluded.created, run_count = 0",
                (name, json.dumps(steps), ts),
            )
            await db.commit()
        logger.debug("Saved workflow %r (%d steps)", name, len(steps))

    async def load(self, name: str) -> list[dict[str, Any]] | None:
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute("SELECT steps FROM workflows WHERE name = ?", (name,))
            row = await cursor.fetchone()
            if row is None:
                return None
            await db.execute(
                "UPDATE workflows SET run_count = run_count + 1 WHERE name = ?", (name,)
            )
            await db.commit()
        return json.loads(row[0])

    async def list_workflows(self) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute(
                "SELECT name, created, run_count FROM workflows ORDER BY name"
            )
            rows = await cursor.fetchall()
        return [{"name": r[0], "created": r[1], "run_count": r[2]} for r in rows]

    async def delete(self, name: str) -> bool:
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute("DELETE FROM workflows WHERE name = ?", (name,))
            await db.commit()
            return cursor.rowcount > 0
