"""Edge-case tests for AuditLog."""

import json
import tempfile
from pathlib import Path

import aiosqlite
import pytest

from laua.executor.audit import AuditLog


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "audit.db"


@pytest.mark.asyncio
async def test_record_before_init_raises():
    """Calling record() before init() should raise (table doesn't exist)."""
    with tempfile.TemporaryDirectory() as d:
        log = AuditLog(Path(d) / "audit.db")
        with pytest.raises(Exception):
            await log.record(["ls"], "out", "err", 0)


@pytest.mark.asyncio
async def test_double_init_is_idempotent(tmp_db):
    """Calling init() twice must not raise (CREATE TABLE IF NOT EXISTS)."""
    log = AuditLog(tmp_db)
    await log.init()
    await log.init()  # should not raise


@pytest.mark.asyncio
async def test_record_basic(tmp_db):
    log = AuditLog(tmp_db)
    await log.init()
    await log.record(["ls", "-la"], "output", "", 0, sudo_used=False, request="list files")

    async with aiosqlite.connect(tmp_db) as db:
        async with db.execute("SELECT * FROM audit_log") as cur:
            rows = await cur.fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][2]) == ["ls", "-la"]  # args column
    assert rows[0][3] == 0  # sudo_used
    assert rows[0][7] == "list files"  # request


@pytest.mark.asyncio
async def test_stdout_truncated_at_4096(tmp_db):
    log = AuditLog(tmp_db)
    await log.init()
    big = "x" * 10_000
    await log.record(["cat", "file"], big, "", 0)

    async with aiosqlite.connect(tmp_db) as db:
        async with db.execute("SELECT stdout FROM audit_log") as cur:
            row = await cur.fetchone()
    assert len(row[0]) == 4096


@pytest.mark.asyncio
async def test_stderr_truncated_at_4096(tmp_db):
    log = AuditLog(tmp_db)
    await log.init()
    await log.record(["bad"], "", "e" * 10_000, 1)

    async with aiosqlite.connect(tmp_db) as db:
        async with db.execute("SELECT stderr FROM audit_log") as cur:
            row = await cur.fetchone()
    assert len(row[0]) == 4096


@pytest.mark.asyncio
async def test_record_none_request(tmp_db):
    log = AuditLog(tmp_db)
    await log.init()
    await log.record(["echo", "hi"], "hi\n", "", 0, request=None)

    async with aiosqlite.connect(tmp_db) as db:
        async with db.execute("SELECT request FROM audit_log") as cur:
            row = await cur.fetchone()
    assert row[0] is None


@pytest.mark.asyncio
async def test_sudo_flag_stored(tmp_db):
    log = AuditLog(tmp_db)
    await log.init()
    await log.record(["sudo", "apt", "update"], "", "", 0, sudo_used=True)

    async with aiosqlite.connect(tmp_db) as db:
        async with db.execute("SELECT sudo_used FROM audit_log") as cur:
            row = await cur.fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_multiple_records_append(tmp_db):
    log = AuditLog(tmp_db)
    await log.init()
    for i in range(5):
        await log.record([f"cmd{i}"], f"out{i}", "", i)

    async with aiosqlite.connect(tmp_db) as db:
        async with db.execute("SELECT COUNT(*) FROM audit_log") as cur:
            count = (await cur.fetchone())[0]
    assert count == 5


@pytest.mark.asyncio
async def test_exact_4096_bytes_not_falsely_truncated(tmp_db):
    """File of exactly max_bytes — truncated flag should be False (≠ read_file, but audit truncates strings)."""
    log = AuditLog(tmp_db)
    await log.init()
    exact = "y" * 4096
    await log.record(["cat"], exact, "", 0)

    async with aiosqlite.connect(tmp_db) as db:
        async with db.execute("SELECT stdout FROM audit_log") as cur:
            row = await cur.fetchone()
    # Stored as-is, not truncated
    assert len(row[0]) == 4096
