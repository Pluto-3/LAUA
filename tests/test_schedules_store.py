"""Tests for SchedulesStore — recurring workflow schedules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from laua.memory.schedules import SchedulesStore, compute_next_run


@pytest.fixture
async def store(tmp_path):
    s = SchedulesStore(tmp_path / "test.db")
    await s.init()
    return s


async def test_init_creates_tables(tmp_path):
    s = SchedulesStore(tmp_path / "schedules.db")
    await s.init()
    await s.init()  # second call must not raise


async def test_create_and_list_schedules(store):
    await store.create("my-sched", "my-workflow", 60)
    schedules = await store.list_schedules()
    assert len(schedules) == 1
    s = schedules[0]
    assert s["name"] == "my-sched"
    assert s["workflow_name"] == "my-workflow"
    assert s["interval_seconds"] == 60
    assert s["enabled"] is True
    assert s["run_count"] == 0
    assert s["last_run"] is None


async def test_create_upsert_resets_state(store):
    await store.create("my-sched", "workflow-a", 60)
    await store.set_enabled("my-sched", False)
    await store.mark_run("my-sched", 60)

    await store.create("my-sched", "workflow-b", 120)

    schedules = await store.list_schedules()
    assert len(schedules) == 1
    s = schedules[0]
    assert s["workflow_name"] == "workflow-b"
    assert s["interval_seconds"] == 120
    assert s["enabled"] is True
    assert s["run_count"] == 0
    assert s["last_run"] is None


async def test_due_schedules_excludes_future_next_run(store):
    await store.create("future-sched", "wf", 3600)
    due = await store.due_schedules()
    assert due == []


async def test_due_schedules_includes_past_due(store):
    await store.create("due-sched", "wf", 60)
    now = datetime.now(timezone.utc)
    past = (now - timedelta(hours=1)).isoformat()
    due = await store.due_schedules(now_iso=past)
    # next_run was set to now+60s, which is after `past`, so nothing due yet
    assert due == []

    future_now = (now + timedelta(hours=1)).isoformat()
    due = await store.due_schedules(now_iso=future_now)
    assert len(due) == 1
    assert due[0]["name"] == "due-sched"


async def test_due_schedules_excludes_disabled(store):
    await store.create("disabled-sched", "wf", 60)
    await store.set_enabled("disabled-sched", False)
    future_now = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    due = await store.due_schedules(now_iso=future_now)
    assert due == []


async def test_due_schedules_is_side_effect_free(store):
    await store.create("sched", "wf", 60)
    future_now = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    first = await store.due_schedules(now_iso=future_now)
    second = await store.due_schedules(now_iso=future_now)

    assert first == second
    schedules = await store.list_schedules()
    assert schedules[0]["run_count"] == 0
    assert schedules[0]["last_run"] is None


async def test_mark_run_advances_next_run_and_increments_run_count(store):
    await store.create("sched", "wf", 60)
    await store.mark_run("sched", 60)
    schedules = await store.list_schedules()
    s = schedules[0]
    assert s["run_count"] == 1
    assert s["last_run"] is not None


async def test_set_enabled_toggle(store):
    await store.create("sched", "wf", 60)
    assert await store.set_enabled("sched", False) is True
    schedules = await store.list_schedules()
    assert schedules[0]["enabled"] is False
    assert await store.set_enabled("sched", True) is True
    schedules = await store.list_schedules()
    assert schedules[0]["enabled"] is True


async def test_set_enabled_unknown_name_returns_false(store):
    assert await store.set_enabled("nonexistent", False) is False


async def test_delete_removes_row(store):
    await store.create("sched", "wf", 60)
    assert await store.delete("sched") is True
    assert await store.list_schedules() == []


async def test_delete_unknown_name_returns_false(store):
    assert await store.delete("nonexistent") is False


def test_compute_next_run_adds_interval():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = compute_next_run(60, from_time=base)
    assert result == "2026-01-01T00:01:00+00:00"
