"""Tests for MemoryStore — session persistence, interaction history, preferences, crash recovery."""

from __future__ import annotations

import pytest

from laua.memory.store import MemoryStore


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(tmp_path / "test.db")
    await s.init()
    return s


async def test_init_creates_tables(tmp_path):
    """init() is idempotent and creates the required tables."""
    s = MemoryStore(tmp_path / "laua.db")
    await s.init()
    await s.init()  # second call must not raise


async def test_create_session_returns_int(store):
    sid = await store.create_session()
    assert isinstance(sid, int)
    assert sid >= 1


async def test_multiple_sessions_get_distinct_ids(store):
    sid1 = await store.create_session()
    sid2 = await store.create_session()
    assert sid1 != sid2


async def test_get_active_session_returns_latest(store):
    sid1 = await store.create_session()
    sid2 = await store.create_session()
    active = await store.get_active_session()
    assert active == sid2


async def test_get_active_session_none_when_empty(tmp_path):
    s = MemoryStore(tmp_path / "fresh.db")
    await s.init()
    assert await s.get_active_session() is None


async def test_end_session_marks_inactive(store):
    sid = await store.create_session()
    await store.end_session(sid)
    active = await store.get_active_session()
    assert active is None


async def test_crash_recovery_active_session_survives(store):
    """Crash recovery: an open session (not ended) is visible on next get_active_session."""
    sid = await store.create_session()
    # Simulate crash — no end_session call
    recovered = await store.get_active_session()
    assert recovered == sid


async def test_add_and_get_history(store):
    sid = await store.create_session()
    await store.add_message(sid, "user", "Hello")
    await store.add_message(sid, "assistant", "Hi there!")
    history = await store.get_history(sid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hi there!"


async def test_get_history_with_limit(store):
    sid = await store.create_session()
    for i in range(5):
        await store.add_message(sid, "user", f"msg {i}")
    history = await store.get_history(sid, limit=3)
    assert len(history) == 3
    # With limit, should return the most recent messages
    assert history[-1]["content"] == "msg 4"


async def test_get_history_empty(store):
    sid = await store.create_session()
    history = await store.get_history(sid)
    assert history == []


async def test_add_message_with_tool_calls(store):
    sid = await store.create_session()
    tool_calls = [{"function": {"name": "run_command", "arguments": {"args": ["ls"]}}}]
    await store.add_message(sid, "assistant", None, tool_calls=tool_calls)
    history = await store.get_history(sid)
    assert len(history) == 1
    assert "tool_calls" in history[0]
    assert history[0]["tool_calls"][0]["function"]["name"] == "run_command"


async def test_add_message_with_token_est(store):
    """token_est is stored and doesn't cause an error."""
    sid = await store.create_session()
    await store.add_message(sid, "user", "test", token_est=42)
    history = await store.get_history(sid)
    assert len(history) == 1


async def test_set_and_get_preference(store):
    await store.set_preference("theme", "dark")
    value = await store.get_preference("theme")
    assert value == "dark"


async def test_get_preference_default(store):
    value = await store.get_preference("nonexistent", default="fallback")
    assert value == "fallback"


async def test_get_preference_none_default(store):
    value = await store.get_preference("nonexistent")
    assert value is None


async def test_set_preference_upsert(store):
    """set_preference overwrites an existing key."""
    await store.set_preference("model", "qwen2.5:7b")
    await store.set_preference("model", "llama3:8b")
    value = await store.get_preference("model")
    assert value == "llama3:8b"


async def test_history_isolated_per_session(store):
    """Messages from different sessions don't bleed into each other."""
    sid1 = await store.create_session()
    sid2 = await store.create_session()
    await store.add_message(sid1, "user", "session one message")
    await store.add_message(sid2, "user", "session two message")
    h1 = await store.get_history(sid1)
    h2 = await store.get_history(sid2)
    assert len(h1) == 1
    assert len(h2) == 1
    assert h1[0]["content"] == "session one message"
    assert h2[0]["content"] == "session two message"
