"""Edge-case tests for MemoryStore."""

from __future__ import annotations

import pytest
import aiosqlite

from laua.memory.store import MemoryStore


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(tmp_path / "test.db")
    await s.init()
    return s


# ── init guards ───────────────────────────────────────────────────────────────

async def test_create_session_before_init_raises(tmp_path):
    """Using the store before init() should raise, not silently corrupt."""
    s = MemoryStore(tmp_path / "uninit.db")
    with pytest.raises(Exception):
        await s.create_session()


async def test_add_message_before_init_raises(tmp_path):
    s = MemoryStore(tmp_path / "uninit.db")
    with pytest.raises(Exception):
        await s.add_message(1, "user", "hello")


# ── end_session edge cases ────────────────────────────────────────────────────

async def test_end_session_nonexistent_id_does_not_raise(store):
    """Ending a session that doesn't exist should be a silent no-op."""
    await store.end_session(99999)  # must not raise


async def test_end_session_twice_is_idempotent(store):
    sid = await store.create_session()
    await store.end_session(sid)
    await store.end_session(sid)  # second call must not raise
    assert await store.get_active_session() is None


# ── get_active_session with multiple active sessions ─────────────────────────

async def test_get_active_session_returns_latest_of_multiple(store):
    sid1 = await store.create_session()
    sid2 = await store.create_session()
    sid3 = await store.create_session()
    assert await store.get_active_session() == sid3


async def test_get_active_session_skips_ended_sessions(store):
    sid1 = await store.create_session()
    sid2 = await store.create_session()
    await store.end_session(sid2)
    assert await store.get_active_session() == sid1


# ── add_message: content variants ────────────────────────────────────────────

async def test_add_message_none_content(store):
    """None content is valid (tool messages can have no text)."""
    sid = await store.create_session()
    await store.add_message(sid, "assistant", None)
    history = await store.get_history(sid)
    assert history[0]["content"] is None


async def test_add_message_empty_string_content(store):
    sid = await store.create_session()
    await store.add_message(sid, "user", "")
    history = await store.get_history(sid)
    assert history[0]["content"] == ""


async def test_add_message_unicode_content(store):
    sid = await store.create_session()
    text = "こんにちは 🎉 emoji and unicode: ☃"
    await store.add_message(sid, "user", text)
    history = await store.get_history(sid)
    assert history[0]["content"] == text


async def test_add_message_multiline_content(store):
    sid = await store.create_session()
    text = "line1\nline2\nline3"
    await store.add_message(sid, "user", text)
    history = await store.get_history(sid)
    assert history[0]["content"] == text


async def test_add_message_json_string_content(store):
    """Content that looks like JSON should be stored as plain text."""
    sid = await store.create_session()
    text = '{"key": "value", "list": [1, 2, 3]}'
    await store.add_message(sid, "tool", text)
    history = await store.get_history(sid)
    assert history[0]["content"] == text


async def test_add_message_very_large_content(store):
    """Large content (> 4096 chars) should be stored in full (no truncation in store)."""
    sid = await store.create_session()
    text = "x" * 100_000
    await store.add_message(sid, "user", text)
    history = await store.get_history(sid)
    assert len(history[0]["content"]) == 100_000


# ── get_history: limit edge cases ─────────────────────────────────────────────

async def test_get_history_limit_larger_than_count(store):
    """limit > actual messages should return all messages, not crash."""
    sid = await store.create_session()
    await store.add_message(sid, "user", "only message")
    history = await store.get_history(sid, limit=100)
    assert len(history) == 1
    assert history[0]["content"] == "only message"


async def test_get_history_limit_zero(store):
    """limit=0 should return empty list."""
    sid = await store.create_session()
    await store.add_message(sid, "user", "msg")
    history = await store.get_history(sid, limit=0)
    assert history == []


async def test_get_history_limit_returns_most_recent(store):
    """With limit=2 from 5 messages, the last 2 are returned in chronological order."""
    sid = await store.create_session()
    for i in range(5):
        await store.add_message(sid, "user", f"msg {i}")
    history = await store.get_history(sid, limit=2)
    assert len(history) == 2
    assert history[0]["content"] == "msg 3"
    assert history[1]["content"] == "msg 4"


async def test_get_history_limit_one(store):
    sid = await store.create_session()
    for i in range(3):
        await store.add_message(sid, "user", f"msg {i}")
    history = await store.get_history(sid, limit=1)
    assert len(history) == 1
    assert history[0]["content"] == "msg 2"


# ── get_history: isolation ────────────────────────────────────────────────────

async def test_get_history_nonexistent_session_returns_empty(store):
    """Querying a session ID that was never created returns empty, not error."""
    history = await store.get_history(99999)
    assert history == []


# ── tool_calls round-trip ─────────────────────────────────────────────────────

async def test_tool_calls_complex_nested_roundtrip(store):
    """Complex nested tool_calls structure survives JSON round-trip."""
    sid = await store.create_session()
    calls = [
        {"function": {"name": "run_command", "arguments": {"args": ["ls", "-la", "/tmp"]}}},
        {"function": {"name": "read_file", "arguments": {"path": "/etc/os-release"}}},
    ]
    await store.add_message(sid, "assistant", None, tool_calls=calls)
    history = await store.get_history(sid)
    assert history[0]["tool_calls"] == calls


async def test_tool_calls_none_not_in_result(store):
    """Messages without tool_calls should not have the key in history."""
    sid = await store.create_session()
    await store.add_message(sid, "user", "hello", tool_calls=None)
    history = await store.get_history(sid)
    assert "tool_calls" not in history[0]


# ── preferences edge cases ────────────────────────────────────────────────────

async def test_preference_empty_string_value(store):
    await store.set_preference("empty_pref", "")
    assert await store.get_preference("empty_pref") == ""


async def test_preference_overwrite_with_empty(store):
    await store.set_preference("key", "original")
    await store.set_preference("key", "")
    assert await store.get_preference("key") == ""


async def test_preference_unicode_value(store):
    await store.set_preference("lang", "日本語")
    assert await store.get_preference("lang") == "日本語"


async def test_preference_very_long_value(store):
    long_value = "v" * 10_000
    await store.set_preference("long_key", long_value)
    assert await store.get_preference("long_key") == long_value


async def test_preference_key_with_special_chars(store):
    """Preference keys can contain dots, slashes etc (SQLite handles it)."""
    await store.set_preference("model.routing/fast", "qwen2.5:7b")
    assert await store.get_preference("model.routing/fast") == "qwen2.5:7b"
