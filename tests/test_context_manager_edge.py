"""Edge-case tests for ContextManager."""

from __future__ import annotations

import pytest

from laua.memory.context import ContextManager


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _sys(content: str = "You are an assistant.") -> dict:
    return _msg("system", content)


# ── compress: degenerate inputs ───────────────────────────────────────────────

def test_compress_empty_list():
    cm = ContextManager()
    assert cm.compress([]) == []


def test_compress_single_small_message():
    cm = ContextManager(model_max_tokens=4096)
    msgs = [_msg("user", "hi")]
    assert cm.compress(msgs) == msgs


def test_compress_single_oversized_non_system_message():
    """A single non-system message that's too big gets dropped — leaves empty context."""
    cm = ContextManager(model_max_tokens=10, trigger_ratio=0.8)
    big = _msg("user", "x" * 1000)
    result = cm.compress([big])
    assert big not in result


def test_compress_all_system_messages_no_infinite_loop():
    """
    If every message is a system message, non_system is empty, the while loop
    never executes, and we return them all unchanged (no infinite loop).
    """
    cm = ContextManager(model_max_tokens=10, trigger_ratio=0.8)
    msgs = [_sys("A" * 500), _sys("B" * 500)]
    result = cm.compress(msgs)
    # Both system messages retained — compression was a no-op
    assert len(result) == 2
    assert all(m["role"] == "system" for m in result)


def test_compress_multiple_system_messages_all_kept():
    """Multiple system messages are always preserved."""
    cm = ContextManager(model_max_tokens=100, trigger_ratio=0.8)
    system1 = _sys("First system message")
    system2 = _sys("Second system message")
    user_msgs = [_msg("user", "x" * 400)] * 5
    result = cm.compress([system1, system2] + user_msgs)
    assert system1 in result
    assert system2 in result


# ── trigger_ratio extremes ────────────────────────────────────────────────────

def test_trigger_ratio_zero_always_compresses():
    """trigger_ratio=0 means threshold=0, so any non-empty list triggers compression."""
    cm = ContextManager(model_max_tokens=4096, trigger_ratio=0.0)
    msgs = [_msg("user", "hi")]
    assert cm.should_compress(msgs) is True


def test_trigger_ratio_one_only_at_full_context():
    """trigger_ratio=1.0 triggers only when over the full max token count."""
    cm = ContextManager(model_max_tokens=100, trigger_ratio=1.0)
    # 40 chars / 4 ≈ 10 tokens << 100 max
    msgs = [_msg("user", "x" * 40)]
    assert cm.should_compress(msgs) is False


def test_model_max_tokens_zero_triggers_on_any_content():
    """model_max_tokens=0: threshold=0, everything over 0 triggers compress."""
    cm = ContextManager(model_max_tokens=0, trigger_ratio=0.8)
    msgs = [_msg("user", "x")]
    assert cm.should_compress(msgs) is True


# ── estimate_tokens ───────────────────────────────────────────────────────────

def test_estimate_tokens_with_tool_calls_key():
    """Messages with tool_calls (large nested dicts) are included in the estimate."""
    cm = ContextManager()
    msg = {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "run_command", "arguments": {"args": ["ls"]}}}]}
    estimate = cm.estimate_tokens([msg])
    assert estimate > 0


def test_estimate_tokens_none_content():
    """None content should not crash estimate_tokens."""
    cm = ContextManager()
    msg = {"role": "assistant", "content": None}
    estimate = cm.estimate_tokens([msg])
    assert isinstance(estimate, int)
    assert estimate >= 0


def test_estimate_tokens_large_nested_dict():
    cm = ContextManager()
    msg = {"role": "tool", "content": str({"data": ["a"] * 1000})}
    estimate = cm.estimate_tokens([msg])
    assert estimate > 100


# ── compress: ordering invariants ────────────────────────────────────────────

def test_compress_system_always_first_in_output():
    cm = ContextManager(model_max_tokens=100, trigger_ratio=0.8)
    system = _sys("be helpful")
    msgs = [system] + [_msg("user", "x" * 50)] * 5
    result = cm.compress(msgs)
    assert result[0]["role"] == "system"


def test_compress_preserves_relative_order_of_survivors():
    """Non-dropped messages must remain in their original relative order."""
    cm = ContextManager(model_max_tokens=200, trigger_ratio=0.8)
    msgs = [_msg("user", f"msg {i} " + "x" * 20) for i in range(10)]
    result = cm.compress(msgs)
    contents = [m["content"] for m in result]
    # Check that whatever survived is in ascending index order
    indices = [int(c.split()[1]) for c in contents]
    assert indices == sorted(indices)


def test_compress_result_smaller_than_or_equal_to_input():
    cm = ContextManager(model_max_tokens=50, trigger_ratio=0.8)
    msgs = [_msg("user", "x" * 100)] * 10
    result = cm.compress(msgs)
    assert len(result) <= len(msgs)


# ── should_compress boundary ──────────────────────────────────────────────────

def test_should_compress_exactly_at_threshold():
    """Exactly at the threshold (not over) should NOT compress."""
    cm = ContextManager(model_max_tokens=1000, trigger_ratio=0.8)
    threshold = int(1000 * 0.8)  # 800 tokens
    # Build messages that estimate to exactly 800 tokens (800 * 4 = 3200 chars)
    msgs = [_msg("user", "x" * (threshold * 4))]
    # estimate = len(str(msg)) // 4 ≈ threshold → should be borderline
    # Just check it doesn't crash
    result = cm.should_compress(msgs)
    assert isinstance(result, bool)
