"""Edge-case tests for ModelRouter."""

from __future__ import annotations

import pytest

from laua.planner.router import ModelRouter, TASK_KEYWORDS

_CFG = {
    "fast": "fast-model",
    "reasoning": "reasoning-model",
    "coding": "coding-model",
    "fallback": "fallback-model",
}


def router(cfg=None) -> ModelRouter:
    return ModelRouter(cfg if cfg is not None else _CFG)


# ── substring false positives ─────────────────────────────────────────────────

def test_errors_substring_matches_error():
    """'errors' contains 'error' → coding (document this substring behaviour)."""
    result = router().classify("there are no errors in this output")
    assert result == "coding"


def test_classes_substring_matches_class():
    """'classes' contains 'class' → coding."""
    result = router().classify("I have several classes in my project")
    assert result == "coding"


def test_whys_contains_why():
    """'whys' contains 'why' → reasoning."""
    result = router().classify("those are the whys of this design")
    assert result == "reasoning"


def test_reclassify_contains_class():
    """'reclassify' contains 'class'? Let's verify the exact substring."""
    # 'reclassify' = r-e-c-l-a-s-s-i-f-y. 'class' = c-l-a-s-s.
    # Check: 'class' in 'reclassify' → True (starts at index 2: c-l-a-s-s-i...)
    assert "class" in "reclassify"
    result = router().classify("I need to reclassify these items")
    assert result == "coding"  # document: this is a known false positive


def test_otherwise_no_false_match():
    """'otherwise' does not contain any keywords — should be 'fast'."""
    result = router().classify("do this otherwise do that")
    # 'why' not in 'otherwise' — 'otherwise' = o-t-h-e-r-w-i-s-e → no 'why'
    # Let's verify: 'why' in 'otherwise' → False
    assert "why" not in "otherwise"
    assert result == "fast"


# ── whitespace / empty inputs ─────────────────────────────────────────────────

def test_whitespace_only_message():
    assert router().classify("   \t\n  ") == "fast"


def test_single_char_message():
    assert router().classify("x") == "fast"


def test_very_long_message_with_no_keywords():
    msg = "do " * 1000
    assert router().classify(msg) == "fast"


def test_very_long_message_with_keyword_at_end():
    msg = "blah " * 500 + "debug this please"
    assert router().classify(msg) == "coding"


# ── config edge cases ─────────────────────────────────────────────────────────

def test_missing_fast_in_config_falls_back_to_hardcoded():
    r = ModelRouter({"coding": "code-model"})
    assert r.get_model("fast") == "qwen2.5:7b"  # hardcoded default


def test_missing_fallback_uses_fast():
    r = ModelRouter({"fast": "my-fast-model"})
    assert r.get_fallback_model() == "my-fast-model"


def test_all_same_model_config():
    """All task types pointing to the same model — realistic default config."""
    r = ModelRouter({"fast": "qwen", "reasoning": "qwen", "coding": "qwen", "fallback": "qwen"})
    assert r.get_model("coding") == "qwen"
    assert r.get_model("reasoning") == "qwen"
    assert r.get_fallback_model() == "qwen"


def test_empty_config_returns_defaults():
    r = ModelRouter({})
    assert r.get_model("anything") == "qwen2.5:7b"
    assert r.get_fallback_model() == "qwen2.5:7b"


def test_get_model_none_value_falls_back_to_fast():
    """If a model name is explicitly None, fall back to fast."""
    r = ModelRouter({"coding": None, "fast": "fast-model"})
    # None is falsy, so `self._config.get("coding") or self._config.get("fast")` → "fast-model"
    assert r.get_model("coding") == "fast-model"


# ── keyword ordering ──────────────────────────────────────────────────────────

def test_coding_takes_priority_over_reasoning_when_both_present():
    """'code' (coding) + 'explain' (reasoning) in same message — coding is checked first."""
    result = router().classify("explain this code")
    assert result == "coding"  # coding keywords checked first in TASK_KEYWORDS


def test_task_keywords_dict_order():
    """Document: TASK_KEYWORDS iteration order determines priority."""
    keys = list(TASK_KEYWORDS.keys())
    assert keys[0] == "coding", "coding should be checked before reasoning"


# ── classify: specific keyword coverage ──────────────────────────────────────

@pytest.mark.parametrize("keyword", TASK_KEYWORDS["coding"])
def test_each_coding_keyword_classified(keyword):
    assert router().classify(f"please {keyword} this thing") == "coding"


@pytest.mark.parametrize("keyword", TASK_KEYWORDS["reasoning"])
def test_each_reasoning_keyword_classified(keyword):
    # Skip if keyword is also a coding keyword (shouldn't overlap but be safe)
    if any(keyword in ck for ck in TASK_KEYWORDS["coding"]):
        pytest.skip(f"'{keyword}' overlaps with coding keywords")
    assert router().classify(f"please {keyword} this thing") == "reasoning"
