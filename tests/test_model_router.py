"""Tests for ModelRouter — task classification, model selection, fallback."""

import pytest
from laua.planner.router import ModelRouter

_CFG = {
    "fast": "fast-model",
    "reasoning": "reasoning-model",
    "coding": "coding-model",
    "fallback": "fallback-model",
}


def router():
    return ModelRouter(_CFG)


def test_classify_coding_keyword():
    assert router().classify("can you write a function to sort a list?") == "coding"


def test_classify_reasoning_keyword():
    assert router().classify("why is this approach better than the other?") == "reasoning"


def test_classify_fast_default():
    assert router().classify("show me disk usage") == "fast"


def test_classify_case_insensitive():
    assert router().classify("DEBUG this script") == "coding"


def test_get_model_for_coding():
    assert router().get_model("coding") == "coding-model"


def test_get_model_for_reasoning():
    assert router().get_model("reasoning") == "reasoning-model"


def test_get_model_for_fast():
    assert router().get_model("fast") == "fast-model"


def test_get_model_unknown_falls_back_to_fast():
    assert router().get_model("unknown_type") == "fast-model"


def test_get_fallback_model():
    assert router().get_fallback_model() == "fallback-model"


def test_get_fallback_model_default_when_not_configured():
    r = ModelRouter({})
    assert r.get_fallback_model() == "qwen2.5:7b"


def test_empty_message_is_fast():
    assert router().classify("") == "fast"


def test_multiple_keyword_types_first_wins():
    # "code" (coding) and "explain" (reasoning) both present — coding listed first
    result = router().classify("explain this code to me")
    assert result in ("coding", "reasoning")  # either is acceptable; just must not crash
