"""Tests for parse_schedule_command — pure parsing, no I/O."""

from __future__ import annotations

import pytest

from laua.scheduler import parse_schedule_command


def test_parses_minutes():
    assert parse_schedule_command("my-sched my-workflow every 30 minutes") == (
        "my-sched", "my-workflow", 1800,
    )


def test_parses_hours():
    assert parse_schedule_command("nightly backup every 2 hours") == (
        "nightly", "backup", 7200,
    )


def test_parses_singular_minute():
    assert parse_schedule_command("s w every 1 minute") == ("s", "w", 60)


def test_parses_singular_hour():
    assert parse_schedule_command("s w every 1 hour") == ("s", "w", 3600)


def test_case_insensitive_every_and_unit():
    assert parse_schedule_command("s w EVERY 5 MINUTES") == ("s", "w", 300)


def test_rejects_missing_every_keyword():
    with pytest.raises(ValueError):
        parse_schedule_command("s w at 5 minutes")


def test_rejects_zero_interval():
    with pytest.raises(ValueError):
        parse_schedule_command("s w every 0 minutes")


def test_rejects_negative_interval():
    with pytest.raises(ValueError):
        parse_schedule_command("s w every -5 minutes")


def test_rejects_non_integer_interval():
    with pytest.raises(ValueError):
        parse_schedule_command("s w every abc minutes")


def test_rejects_unknown_unit():
    with pytest.raises(ValueError):
        parse_schedule_command("s w every 5 fortnights")


def test_rejects_malformed_token_count():
    with pytest.raises(ValueError):
        parse_schedule_command("s w every 5")
    with pytest.raises(ValueError):
        parse_schedule_command("s")
    with pytest.raises(ValueError):
        parse_schedule_command("")
