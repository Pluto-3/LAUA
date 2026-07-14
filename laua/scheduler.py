"""Parsing for the /schedule command — pure, no I/O, no Textual dependency."""

from __future__ import annotations

_USAGE = "Usage: /schedule <name> <workflow_name> every <N> <minutes|hours>"

_UNIT_SECONDS = {
    "minute": 60,
    "minutes": 60,
    "hour": 3600,
    "hours": 3600,
}


def parse_schedule_command(rest: str) -> tuple[str, str, int]:
    """'<name> <workflow_name> every <N> <minutes|hours>' -> (name, workflow_name, interval_seconds)."""
    tokens = rest.split()
    if len(tokens) != 5:
        raise ValueError(_USAGE)

    name, workflow_name, every_kw, count_str, unit = tokens

    if every_kw.lower() != "every":
        raise ValueError(_USAGE)

    try:
        count = int(count_str)
    except ValueError:
        raise ValueError(f"'{count_str}' is not a whole number. {_USAGE}") from None

    if count <= 0:
        raise ValueError(f"Interval must be positive, got {count}. {_USAGE}")

    unit_seconds = _UNIT_SECONDS.get(unit.lower())
    if unit_seconds is None:
        raise ValueError(f"Unknown unit '{unit}' — use minutes or hours. {_USAGE}")

    return name, workflow_name, count * unit_seconds
