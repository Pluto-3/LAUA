"""Command blacklist and confirmation-required classification."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Patterns that are unconditionally blocked (§2.3)
_BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+/\s*\*?$"),   # rm -rf / or rm -rf /*
    re.compile(r"rm\s+-[a-zA-Z]*f[a-zA-Z]*r?\s+/\s*\*?$"),
    re.compile(r"mkfs\."),
    re.compile(r"dd\s+.*of=/dev/sd"),
    re.compile(r":\s*\(\s*\)\s*\{"),                            # fork bomb
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"chmod\s+-R\s+777\s+/"),
    re.compile(r">\s*/dev/sda"),
]

# Sudo-required path prefixes (§2.2)
SUDO_PATHS = ("/etc/", "/var/", "/usr/", "/boot/", "/sys/", "/proc/")

# Confirmation-required command verbs / patterns
_CONFIRM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\b"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bkillall\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\bsystemctl\s+(stop|restart|disable)\b"),
    re.compile(r"\bapt(-get)?\s+(install|remove|purge)\b"),
    re.compile(r"\bpip\s+uninstall\b"),
    re.compile(r"\btruncate\b"),
    re.compile(r"\bmv\b.*\s+/"),
]


@dataclass(frozen=True)
class SafetyVerdict:
    blocked: bool
    requires_confirmation: bool
    requires_sudo: bool
    reason: str | None = None


def check_command(args: list[str]) -> SafetyVerdict:
    """
    Evaluate a command arg array for safety. Returns a SafetyVerdict.
    args[0] is the executable; never receives a shell string.
    """
    flat = " ".join(args)

    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(flat):
            return SafetyVerdict(
                blocked=True,
                requires_confirmation=False,
                requires_sudo=False,
                reason=f"Blocked by safety pattern: {pattern.pattern}",
            )

    needs_sudo = args[0] == "sudo" or any(flat.find(p) != -1 for p in SUDO_PATHS)
    needs_confirm = needs_sudo or any(p.search(flat) for p in _CONFIRM_PATTERNS)

    return SafetyVerdict(
        blocked=False,
        requires_confirmation=needs_confirm,
        requires_sudo=needs_sudo,
    )
