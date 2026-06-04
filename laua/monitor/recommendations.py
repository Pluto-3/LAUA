"""Post-execution recommendation engine — checks thresholds and suggests follow-up actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from laua.planner.orchestrator import StepResult


class RecommendationEngine:
    def __init__(
        self,
        disk_warn: float = 85.0,
        disk_critical: float = 95.0,
        memory_warn: float = 88.0,
        memory_critical: float = 95.0,
    ) -> None:
        self._disk_warn = disk_warn
        self._disk_critical = disk_critical
        self._memory_warn = memory_warn
        self._memory_critical = memory_critical

    def check(self, steps: list[Any]) -> str | None:
        """Scan completed steps for threshold violations; return a suggestion or None."""
        for step in steps:
            if not isinstance(step.result, dict):
                continue
            result = step.result

            disk = result.get("disk", {})
            if isinstance(disk, dict):
                pct = disk.get("percent", 0)
                free = disk.get("free_gb", 0)
                if pct >= self._disk_critical:
                    return f"Disk is critically full at {pct}% — want me to find the largest files?"
                if pct >= self._disk_warn:
                    return f"Disk at {pct}% ({free} GB free) — want me to find large files to clean up?"

            mem = result.get("memory", {})
            if isinstance(mem, dict):
                pct = mem.get("percent", 0)
                if pct >= self._memory_critical:
                    return f"RAM critically high at {pct}% — want me to show which processes are using the most?"
                if pct >= self._memory_warn:
                    return f"RAM at {pct}% — want me to check which processes are consuming the most memory?"

        return None
