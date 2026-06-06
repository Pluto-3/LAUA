"""Post-execution recommendation engine — checks thresholds and suggests follow-up actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from laua.monitor.system import SystemSnapshot
    from laua.planner.orchestrator import StepResult

# Process name prefixes → human-readable service label
_SERVICE_PATTERNS: dict[str, str] = {
    "celery": "Paperless/Celery workers",
    "celeryd": "Paperless/Celery workers",
    "node": "Node.js processes",
    "mysqld": "MySQL",
    "postgres": "PostgreSQL",
    "redis-server": "Redis",
    "mongod": "MongoDB",
    "java": "Java processes",
}
# Flag a service group when its combined RAM % hits this level
_PROCESS_GROUP_MEM_THRESHOLD = 1.5


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

            proc_rec = self._check_processes(result.get("top_processes", []))
            if proc_rec:
                return proc_rec

        return None

    def check_snapshot(self, snap: "SystemSnapshot") -> str | None:
        """Check a live SystemSnapshot — used by the background monitor."""
        if snap.disk_percent >= self._disk_critical:
            return f"Disk critically full at {snap.disk_percent:.1f}% — want me to find the largest files?"
        if snap.disk_percent >= self._disk_warn:
            return (
                f"Disk at {snap.disk_percent:.1f}% ({snap.disk_free_gb} GB free)"
                " — want me to find large files to clean up?"
            )
        # Only dig into processes when RAM is actually elevated
        if snap.memory_percent >= self._memory_warn:
            proc_rec = self._check_processes(snap.top_processes)
            if proc_rec:
                return f"RAM at {snap.memory_percent:.1f}% — {proc_rec}"
        if snap.memory_percent >= self._memory_critical:
            return f"RAM critically high at {snap.memory_percent:.1f}% — want me to show the top consumers?"
        if snap.memory_percent >= self._memory_warn:
            return f"RAM at {snap.memory_percent:.1f}% — want me to check which processes are consuming the most?"
        return None

    def _check_processes(self, processes: list[dict]) -> str | None:
        """Flag heavyweight known-service groups by combined RAM %."""
        if not processes:
            return None

        # display_name → (total_mem_pct, count)
        groups: dict[str, list] = {}
        for proc in processes:
            name = (proc.get("name") or "").lower()
            mem_pct = proc.get("memory_percent") or 0.0
            for pattern, display in _SERVICE_PATTERNS.items():
                if name == pattern or name.startswith(pattern):
                    if display not in groups:
                        groups[display] = [0.0, 0]
                    groups[display][0] += mem_pct
                    groups[display][1] += 1
                    break

        for display, (total_pct, count) in groups.items():
            if total_pct >= _PROCESS_GROUP_MEM_THRESHOLD:
                label = f"{count} process{'es' if count > 1 else ''}"
                return (
                    f"{display} ({label}, {total_pct:.1f}% RAM) are running"
                    " — want me to stop them?"
                )

        return None
