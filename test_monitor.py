"""Test background monitor: silent, alert, and autonomous act-and-notify."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from textual.widgets import Static

from laua.planner.orchestrator import OrchestratorResult, StepResult
from laua.ui.app import LauaApp


def _monitor_lines(app: LauaApp) -> list[str]:
    return [w.content for w in app.query("Static") if "[monitor]" in str(w.content)]


async def main() -> None:
    app = LauaApp()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(3.0)
        assert app._recommender is not None
        assert app._orchestrator is not None

        # ── Pass 1: real thresholds → silent ─────────────────────────────────
        if app._monitor_timer:
            app._monitor_timer.stop()
        app._monitor_timer = app.set_interval(1.5, app._bg_poll)
        await pilot.pause(3.0)
        after_p1 = _monitor_lines(app)

        # ── Pass 2: _monitor_act with mocked orchestrator (fast, deterministic)
        if app._monitor_timer:
            app._monitor_timer.stop()

        fake_step = StepResult(
            step=1, tool_name="run_command",
            arguments={"args": ["find", "/", "-size", "+100M"]},
            result={"stdout": "/var/lib/docker/overlay2/abc123 4.2G\n/home/wzrd-pluto/Videos 12G\n", "exit_code": 0},
        )
        fake_result = OrchestratorResult(
            steps=[fake_step],
            final_response="Top disk consumers: /home/wzrd-pluto/Videos at 12 GB, /var/lib/docker/overlay2 at 4.2 GB.",
            model_used="qwen3.5:4b",
        )
        app._orchestrator.run = AsyncMock(return_value=fake_result)

        before_p2 = list(after_p1)
        await app._monitor_act(
            "Disk at 87.1% (61.24 GB free) — want me to find large files to clean up?"
        )
        await pilot.pause(0.5)
        after_p2 = _monitor_lines(app)

    print("\n=== Pass 1: real thresholds → should be silent ===")
    print("PASS: silent" if not after_p1 else f"UNEXPECTED: {after_p1}")

    print("\n=== Pass 2: autonomous act → acting message + result ===")
    new = [a for a in after_p2 if a not in before_p2]
    acting = [a for a in new if "acting autonomously" in str(a)]
    result = [a for a in new if "acting autonomously" not in str(a)]

    if acting:
        print(f"  acting : {acting[0]}")
    else:
        print("  FAIL: no 'acting autonomously' message")

    if result:
        print(f"  result : {result[0]}")
        print("PASS")
    else:
        print("  FAIL: no result message after orchestrator ran")
        print(f"  all monitor lines: {after_p2}")

    # Verify history was not polluted
    history_clean = len(app._orchestrator._history) == 0
    print(f"\n=== Conversation history clean: {'PASS' if history_clean else 'FAIL'} ===")

    # Verify flags are reset
    flags_clean = not app._monitor_acting and not app._orchestrator_busy
    print(f"=== State flags reset: {'PASS' if flags_clean else 'FAIL'} ===")


if __name__ == "__main__":
    asyncio.run(main())
