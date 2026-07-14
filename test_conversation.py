"""Replay the lost-session conversation to verify the three gaps are closed."""

import asyncio
import sys

from laua.config import load_config
from laua.executor.audit import AuditLog
from laua.executor.pty_session import PtySession
from laua.memory.context import ContextManager
from laua.memory.store import MemoryStore
from laua.monitor.recommendations import RecommendationEngine
from laua.ollama_client import OllamaClient
from laua.planner.orchestrator import Orchestrator, StepResult
from laua.planner.planner import Planner
from laua.planner.router import ModelRouter
from laua.tools.core import register_core_tools
from laua.tools.docker_tool import register_docker_tools
from laua.tools.file_manager import register_file_tools
from laua.tools.registry import ToolRegistry


TURNS = [
    "what processes are on my ram",
    "stop the paperless broker container, then check what other containers are still running",
    "ram status",
]


async def main() -> None:
    cfg = load_config()
    ollama = OllamaClient(cfg["ollama"]["base_url"], cfg["ollama"]["request_timeout"])
    session = PtySession()
    audit = AuditLog(cfg["audit"]["log_path"])
    memory = MemoryStore(cfg["memory"]["db_path"])
    await audit.init()
    await memory.init()

    session_id = await memory.create_session()
    history: list = []

    mem_cfg = cfg["memory"]
    context_mgr = ContextManager(
        model_max_tokens=mem_cfg.get("max_history_tokens", 4096),
        trigger_ratio=mem_cfg.get("context_window_trigger", 0.80),
    )
    router = ModelRouter(cfg.get("model_routing", {}))
    mon_cfg = cfg.get("monitor", {})
    recommender = RecommendationEngine(
        disk_warn=mon_cfg.get("disk_alert_threshold", 85.0),
        disk_critical=95.0,
        memory_warn=mon_cfg.get("memory_alert_threshold", 88.0),
        memory_critical=95.0,
    )
    planner = Planner(ollama, cfg["ollama"]["default_model"])
    restricted = cfg["permissions"]["restricted_paths"]
    fm_cfg = cfg.get("file_manager", {})

    tools = ToolRegistry()

    async def auto_confirm(args: list[str], requires_sudo: bool = False) -> bool:
        print(f"  [confirm] {' '.join(args)} → auto-approved", flush=True)
        return True

    register_core_tools(tools, session, confirm_fn=auto_confirm, audit_fn=audit.record)
    register_docker_tools(tools, ollama_client=ollama, confirm_fn=auto_confirm)
    register_file_tools(
        tools, confirm_fn=auto_confirm, audit_fn=audit.record,
        restricted_paths=restricted,
        max_search_results=fm_cfg.get("max_search_results", 50),
        max_write_bytes=fm_cfg.get("max_write_bytes", 10 * 1024 * 1024),
    )

    orchestrator = Orchestrator(
        ollama=ollama,
        registry=tools,
        model=cfg["ollama"]["default_model"],
        history=history,
        context_manager=context_mgr,
        model_router=router,
        planner=planner,
        recommendation_engine=recommender,
    )

    def on_step(step: StepResult) -> None:
        status = "ok" if step.error is None else f"ERROR: {step.error}"
        print(f"  [step {step.step}] {step.tool_name}({step.arguments}) → {status}", flush=True)

    sep = "─" * 60
    for prompt in TURNS:
        print(f"\n{sep}", flush=True)
        print(f"you> {prompt}", flush=True)
        print(sep, flush=True)

        result = await orchestrator.run(prompt, on_step=on_step)

        if result.error:
            print(f"laua> ERROR: {result.error}", flush=True)
        else:
            print(f"laua> {result.final_response}", flush=True)

        if result.recommendation:
            print(f"  [→ rec] {result.recommendation}", flush=True)
        else:
            print("  [→ rec] (none)", flush=True)

        await memory.add_message(session_id, "user", prompt)
        await memory.add_message(session_id, "assistant", result.final_response)

    await memory.end_session(session_id)
    await ollama.close()


if __name__ == "__main__":
    asyncio.run(main())
