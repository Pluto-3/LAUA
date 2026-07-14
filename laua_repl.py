"""Headless LAUA session runner — one turn per invocation, history via MemoryStore."""

import asyncio
import sys

from laua.config import load_config
from laua.executor.audit import AuditLog
from laua.executor.pty_session import PtySession
from laua.memory.context import ContextManager
from laua.memory.store import MemoryStore
from laua.ollama_client import OllamaClient
from laua.planner.orchestrator import Orchestrator, StepResult
from laua.planner.router import ModelRouter
from laua.tools.core import register_core_tools
from laua.tools.docker_tool import register_docker_tools
from laua.tools.file_manager import register_file_tools
from laua.tools.registry import ToolRegistry


async def run_turn(prompt: str) -> None:
    cfg = load_config()

    ollama = OllamaClient(cfg["ollama"]["base_url"], cfg["ollama"]["request_timeout"])
    session = PtySession()
    audit = AuditLog(cfg["audit"]["log_path"])
    memory = MemoryStore(cfg["memory"]["db_path"])

    await audit.init()
    await memory.init()

    session_id = await memory.get_active_session()
    if session_id is None:
        session_id = await memory.create_session()
        history = []
    else:
        history = await memory.get_history(session_id)

    mem_cfg = cfg["memory"]
    context_mgr = ContextManager(
        model_max_tokens=mem_cfg.get("max_history_tokens", 4096),
        trigger_ratio=mem_cfg.get("context_window_trigger", 0.80),
    )
    router = ModelRouter(cfg.get("model_routing", {}))
    restricted = cfg["permissions"]["restricted_paths"]
    fm_cfg = cfg.get("file_manager", {})

    tools = ToolRegistry()

    async def auto_confirm(args: list[str], requires_sudo: bool = False) -> bool:
        print(f"  [confirm] {' '.join(args)} (auto-approved)", flush=True)
        return True

    register_core_tools(tools, session, confirm_fn=auto_confirm, audit_fn=audit.record)
    register_docker_tools(tools, ollama_client=ollama, confirm_fn=auto_confirm)
    register_file_tools(
        tools, confirm_fn=auto_confirm,
        audit_fn=audit.record,
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
    )

    def on_step(step: StepResult) -> None:
        status = "ok" if step.error is None else f"ERROR: {step.error}"
        print(f"  [step {step.step}] {step.tool_name}({step.arguments}) → {status}", flush=True)

    print(f"you> {prompt}", flush=True)
    result = await orchestrator.run(prompt, on_step=on_step)

    if result.error:
        print(f"laua> ERROR: {result.error}", flush=True)
    else:
        print(f"laua> {result.final_response}", flush=True)
        if result.hit_step_ceiling:
            print("  [hit step ceiling]", flush=True)

    # Persist this turn
    await memory.add_message(session_id, "user", prompt)
    await memory.add_message(session_id, "assistant", result.final_response)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python laua_repl.py '<prompt>'")
        sys.exit(1)
    prompt = " ".join(sys.argv[1:])
    asyncio.run(run_turn(prompt))
