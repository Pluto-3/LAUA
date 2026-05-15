"""Textual UI — Phase 2: memory, model router, context manager, Docker, file tools."""

from __future__ import annotations

import asyncio

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, RichLog

from laua.config import load_config
from laua.executor.audit import AuditLog
from laua.executor.pty_session import PtySession
from laua.memory.context import ContextManager
from laua.memory.store import MemoryStore
from laua.ollama_client import OllamaClient
from laua.planner.orchestrator import Orchestrator
from laua.planner.router import ModelRouter
from laua.tools.core import register_core_tools
from laua.tools.docker_tool import register_docker_tools
from laua.tools.file_manager import register_file_tools
from laua.tools.plugin_loader import load_plugins
from laua.tools.registry import ToolRegistry


class LauaApp(App):
    TITLE = "LAUA — Local Autonomous Utility Agent"
    CSS = """
    RichLog {
        height: 1fr;
        border: solid $primary;
    }
    Input {
        dock: bottom;
    }
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._cfg = load_config()
        self._ollama = OllamaClient(
            self._cfg["ollama"]["base_url"],
            self._cfg["ollama"]["request_timeout"],
        )
        self._session = PtySession()
        self._audit = AuditLog(self._cfg["audit"]["log_path"])
        self._memory = MemoryStore(self._cfg["memory"]["db_path"])
        self._registry = ToolRegistry()
        self._orchestrator: Orchestrator | None = None
        self._session_id: int | None = None
        self._pending_confirm: asyncio.Future[str] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="log", highlight=True, markup=True)
        yield Input(placeholder="Ask LAUA anything...", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        await self._audit.init()
        await self._memory.init()

        # Crash recovery: resume last active session if available
        self._session_id = await self._memory.get_active_session()
        if self._session_id is None:
            self._session_id = await self._memory.create_session()
            prior_history: list = []
        else:
            prior_history = await self._memory.get_history(self._session_id)

        mem_cfg = self._cfg["memory"]
        context_mgr = ContextManager(
            model_max_tokens=mem_cfg.get("max_history_tokens", 4096),
            trigger_ratio=mem_cfg.get("context_window_trigger", 0.80),
        )
        router = ModelRouter(self._cfg.get("model_routing", {}))

        restricted = self._cfg["permissions"]["restricted_paths"]
        fm_cfg = self._cfg.get("file_manager", {})

        register_core_tools(
            self._registry, self._session,
            confirm_fn=self._confirm, audit_fn=self._audit.record,
        )
        register_docker_tools(
            self._registry, ollama_client=self._ollama, confirm_fn=self._confirm,
        )
        register_file_tools(
            self._registry, confirm_fn=self._confirm,
            restricted_paths=restricted,
            max_search_results=fm_cfg.get("max_search_results", 50),
            max_write_bytes=fm_cfg.get("max_write_bytes", 10 * 1024 * 1024),
        )
        load_plugins(self._registry, self._session, self._confirm, self._audit.record)

        self._orchestrator = Orchestrator(
            ollama=self._ollama,
            registry=self._registry,
            model=self._cfg["ollama"]["default_model"],
            history=prior_history,
            context_manager=context_mgr,
            model_router=router,
        )

        log = self.query_one("#log", RichLog)
        if prior_history:
            log.write(Text(f"Resumed session #{self._session_id}.", style="dim"))

        healthy = await self._ollama.health_check()
        if healthy:
            log.write(Text("Ollama is reachable.", style="green"))
        else:
            log.write(Text(
                f"Warning: Ollama not reachable at {self._cfg['ollama']['base_url']}. "
                "Check that the Docker container is running.",
                style="yellow",
            ))
        log.write(Text("Ready. Type a request and press Enter.", style="dim"))

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        # If a confirmation is pending, route this input to the waiting future
        if self._pending_confirm is not None and not self._pending_confirm.done():
            self._pending_confirm.set_result(event.value)
            event.input.clear()
            return

        prompt = event.value.strip()
        if not prompt:
            return
        event.input.clear()
        log = self.query_one("#log", RichLog)
        log.write(Text(f"\n> {prompt}", style="bold cyan"))

        assert self._orchestrator is not None
        result = await self._orchestrator.run(prompt)

        # Persist conversation turn to memory
        if self._session_id is not None:
            await self._memory.add_message(self._session_id, "user", prompt)
            await self._memory.add_message(
                self._session_id, "assistant", result.final_response
            )

        for step in result.steps:
            if step.error:
                log.write(Text(
                    f"[Step {step.step}] {step.tool_name} error: {step.error}", style="red"
                ))
            else:
                log.write(Text(f"[Step {step.step}] {step.tool_name}", style="dim"))
                if isinstance(step.result, dict):
                    if stdout := step.result.get("stdout"):
                        log.write(stdout.rstrip())
                    if stderr := step.result.get("stderr"):
                        log.write(Text(stderr.rstrip(), style="yellow"))

        if result.hit_step_ceiling:
            log.write(Text(result.final_response, style="yellow"))
        elif result.error:
            log.write(Text(f"Error: {result.error}", style="red"))
        elif result.final_response:
            log.write(Text(result.final_response, style="green"))

    async def _confirm(self, args: list[str], requires_sudo: bool = False) -> bool:
        """Inline confirmation prompt in the TUI."""
        log = self.query_one("#log", RichLog)
        display = " ".join(args)
        prefix = "[SUDO] " if requires_sudo else ""
        log.write(Text(f"\n{prefix}About to run: {display}", style="yellow bold"))
        log.write(Text("Confirm? (y/N): ", style="yellow"))

        inp = self.query_one("#prompt", Input)
        inp.placeholder = "y to confirm, anything else to cancel"

        self._pending_confirm = asyncio.get_event_loop().create_future()
        response = await self._pending_confirm
        self._pending_confirm = None

        inp.placeholder = "Ask LAUA anything..."
        return response.strip().lower() == "y"

    async def on_unmount(self) -> None:
        if self._session_id is not None:
            await self._memory.end_session(self._session_id)
        await self._ollama.close()
