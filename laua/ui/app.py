"""Minimal Textual UI for Phase 1."""

from __future__ import annotations

import asyncio
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, RichLog

from laua.config import load_config
from laua.executor.audit import AuditLog
from laua.executor.pty_session import PtySession
from laua.ollama_client import OllamaClient
from laua.planner.orchestrator import Orchestrator
from laua.tools.core import register_core_tools
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
        self._registry = ToolRegistry()
        self._orchestrator: Orchestrator | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="log", highlight=True, markup=True)
        yield Input(placeholder="Ask LAUA anything...", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        await self._audit.init()
        register_core_tools(
            self._registry,
            self._session,
            confirm_fn=self._confirm,
            audit_fn=self._audit.record,
        )
        self._orchestrator = Orchestrator(
            ollama=self._ollama,
            registry=self._registry,
            model=self._cfg["ollama"]["default_model"],
        )
        log = self.query_one("#log", RichLog)

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
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.clear()
        log = self.query_one("#log", RichLog)
        log.write(Text(f"\n> {prompt}", style="bold cyan"))

        assert self._orchestrator is not None
        result = await self._orchestrator.run(prompt)

        for step in result.steps:
            if step.error:
                log.write(Text(f"[Step {step.step}] {step.tool_name} error: {step.error}", style="red"))
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

        # Simple async input via a one-shot Input widget replacement
        # For Phase 1 we use a basic approach: the input box becomes the confirm prompt
        inp = self.query_one("#prompt", Input)
        inp.placeholder = "y to confirm, anything else to cancel"

        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

        def on_confirm(event: Input.Submitted) -> None:
            future.set_result(event.value.strip().lower())
            event.input.clear()
            inp.placeholder = "Ask LAUA anything..."

        inp.action_submit = on_confirm  # type: ignore[attr-defined]
        response = await future
        return response == "y"
