"""Textual UI — terminal-style single-pane layout."""

from __future__ import annotations

import asyncio
import time

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.timer import Timer
from textual.widgets import Input, Static

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

from laua.config import load_config
from laua.executor.audit import AuditLog
from laua.executor.pty_session import PtySession
from laua.memory.context import ContextManager
from laua.memory.schedules import SchedulesStore
from laua.memory.store import MemoryStore
from laua.memory.workflows import WorkflowStore
from laua.monitor.recommendations import RecommendationEngine
from laua.monitor.system import SystemMonitor, detect_anomalies
from laua.ollama_client import OllamaClient
from laua.planner.orchestrator import Orchestrator, StepResult
from laua.planner.planner import Planner
from laua.planner.router import ModelRouter
from laua.scheduler import parse_schedule_command
from laua.tools.core import register_core_tools
from laua.tools.docker_tool import register_docker_tools
from laua.tools.file_manager import register_file_tools
from laua.tools.plugin_loader import load_plugins
from laua.tools.registry import ToolRegistry


class _LogPane(VerticalScroll):
    can_focus = False


class _PromptInput(Input):
    DEFAULT_CSS = """
    _PromptInput {
        background: ansi_default;
        color: $text;
        border: none;
        padding: 1 2;
    }
    _PromptInput:focus {
        border: none;
    }
    """


class LauaApp(App):
    TITLE = "LAUA — Local Autonomous Utility Agent"
    ENABLE_MOUSE = False
    CSS = """
    Screen {
        background: ansi_default;
    }
    #log {
        height: 1fr;
        background: ansi_default;
        border: none;
        padding: 0 2;
    }
    #log Static {
        background: ansi_default;
        width: 1fr;
        height: auto;
        padding: 0;
    }
    .user-msg {
        color: $accent;
        text-style: bold;
        margin-top: 1;
    }
    .response {
        color: $text;
    }
    .step {
        color: $text-muted;
    }
    .error-msg {
        color: $error;
    }
    .warn-msg {
        color: $warning;
    }
    .dim-msg {
        color: $text-muted;
    }
    .banner {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    .banner-sub {
        color: $text-muted;
        margin-bottom: 1;
    }
    .plan-msg {
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
    }
    .recommend-msg {
        color: $accent;
        margin-top: 1;
    }
    .dry-run-msg {
        color: $warning;
        text-style: bold;
    }
    .monitor-alert {
        color: $warning;
        margin-top: 1;
    }
    Input {
        dock: bottom;
        border: none;
        padding: 1 2;
    }
    #status {
        dock: bottom;
        height: 1;
        background: ansi_default;
        color: $text-muted;
        padding: 0 2;
    }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
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
        self._tools = ToolRegistry()
        self._orchestrator: Orchestrator | None = None
        self._session_id: int | None = None
        self._pending_confirm: asyncio.Future[str] | None = None
        self._stream_buffer: str = ""
        self._stream_widget: Static | None = None
        self._stream_widget_mounted: bool = False
        self._show_plan: bool = True
        self._current_model: str = ""
        self._spinner_idx: int = 0
        self._spinner_timer: Timer | None = None
        self._think_start: float = 0.0
        self._think_step: int = 0
        self._think_tool: str = ""
        self._step_start: float = 0.0
        self._cmd_timeout: int = 30
        self._context_mgr: ContextManager | None = None
        self._shell_active: bool = False
        self._workflows: WorkflowStore | None = None
        self._recording: bool = False
        self._record_name: str = ""
        self._recording_steps: list[dict] = []
        self._last_response: str = ""
        self._sys_monitor: SystemMonitor | None = None
        self._monitor_timer: Timer | None = None
        self._recommender: RecommendationEngine | None = None
        self._orchestrator_busy: bool = False
        self._monitor_acting: bool = False
        self._schedules: SchedulesStore | None = None
        self._scheduler_timer: Timer | None = None
        self._scheduler_acting: bool = False

    def compose(self) -> ComposeResult:
        yield _LogPane(id="log")
        yield _PromptInput(placeholder="Ask LAUA anything...", id="prompt")
        yield Static("", id="status")

    async def on_mount(self) -> None:
        await self._audit.init()
        await self._memory.init()

        mem_cfg = self._cfg["memory"]
        workflows_path = mem_cfg.get("workflows_db_path", "~/.laua/workflows.db")
        self._workflows = WorkflowStore(workflows_path)
        await self._workflows.init()

        schedules_path = mem_cfg.get("schedules_db_path", "~/.laua/schedules.db")
        self._schedules = SchedulesStore(schedules_path)
        await self._schedules.init()

        self._show_plan = self._cfg.get("ui", {}).get("show_plan", True)
        self._current_model = self._cfg["ollama"]["default_model"]
        self._cmd_timeout = self._cfg.get("executor", {}).get("command_timeout", 30)

        self._session_id = await self._memory.get_active_session()
        if self._session_id is None:
            self._session_id = await self._memory.create_session()
            prior_history: list = []
        else:
            prior_history = await self._memory.get_history(self._session_id)
            # Don't resume a context-heavy session — start fresh to avoid silent failures
            _ctx_check = ContextManager(
                model_max_tokens=mem_cfg.get("max_history_tokens", 4096),
                trigger_ratio=0.40,
            )
            if _ctx_check.should_compress(prior_history):
                await self._memory.end_session(self._session_id)
                self._session_id = await self._memory.create_session()
                prior_history = []

        context_mgr = ContextManager(
            model_max_tokens=mem_cfg.get("max_history_tokens", 4096),
            trigger_ratio=mem_cfg.get("context_window_trigger", 0.80),
        )
        self._context_mgr = context_mgr
        router = ModelRouter(self._cfg.get("model_routing", {}))
        planner = Planner(self._ollama, self._cfg["ollama"]["default_model"])
        mon_cfg = self._cfg.get("monitor", {})
        recommender = RecommendationEngine(
            disk_warn=mon_cfg.get("disk_alert_threshold", 85.0),
            disk_critical=95.0,
            memory_warn=mon_cfg.get("memory_alert_threshold", 88.0),
            memory_critical=95.0,
        )
        self._recommender = recommender
        restricted = self._cfg["permissions"]["restricted_paths"]
        fm_cfg = self._cfg.get("file_manager", {})

        register_core_tools(
            self._tools, self._session,
            confirm_fn=self._confirm, audit_fn=self._audit.record,
        )
        register_docker_tools(
            self._tools, ollama_client=self._ollama, confirm_fn=self._confirm,
        )
        register_file_tools(
            self._tools, confirm_fn=self._confirm,
            audit_fn=self._audit.record,
            restricted_paths=restricted,
            max_search_results=fm_cfg.get("max_search_results", 50),
            max_write_bytes=fm_cfg.get("max_write_bytes", 10 * 1024 * 1024),
        )
        load_plugins(self._tools, self._session, self._confirm, self._audit.record)

        self._orchestrator = Orchestrator(
            ollama=self._ollama,
            registry=self._tools,
            model=self._cfg["ollama"]["default_model"],
            history=prior_history,
            context_manager=context_mgr,
            model_router=router,
            planner=planner,
            recommendation_engine=recommender,
        )

        log = self.query_one("#log", _LogPane)

        if not prior_history:
            await log.mount(Static(
                "LAUA  ·  Local Autonomous Utility Agent  ·  v0.1.0",
                classes="banner",
            ))
            await log.mount(Static(
                f"Created by wzrdpluto  ·  AI infrastructure & automation engineer  ·  github.com/Pluto-3",
                classes="banner-sub",
            ))
            await log.mount(Static(
                f"Manage your Ubuntu workstation in plain language — "
                f"system stats, commands, Docker, files. Just ask.  "
                f"·  Model: {self._current_model}  ·  ctrl+q to quit",
                classes="dim-msg",
            ))
        else:
            await log.mount(Static(
                f"LAUA  ·  Resumed session #{self._session_id}  ·  {self._current_model}",
                classes="banner",
            ))

        healthy = await self._ollama.health_check()
        if not healthy:
            await log.mount(Static(
                f"Warning: Ollama not reachable at {self._cfg['ollama']['base_url']}.",
                classes="warn-msg",
            ))

        poll_interval = mon_cfg.get("poll_interval", 60)
        self._sys_monitor = SystemMonitor(
            gpu_enabled=mon_cfg.get("gpu_enabled", True),
            thermal_enabled=False,
        )
        await self._sys_monitor.refresh_excluded_pids()
        self._monitor_timer = self.set_interval(poll_interval, self._bg_poll)

        sched_cfg = self._cfg.get("scheduler", {})
        if sched_cfg.get("enabled", True):
            self._scheduler_timer = self.set_interval(
                sched_cfg.get("check_interval_seconds", 30), self._scheduler_tick
            )

        self._set_status_idle()
        self.query_one("#prompt", _PromptInput).focus()

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        s = int(seconds)
        return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"

    def _ctx_pct(self) -> int:
        if self._orchestrator is None or self._context_mgr is None:
            return 0
        tokens = self._context_mgr.estimate_tokens(self._orchestrator._history)
        return min(int(tokens / self._context_mgr.model_max_tokens * 100), 100)

    def _cwd_display(self) -> str:
        from pathlib import Path
        cwd = self._session.cwd
        try:
            rel = Path(cwd).relative_to(Path.home())
            display = f"~/{rel}" if str(rel) != "." else "~"
        except ValueError:
            display = cwd
        return display if len(display) <= 32 else "…" + display[-30:]

    def _set_status_idle(self) -> None:
        parts = [f"model: {self._current_model}"]
        ctx = self._ctx_pct()
        if ctx > 0:
            ctx_str = f"ctx {ctx}%"
            if ctx >= 50:
                ctx_str += " ⚠ /reset to clear"
            parts.append(ctx_str)
        if self._shell_active:
            parts.append(self._cwd_display())
        parts.append("^q quit")
        self.query_one("#status", Static).update("  ·  ".join(parts))

    def _start_spinner(self) -> None:
        self._spinner_idx = 0
        self._think_start = time.monotonic()
        self._step_start = time.monotonic()
        self._think_step = 0
        self._think_tool = ""
        self._spinner_timer = self.set_interval(0.1, self._tick_spinner)

    def _tick_spinner(self) -> None:
        frame = _SPINNER[self._spinner_idx % len(_SPINNER)]
        self._spinner_idx += 1
        now = time.monotonic()
        total = now - self._think_start
        parts = [f"{frame} {self._think_tool or 'thinking'} [{self._current_model}]"]
        if self._think_step:
            parts.append(f"step {self._think_step}")
            step_s = now - self._step_start
            step_str = self._fmt_elapsed(step_s)
            if self._think_tool == "run_command" and step_s > self._cmd_timeout * 0.8:
                step_str += f" ⚠ {self._cmd_timeout}s limit"
            parts.append(step_str)
        parts.append(f"total {self._fmt_elapsed(total)}")
        self.query_one("#status", Static).update("  ·  ".join(parts))

    def _stop_spinner(self, model: str | None = None) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
        if model:
            self._current_model = model
        elapsed = self._fmt_elapsed(time.monotonic() - self._think_start)
        steps = self._think_step
        self.query_one("#status", Static).update(
            f"✓ done in {elapsed}" + (f"  ·  {steps} steps" if steps else "")
        )
        self.set_timer(2.0, self._set_status_idle)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._pending_confirm is not None and not self._pending_confirm.done():
            self._pending_confirm.set_result(event.value)
            event.input.clear()
            return

        prompt = event.value.strip()
        if not prompt:
            return
        event.input.clear()

        log = self.query_one("#log", _LogPane)

        if prompt == "/clear":
            await log.query("Static").remove()
            return


        if prompt == "/reset":
            await log.query("Static").remove()
            assert self._orchestrator is not None
            self._orchestrator._history.clear()
            if self._session_id is not None:
                await self._memory.end_session(self._session_id)
            self._session_id = await self._memory.create_session()
            self._shell_active = False
            self._set_status_idle()
            await log.mount(Static("Session reset — history cleared.", classes="dim-msg"))
            log.scroll_end(animate=False)
            return

        if prompt == "/workflows":
            asyncio.create_task(self._cmd_list_workflows())
            return

        if prompt.startswith("/record "):
            name = prompt[8:].strip()
            if name:
                self._recording = True
                self._record_name = name
                self._recording_steps = []
                await log.mount(Static(
                    f"Recording '{name}' — run your commands, then type /stop to save.",
                    classes="dim-msg",
                ))
                log.scroll_end(animate=False)
            return

        if prompt == "/stop":
            if self._recording and self._recording_steps:
                assert self._workflows is not None
                await self._workflows.save(self._record_name, self._recording_steps)
                await log.mount(Static(
                    f"Saved workflow '{self._record_name}' ({len(self._recording_steps)} steps).",
                    classes="dim-msg",
                ))
            elif self._recording:
                await log.mount(Static("Nothing recorded — workflow not saved.", classes="warn-msg"))
            else:
                await log.mount(Static("Not currently recording.", classes="warn-msg"))
            self._recording = False
            self._record_name = ""
            self._recording_steps = []
            log.scroll_end(animate=False)
            return

        if prompt.startswith("/run "):
            name = prompt[5:].strip()
            if name:
                asyncio.create_task(self._replay_workflow(name))
            return

        if prompt == "/schedules":
            asyncio.create_task(self._cmd_list_schedules())
            return

        if prompt.startswith("/schedule-enable "):
            asyncio.create_task(self._cmd_set_schedule_enabled(prompt[17:].strip(), True))
            return

        if prompt.startswith("/schedule-disable "):
            asyncio.create_task(self._cmd_set_schedule_enabled(prompt[18:].strip(), False))
            return

        if prompt.startswith("/schedule-delete "):
            asyncio.create_task(self._cmd_delete_schedule(prompt[17:].strip()))
            return

        if prompt.startswith("/schedule "):
            asyncio.create_task(self._cmd_create_schedule(prompt[10:].strip()))
            return

        dry_run = False
        actual_prompt = prompt
        if prompt.lower().startswith("/dry-run "):
            dry_run = True
            actual_prompt = prompt[9:].strip()
            await log.mount(Static("[DRY RUN] Simulating — no destructive actions will execute.", classes="dry-run-msg"))
            log.scroll_end(animate=False)

        event.input.disabled = True

        await log.mount(Static(f"> {actual_prompt}", classes="user-msg"))
        log.scroll_end(animate=False)

        self._stream_buffer = ""
        self._stream_widget = None
        self._stream_widget_mounted = False
        assert self._orchestrator is not None
        asyncio.create_task(self._process_request(actual_prompt, dry_run=dry_run))

    async def _process_request(self, prompt: str, dry_run: bool = False) -> None:
        log = self.query_one("#log", _LogPane)
        inp = self.query_one("#prompt", _PromptInput)

        self._orchestrator_busy = True
        self._start_spinner()
        try:
            result = await self._orchestrator.run(
                prompt,
                on_step=self._on_step,
                on_step_start=self._on_step_start,
                on_token=self._on_token,
                on_plan=self._on_plan,
                dry_run=dry_run,
            )
        except Exception as exc:
            self._orchestrator_busy = False
            self._stop_spinner()
            await log.mount(Static(f"Error: {exc}", classes="error-msg"))
            log.scroll_end(animate=False)
            inp.disabled = False
            inp.focus()
            return

        self._orchestrator_busy = False
        self._stop_spinner(result.model_used or None)

        if self._session_id is not None:
            await self._memory.add_message(self._session_id, "user", prompt)
            await self._memory.add_message(
                self._session_id, "assistant", result.final_response
            )

        if result.hit_step_ceiling:
            self._last_response = result.final_response
            await log.mount(Static(result.final_response, classes="warn-msg"))
            log.scroll_end(animate=False)
        elif result.error:
            await log.mount(Static(f"Error: {result.error}", classes="error-msg"))
            log.scroll_end(animate=False)
        elif self._stream_widget and self._stream_widget_mounted and result.final_response:
            # Streamed content was displayed raw — replace with the stripped final response
            self._last_response = result.final_response
            self._stream_widget.update(result.final_response)
            log.scroll_end(animate=False)
        elif not self._stream_widget and result.final_response:
            self._last_response = result.final_response
            await log.mount(Static(result.final_response, classes="response"))
            log.scroll_end(animate=False)

        if result.recommendation:
            await log.mount(Static(f"→ {result.recommendation}", classes="recommend-msg"))
            log.scroll_end(animate=False)

        self._stream_widget = None
        self._stream_buffer = ""
        self._stream_widget_mounted = False
        inp.disabled = False
        inp.focus()

    def _on_step_start(self, tool_name: str, step: int) -> None:
        self._think_tool = tool_name
        self._think_step = step
        self._step_start = time.monotonic()

    def _on_plan(self, plan: list[dict]) -> None:
        asyncio.create_task(self._mount_plan(plan))

    async def _mount_plan(self, plan: list[dict]) -> None:
        log = self.query_one("#log", _LogPane)
        lines = "  ".join(f"{s['step']}. {s['description']}" for s in plan)
        await log.mount(Static(f"Plan: {lines}", classes="plan-msg"))
        log.scroll_end(animate=False)

    def _on_step(self, step: StepResult) -> None:
        self._think_step = step.step
        self._think_tool = step.tool_name
        self._step_start = time.monotonic()
        if step.tool_name == "run_command" and isinstance(step.result, dict):
            self._shell_active = True
            if cwd := step.result.get("cwd"):
                self._session.cwd = cwd
        if self._recording and not step.error:
            self._recording_steps.append({
                "tool_name": step.tool_name,
                "arguments": step.arguments,
            })
        if self._show_plan:
            asyncio.create_task(self._mount_step(step))

    async def _mount_step(self, step: StepResult) -> None:
        log = self.query_one("#log", _LogPane)
        if step.error:
            await log.mount(Static(
                f"[Step {step.step}] {step.tool_name}: {step.error}", classes="error-msg"
            ))
        else:
            lines = [f"[Step {step.step}] {step.tool_name}"]
            if isinstance(step.result, dict):
                if stdout := step.result.get("stdout"):
                    lines.append(stdout.rstrip())
                if stderr := step.result.get("stderr"):
                    lines.append(stderr.rstrip())
            await log.mount(Static("\n".join(lines), classes="step"))
        log.scroll_end(animate=False)

    def _on_token(self, token: str) -> None:
        self._stream_buffer += token
        if self._stream_widget is None:
            self._stream_widget = Static("", classes="response")
            asyncio.create_task(self._mount_stream_widget())
        elif self._stream_widget_mounted:
            self._stream_widget.update(self._stream_buffer)
            self.query_one("#log", _LogPane).scroll_end(animate=False)

    async def _mount_stream_widget(self) -> None:
        log = self.query_one("#log", _LogPane)
        await log.mount(self._stream_widget)
        self._stream_widget_mounted = True
        if self._stream_widget is not None:
            self._stream_widget.update(self._stream_buffer)
        log.scroll_end(animate=False)

    async def _cmd_list_workflows(self) -> None:
        log = self.query_one("#log", _LogPane)
        assert self._workflows is not None
        workflows = await self._workflows.list_workflows()
        if not workflows:
            await log.mount(Static("No saved workflows.", classes="dim-msg"))
        else:
            lines = ["Saved workflows:"]
            for w in workflows:
                lines.append(f"  {w['name']}  (run {w['run_count']}x)")
            await log.mount(Static("\n".join(lines), classes="dim-msg"))
        log.scroll_end(animate=False)

    async def _cmd_list_schedules(self) -> None:
        log = self.query_one("#log", _LogPane)
        assert self._schedules is not None
        schedules = await self._schedules.list_schedules()
        if not schedules:
            await log.mount(Static("No schedules.", classes="dim-msg"))
        else:
            lines = ["Schedules:"]
            for s in schedules:
                state = "enabled" if s["enabled"] else "disabled"
                lines.append(
                    f"  {s['name']} -> {s['workflow_name']}"
                    f"  every {s['interval_seconds']}s  ({state}, run {s['run_count']}x,"
                    f" next {s['next_run']})"
                )
            await log.mount(Static("\n".join(lines), classes="dim-msg"))
        log.scroll_end(animate=False)

    async def _cmd_create_schedule(self, rest: str) -> None:
        log = self.query_one("#log", _LogPane)
        assert self._schedules is not None
        assert self._workflows is not None
        try:
            name, workflow_name, interval_seconds = parse_schedule_command(rest)
        except ValueError as exc:
            await log.mount(Static(str(exc), classes="error-msg"))
            log.scroll_end(animate=False)
            return

        known = {w["name"] for w in await self._workflows.list_workflows()}
        if workflow_name not in known:
            await log.mount(Static(
                f"No workflow named '{workflow_name}'. Record it first with /record.",
                classes="error-msg",
            ))
            log.scroll_end(animate=False)
            return

        await self._schedules.create(name, workflow_name, interval_seconds)
        await log.mount(Static(
            f"Scheduled '{name}' -> '{workflow_name}' every {interval_seconds}s.",
            classes="dim-msg",
        ))
        log.scroll_end(animate=False)

    async def _cmd_set_schedule_enabled(self, name: str, enabled: bool) -> None:
        log = self.query_one("#log", _LogPane)
        assert self._schedules is not None
        if not name:
            return
        found = await self._schedules.set_enabled(name, enabled)
        state = "enabled" if enabled else "disabled"
        if found:
            await log.mount(Static(f"Schedule '{name}' {state}.", classes="dim-msg"))
        else:
            await log.mount(Static(f"No schedule named '{name}'.", classes="error-msg"))
        log.scroll_end(animate=False)

    async def _cmd_delete_schedule(self, name: str) -> None:
        log = self.query_one("#log", _LogPane)
        assert self._schedules is not None
        if not name:
            return
        found = await self._schedules.delete(name)
        if found:
            await log.mount(Static(f"Deleted schedule '{name}'.", classes="dim-msg"))
        else:
            await log.mount(Static(f"No schedule named '{name}'.", classes="error-msg"))
        log.scroll_end(animate=False)

    async def _replay_workflow(self, name: str) -> None:
        log = self.query_one("#log", _LogPane)
        inp = self.query_one("#prompt", _PromptInput)
        assert self._workflows is not None

        steps = await self._workflows.load(name)
        if steps is None:
            await log.mount(Static(f"No workflow named '{name}'.", classes="error-msg"))
            log.scroll_end(animate=False)
            return

        await log.mount(Static(f"> /run {name}", classes="user-msg"))
        await log.mount(Static(f"Replaying '{name}' ({len(steps)} steps)…", classes="dim-msg"))
        log.scroll_end(animate=False)
        inp.disabled = True

        self._start_spinner()
        for i, step_data in enumerate(steps, start=1):
            tool_name = step_data["tool_name"]
            arguments = step_data["arguments"]
            self._on_step_start(tool_name, i)
            try:
                result = await self._tools.dispatch(tool_name, arguments)
                sr = StepResult(step=i, tool_name=tool_name, arguments=arguments, result=result)
            except Exception as exc:
                sr = StepResult(step=i, tool_name=tool_name, arguments=arguments, result=None, error=str(exc))
            self._on_step(sr)

        self._stop_spinner()
        await log.mount(Static(f"Workflow '{name}' complete.", classes="dim-msg"))
        log.scroll_end(animate=False)
        inp.disabled = False
        inp.focus()

    async def _confirm(self, args: list[str], requires_sudo: bool = False) -> bool:
        if self._monitor_acting or self._scheduler_acting:
            prefix = "[monitor]" if self._monitor_acting else "[scheduler]"
            log = self.query_one("#log", _LogPane)
            await log.mount(Static(
                f"{prefix} auto-approved: {' '.join(args)}", classes="dim-msg"
            ))
            log.scroll_end(animate=False)
            return True
        log = self.query_one("#log", _LogPane)
        display = " ".join(args)
        prefix = "[SUDO] " if requires_sudo else ""
        await log.mount(Static(f"{prefix}About to run: {display}", classes="warn-msg"))
        await log.mount(Static("Confirm? (y/N):", classes="warn-msg"))
        log.scroll_end(animate=False)

        inp = self.query_one("#prompt", _PromptInput)
        inp.disabled = False
        inp.placeholder = "y to confirm, anything else to cancel"
        inp.focus()

        self._pending_confirm = asyncio.get_running_loop().create_future()
        response = await self._pending_confirm
        self._pending_confirm = None

        inp.disabled = True
        inp.placeholder = "Ask LAUA anything..."
        return response.strip().lower() == "y"

    async def key_ctrl_c(self) -> None:
        if not self._last_response:
            return
        import subprocess
        data = self._last_response.encode()

        def _run_xclip() -> None:
            try:
                subprocess.run(["xclip", "-selection", "clipboard"], input=data, capture_output=True)
            except FileNotFoundError:
                subprocess.run(["xsel", "--clipboard", "--input"], input=data, capture_output=True)

        await asyncio.to_thread(_run_xclip)

    @staticmethod
    def _is_actionable(suggestion: str) -> bool:
        return "want me to stop them?" in suggestion or "want me to find" in suggestion

    @staticmethod
    def _to_directive(suggestion: str) -> str:
        s = suggestion
        s = s.replace("— want me to stop them?", "— stop them now. After stopping, verify they are gone.")
        s = s.replace("want me to find large files to clean up?", "find the top 10 largest files and list them with sizes.")
        s = s.replace("want me to find the largest files?", "find the top 10 largest files and list them with sizes.")
        return s

    async def _monitor_act(self, suggestion: str) -> None:
        if self._orchestrator is None or self._sys_monitor is None:
            return
        log = self.query_one("#log", _LogPane)

        # Capture baseline before acting
        pre_snap = await self._sys_monitor.snapshot()

        context = suggestion.split("—")[0].strip()
        await log.mount(Static(
            f"[monitor] {context} — acting autonomously…", classes="monitor-alert"
        ))
        log.scroll_end(animate=False)

        directive = self._to_directive(suggestion)
        history_len = len(self._orchestrator._history)
        self._monitor_acting = True
        self._orchestrator_busy = True
        try:
            result = await self._orchestrator.run(
                directive,
                on_step=self._on_step,
                on_step_start=self._on_step_start,
            )
            post_snap = await self._sys_monitor.snapshot()

            if result.error:
                await log.mount(Static(
                    f"[monitor] action failed: {result.error}", classes="error-msg"
                ))
            else:
                summary = result.final_response
                ram_delta = pre_snap.memory_percent - post_snap.memory_percent
                disk_delta = pre_snap.disk_percent - post_snap.disk_percent
                if ram_delta > 0.5:
                    summary += (
                        f" RAM: {pre_snap.memory_percent:.1f}%"
                        f" → {post_snap.memory_percent:.1f}%."
                    )
                elif disk_delta > 0.5:
                    summary += (
                        f" Disk: {pre_snap.disk_percent:.1f}%"
                        f" → {post_snap.disk_percent:.1f}%."
                    )
                await log.mount(Static(f"[monitor] {summary}", classes="monitor-alert"))
            log.scroll_end(animate=False)
        finally:
            # Don't let monitor actions pollute the user's conversation history
            del self._orchestrator._history[history_len:]
            self._monitor_acting = False
            self._orchestrator_busy = False

    async def _bg_poll(self) -> None:
        if self._sys_monitor is None or self._recommender is None:
            return
        if self._orchestrator_busy:
            return
        try:
            snap = await self._sys_monitor.snapshot()
            suggestion = self._recommender.check_snapshot(snap)
            if suggestion:
                if self._is_actionable(suggestion):
                    await self._monitor_act(suggestion)
                else:
                    log = self.query_one("#log", _LogPane)
                    await log.mount(Static(f"[monitor] {suggestion}", classes="monitor-alert"))
                    log.scroll_end(animate=False)
                return
            alerts = detect_anomalies(snap)
            if alerts:
                log = self.query_one("#log", _LogPane)
                for alert in alerts:
                    await log.mount(Static(f"[monitor] {alert}", classes="monitor-alert"))
                log.scroll_end(animate=False)
        except Exception:
            pass

    async def _run_scheduled(self, schedule: dict) -> None:
        assert self._workflows is not None
        assert self._schedules is not None
        log = self.query_one("#log", _LogPane)
        name = schedule["name"]
        workflow_name = schedule["workflow_name"]

        steps = await self._workflows.load(workflow_name)
        if steps is None:
            await log.mount(Static(
                f"[scheduler] '{name}': workflow '{workflow_name}' no longer exists — disabling.",
                classes="error-msg",
            ))
            log.scroll_end(animate=False)
            await self._schedules.set_enabled(name, False)
            return

        await log.mount(Static(
            f"[scheduler] running '{name}' ({workflow_name}, {len(steps)} steps)…",
            classes="monitor-alert",
        ))
        log.scroll_end(animate=False)

        history_len = len(self._orchestrator._history) if self._orchestrator else 0
        self._scheduler_acting = True
        self._orchestrator_busy = True
        had_error = False
        try:
            for i, step_data in enumerate(steps, start=1):
                tool_name = step_data["tool_name"]
                arguments = step_data["arguments"]
                try:
                    result = await self._tools.dispatch(tool_name, arguments)
                    sr = StepResult(step=i, tool_name=tool_name, arguments=arguments, result=result)
                except Exception as exc:
                    sr = StepResult(step=i, tool_name=tool_name, arguments=arguments, result=None, error=str(exc))
                    had_error = True
                await self._mount_step(sr)
            status = "completed with errors" if had_error else "completed"
            await log.mount(Static(f"[scheduler] '{name}' {status}.", classes="monitor-alert"))
            log.scroll_end(animate=False)
        finally:
            if self._orchestrator is not None:
                del self._orchestrator._history[history_len:]
            self._scheduler_acting = False
            self._orchestrator_busy = False
            await self._schedules.mark_run(name, schedule["interval_seconds"])

    async def _scheduler_tick(self) -> None:
        if self._schedules is None or self._workflows is None:
            return
        if self._orchestrator_busy:
            return
        try:
            for schedule in await self._schedules.due_schedules():
                await self._run_scheduled(schedule)
        except Exception:
            pass

    async def on_unmount(self) -> None:
        if self._monitor_timer is not None:
            self._monitor_timer.stop()
        if self._scheduler_timer is not None:
            self._scheduler_timer.stop()
        if self._session_id is not None:
            await self._memory.end_session(self._session_id)
        await self._ollama.close()
