"""Agent orchestrator — drives the plan → tool → result loop."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from laua.ollama_client import OllamaClient, OllamaUnavailableError
from laua.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_AGENT_STEPS = 10


def _coerce_args(args: dict) -> dict:
    """Parse JSON-string values that some models pass instead of native JSON types."""
    result = {}
    for k, v in args.items():
        if isinstance(v, str):
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                result[k] = v
        else:
            result[k] = v
    return result
_MAX_MODEL_FAILURES = 2
_TOOL_OUTPUT_MAX_CHARS = 2500


def _truncate_tool_output(tool_content: Any) -> Any:
    """Clip long stdout/output fields so the model doesn't misread walls of text."""
    if not isinstance(tool_content, dict):
        return tool_content
    result = dict(tool_content)
    for field in ("stdout", "output", "content"):
        if field in result and isinstance(result[field], str):
            val = result[field]
            if len(val) > _TOOL_OUTPUT_MAX_CHARS:
                omitted = len(val) - _TOOL_OUTPUT_MAX_CHARS
                result[field] = val[:_TOOL_OUTPUT_MAX_CHARS] + f"\n[... {omitted} chars truncated]"
    return result


_SYSTEM_PROMPT = """You are LAUA — a local system management agent for Ubuntu Linux. You are not a coding assistant, teacher, or documentation writer.

RESPONSE FORMAT (strictly enforced):
- 1 to 4 plain sentences. No more.
- No markdown of any kind: no bold (**text**), no italic, no headers, no code blocks.
- No bullet lists, no numbered lists, no label:value lines. Write as prose sentences only.
- Separate multiple data points with commas within a sentence. Do not use newlines as separators.
- No preamble ("Sure!", "Let me...", "Of course"). Get to the point.
- Never suggest commands for the user to run. Execute them yourself with tools.
- Never fabricate data. Report only what tools return.
- Copy exact numeric values from tool output — never round, approximate, or add qualifiers like "about" or "roughly". If the tool says 15.98 GB, say 15.98 GB. If it says 58.7%, say 58.7%.
- If tool output was truncated, say so. Do not fill in missing values from memory or assumption.

IDENTITY:
- Your name is LAUA — Local Autonomous Utility Agent.
- You were created by wzrdpluto.
- You run fully locally on wzrdpluto's Ubuntu workstation, powered by Ollama.
- When asked who made you, who built you, or who created you — answer directly: wzrdpluto.

CONVERSATIONAL INPUT (greetings, "how are you", "what can you do", small talk):
- Reply in plain text. Do NOT call any tool.

LINGO MAP — one tool call, no extra steps:
- "sys stats" / "check sys" / "stats" → get_system_info (omit include for all)
- "uptime" / "how long" / "boot time" → get_system_info(include=["uptime"])
- "ram" / "memory" / "mem" → get_system_info(include=["memory"])
- "cpu" / "processor" → get_system_info(include=["cpu"])
- "disk" / "storage" / "space" → get_system_info(include=["disk"])
- "procs" / "processes" / "what's running" → get_system_info(include=["processes"])
- "check temps" / "how hot" → run_command(["cat", "/sys/class/thermal/thermal_zone0/temp"])
- "net info" / "my ip" / "network" → run_command(["ip", "-brief", "addr", "show"])
- "check X" → use the most direct tool to inspect X

TOOL RULES:
- One tool call per step. Wait for results before calling again.
- Before any destructive action (delete, stop, kill), state exactly what you will do first.
- If a tool returns an error, report it and stop. Do not retry blindly.
- ALWAYS call the tool before reporting any outcome. Never predict or assume results — not even errors. If asked to read, execute, or check something, call the tool first, then report what it actually returned.
- Live system data (cpu, memory, disk, processes, temperatures, network) changes constantly. Never answer these from conversation history. Always call get_system_info or run_command for a fresh value.
"""


@dataclass
class StepResult:
    step: int
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    error: str | None = None


@dataclass
class OrchestratorResult:
    steps: list[StepResult] = field(default_factory=list)
    final_response: str = ""
    hit_step_ceiling: bool = False
    error: str | None = None
    model_used: str = ""


class Orchestrator:
    def __init__(
        self,
        ollama: OllamaClient,
        registry: ToolRegistry,
        model: str,
        history: list[dict[str, Any]] | None = None,
        context_manager: Any | None = None,
        model_router: Any | None = None,
    ) -> None:
        self._ollama = ollama
        self._registry = registry
        self._default_model = model
        self._history: list[dict[str, Any]] = history or []
        self._context_manager = context_manager
        self._router = model_router
        self._model_failures: dict[str, int] = {}

    def _pick_model(self, user_request: str) -> str:
        if self._router is None:
            return self._default_model
        task_type = self._router.classify(user_request)
        model = self._router.get_model(task_type)
        failures = self._model_failures.get(model, 0)
        if failures >= _MAX_MODEL_FAILURES:
            fallback = self._router.get_fallback_model()
            logger.warning(
                "Model %s has %d failures — falling back to %s", model, failures, fallback
            )
            return fallback
        return model

    async def run(
        self,
        user_request: str,
        on_step: Callable[[StepResult], None] | None = None,
        on_step_start: Callable[[str, int], None] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> OrchestratorResult:
        result = OrchestratorResult()
        active_model = self._pick_model(user_request)
        result.model_used = active_model

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": user_request},
        ]
        tools = self._registry.all_schemas()

        for step in range(1, MAX_AGENT_STEPS + 1):
            # Compress context if approaching token limit
            if self._context_manager and self._context_manager.should_compress(messages):
                messages = self._context_manager.compress(messages)

            try:
                response = await self._ollama.chat_with_tools_stream(
                    model=active_model,
                    messages=messages,
                    tools=tools,
                    on_token=on_token,
                )
            except OllamaUnavailableError as exc:
                result.error = str(exc)
                return result

            message = response.get("message", {})
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                content = message.get("content", "")
                if not content and step == 1:
                    # Empty first response — count as a model failure
                    self._model_failures[active_model] = (
                        self._model_failures.get(active_model, 0) + 1
                    )
                result.final_response = content
                break

            messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

            for call in tool_calls:
                fn = call.get("function", {})
                tool_name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                arguments = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
                arguments = _coerce_args(arguments)

                if on_step_start is not None:
                    on_step_start(tool_name, step)

                try:
                    tool_result = await self._registry.dispatch(tool_name, arguments)
                    step_result = StepResult(
                        step=step, tool_name=tool_name, arguments=arguments, result=tool_result
                    )
                except Exception as exc:
                    step_result = StepResult(
                        step=step, tool_name=tool_name,
                        arguments=arguments, result=None, error=str(exc),
                    )
                    logger.warning("Tool %s failed: %s", tool_name, exc)

                result.steps.append(step_result)
                if on_step is not None:
                    on_step(step_result)
                err = step_result.error
                tool_content = step_result.result if err is None else {"error": err}
                messages.append({"role": "tool", "content": json.dumps(_truncate_tool_output(tool_content))})
        else:
            result.hit_step_ceiling = True
            result.final_response = (
                f"Reached the {MAX_AGENT_STEPS}-step limit without completing the task. "
                "Please provide further guidance."
            )

        self._history.extend([
            {"role": "user", "content": user_request},
            {"role": "assistant", "content": result.final_response},
        ])
        return result
