"""Agent orchestrator — drives the plan → tool → result loop."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from laua.ollama_client import OllamaClient, OllamaUnavailableError
from laua.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_AGENT_STEPS = 10

_SYSTEM_PROMPT = """You are LAUA, a Local Autonomous Utility Agent running on Ubuntu Linux.
You help the user manage their system through natural language.

Rules you must follow:
- Always use the provided tools to act. Never output raw shell commands as text.
- Output valid JSON tool calls only — never free-form shell strings.
- If you need to run multiple commands, do them one at a time using run_command.
- If a task is ambiguous, ask for clarification before acting.
- Before destructive actions, explain what you're about to do.
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


class Orchestrator:
    def __init__(
        self,
        ollama: OllamaClient,
        registry: ToolRegistry,
        model: str,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self._ollama = ollama
        self._registry = registry
        self._model = model
        self._history: list[dict[str, Any]] = history or []

    async def run(self, user_request: str) -> OrchestratorResult:
        result = OrchestratorResult()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": user_request},
        ]
        tools = self._registry.all_schemas()

        for step in range(1, MAX_AGENT_STEPS + 1):
            try:
                response = await self._ollama.chat_with_tools(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                )
            except OllamaUnavailableError as exc:
                result.error = str(exc)
                return result

            message = response.get("message", {})
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                # No more tool calls — final text response
                result.final_response = message.get("content", "")
                break

            messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

            for call in tool_calls:
                fn = call.get("function", {})
                tool_name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                arguments = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)

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
                err = step_result.error
                tool_content = step_result.result if err is None else {"error": err}
                messages.append({"role": "tool", "content": json.dumps(tool_content)})
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
