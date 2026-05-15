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
_MAX_MODEL_FAILURES = 2

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

    async def run(self, user_request: str) -> OrchestratorResult:
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
                response = await self._ollama.chat_with_tools(
                    model=active_model,
                    messages=messages,
                    tools=tools,
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
