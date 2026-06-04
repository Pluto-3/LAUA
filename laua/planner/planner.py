"""Multi-step planner — generates an execution plan before tool dispatch."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_PLANNING_PROMPT = """\
You are a planning assistant for a Linux system management agent.
Given a user request, output a JSON array of steps the agent should take.
Each step: {"step": N, "description": "what this step does"}
Output ONLY the JSON array. No markdown, no explanation, no code blocks.
Example: [{"step": 1, "description": "Check disk usage"}, {"step": 2, "description": "Find files over 500MB"}]
"""

_COMPLEX_SIGNALS = [
    " and ", " then ", "after that", " next ", "finally ",
    "set up", "configure", "clean up", "organize", "backup",
    "find and", "check and", "make sure", "also ", "first ",
]

# Single high-confidence signals that alone indicate multi-step work
_STRONG_SIGNALS = {"set up", "configure", "clean up", "organize", "backup"}


@dataclass
class PlanStep:
    step: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "description": self.description}


class Planner:
    def __init__(self, ollama: Any, model: str) -> None:
        self._ollama = ollama
        self._model = model

    def is_complex(self, request: str) -> bool:
        lower = request.lower()
        if any(s in lower for s in _STRONG_SIGNALS):
            return True
        return sum(1 for s in _COMPLEX_SIGNALS if s in lower) >= 2

    async def plan(self, request: str) -> list[PlanStep] | None:
        messages = [
            {"role": "system", "content": _PLANNING_PROMPT},
            {"role": "user", "content": request},
        ]
        try:
            response = await self._ollama.chat_with_tools(
                model=self._model,
                messages=messages,
                tools=[],
            )
            content = response.get("message", {}).get("content", "").strip()
            if content.startswith("```"):
                lines = content.splitlines()
                end = len(lines) - 1 if lines[-1].startswith("```") else len(lines)
                content = "\n".join(lines[1:end]).strip()
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                return None
            steps = [
                PlanStep(step=s.get("step", i + 1), description=s.get("description", ""))
                for i, s in enumerate(parsed)
                if isinstance(s, dict) and s.get("description")
            ]
            return steps or None
        except Exception as exc:
            logger.debug("Planner could not produce a plan: %s", exc)
            return None
