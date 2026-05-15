"""Model router — config-driven task classification and model selection."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

TASK_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code", "script", "function", "debug", "implement", "class",
        "bug", "error", "syntax", "compile",
    ],
    "reasoning": [
        "why", "explain", "analyze", "reason", "think", "compare",
        "should i", "what if", "pros", "cons",
    ],
}


class ModelRouter:
    """
    Classifies user messages into task types and returns the appropriate model.
    Falls back to the 'fast' model when no keywords match.
    """

    def __init__(self, routing_config: dict) -> None:
        self._config = routing_config

    def classify(self, user_message: str) -> str:
        """Return 'coding', 'reasoning', or 'fast' based on keyword matching."""
        lower = user_message.lower()
        for task_type, keywords in TASK_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return task_type
        return "fast"

    def get_model(self, task_type: str) -> str:
        """Return the configured model name for the given task type."""
        model = self._config.get(task_type) or self._config.get("fast", "qwen2.5:7b")
        logger.debug("ModelRouter: task_type=%s → model=%s", task_type, model)
        return model

    def get_fallback_model(self) -> str:
        """Return the configured fallback model (fastest available)."""
        return self._config.get("fallback", self._config.get("fast", "qwen2.5:7b"))
