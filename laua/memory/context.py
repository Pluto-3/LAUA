"""Context window manager — token estimation and message compression."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Tracks approximate token usage and compresses the message list when
    it approaches the model's context limit.
    """

    def __init__(
        self,
        model_max_tokens: int = 4096,
        trigger_ratio: float = 0.80,
    ) -> None:
        self.model_max_tokens = model_max_tokens
        self.trigger_ratio = trigger_ratio

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Rough token estimate: sum of character lengths divided by 4."""
        return sum(len(str(m)) // 4 for m in messages)

    def should_compress(self, messages: list[dict[str, Any]]) -> bool:
        """Return True when estimated token usage exceeds the trigger threshold."""
        return self.estimate_tokens(messages) > self.model_max_tokens * self.trigger_ratio

    def compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Drop the oldest non-system messages until we are under the trigger threshold.
        Always keeps the system message (if present) and the most recent messages.
        """
        threshold = int(self.model_max_tokens * self.trigger_ratio)

        # Separate system messages from the rest
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # Drop from the front of the non-system list until under threshold
        while non_system and self.estimate_tokens(system_msgs + non_system) > threshold:
            dropped = non_system.pop(0)
            logger.debug(
                "Context compression: dropped %s message (role=%s)",
                "tool_calls" if dropped.get("tool_calls") else "text",
                dropped.get("role"),
            )

        compressed = system_msgs + non_system
        logger.info(
            "Context compressed: %d → %d estimated tokens",
            self.estimate_tokens(messages),
            self.estimate_tokens(compressed),
        )
        return compressed
