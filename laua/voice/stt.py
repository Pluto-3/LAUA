"""Speech-to-text via faster-whisper, CPU-only (keeps VRAM free for Ollama)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SpeechToText:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            # Deferred import — laua.voice.stt (and therefore app.py, which imports
            # this module at load time) must import cleanly even when faster-whisper
            # isn't installed, as long as voice.enabled stays false.
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def transcribe(self, path: Path) -> str:
        """Blocking — call via asyncio.to_thread. CPU-bound, can take several seconds."""
        segments, _info = self._ensure_model().transcribe(str(path), language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()
