"""Text-to-speech via Piper's CLI — subprocess only, no piper-tts Python API dependency."""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from laua.voice.audio import play_file


def build_piper_command(piper_binary: str, model_path: Path, output_path: Path) -> list[str]:
    return [piper_binary, "--model", str(model_path), "--output_file", str(output_path)]


class TextToSpeech:
    def __init__(self, model_path: str, piper_binary: str = "piper") -> None:
        self._model_path = Path(model_path).expanduser()
        self._piper_binary = piper_binary

    def synthesize_and_play(self, text: str) -> None:
        """Blocking — call via asyncio.to_thread."""
        out_path = Path("/tmp") / f"laua-tts-{uuid.uuid4().hex}.wav"
        try:
            subprocess.run(
                build_piper_command(self._piper_binary, self._model_path, out_path),
                input=text.encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            play_file(out_path)
        finally:
            out_path.unlink(missing_ok=True)
