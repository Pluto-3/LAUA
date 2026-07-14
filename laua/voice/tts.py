"""Text-to-speech via Piper's CLI — subprocess only, no piper-tts Python API dependency."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from laua.voice.audio import play_file

# Words Piper's grapheme-to-phoneme model can't read correctly as written —
# spelled-out equivalents used for speech only. The on-screen text is never
# touched; this only affects what gets sent to the synthesizer.
_PRONUNCIATION_OVERRIDES: dict[str, str] = {
    "wzrdpluto": "wizard pluto",
}


def apply_pronunciation_overrides(text: str) -> str:
    for written, spoken in _PRONUNCIATION_OVERRIDES.items():
        text = re.sub(rf"\b{re.escape(written)}\b", spoken, text, flags=re.IGNORECASE)
    return text


def build_piper_command(piper_binary: str, model_path: Path, output_path: Path) -> list[str]:
    return [piper_binary, "--model", str(model_path), "--output_file", str(output_path)]


def resolve_piper_binary(configured: str) -> str:
    """`laua` runs via a venv-python shebang, which does NOT put the venv's bin/ on
    PATH — so a bare "piper" only resolves if the venv happens to be activated.
    Fall back to the console script installed next to the running interpreter
    (pip puts piper's entry point in the same bin/ as python) before giving up.
    """
    if configured != "piper":
        return configured  # explicit path/binary — respect it as-is
    if shutil.which("piper"):
        return "piper"
    sibling = Path(sys.executable).parent / "piper"
    if sibling.exists():
        return str(sibling)
    return "piper"  # nothing found — let subprocess raise FileNotFoundError as before


class TextToSpeech:
    def __init__(self, model_path: str, piper_binary: str = "piper") -> None:
        self._model_path = Path(model_path).expanduser()
        self._piper_binary = resolve_piper_binary(piper_binary)

    def synthesize_and_play(self, text: str) -> None:
        """Blocking — call via asyncio.to_thread."""
        out_path = Path("/tmp") / f"laua-tts-{uuid.uuid4().hex}.wav"
        spoken_text = apply_pronunciation_overrides(text)
        try:
            subprocess.run(
                build_piper_command(self._piper_binary, self._model_path, out_path),
                input=spoken_text.encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            play_file(out_path)
        finally:
            out_path.unlink(missing_ok=True)
