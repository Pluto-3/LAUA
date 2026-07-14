"""Audio capture/playback via subprocess — arecord/paplay, no Python audio bindings."""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path


def build_arecord_command(
    output_path: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    audio_format: str = "S16_LE",
) -> list[str]:
    return [
        "arecord",
        "-f", audio_format,
        "-r", str(sample_rate),
        "-c", str(channels),
        str(output_path),
    ]


def build_paplay_command(input_path: Path) -> list[str]:
    return ["paplay", str(input_path)]


def play_file(path: Path) -> None:
    """Blocking. Call via asyncio.to_thread."""
    subprocess.run(
        build_paplay_command(path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class AudioRecorder:
    def __init__(
        self,
        tmp_dir: Path | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self._tmp_dir = tmp_dir or Path("/tmp")
        self._sample_rate = sample_rate
        self._channels = channels
        self._proc: subprocess.Popen | None = None
        self._current_path: Path | None = None

    @property
    def is_recording(self) -> bool:
        return self._proc is not None

    def start(self) -> Path:
        """Spawn arecord into a fresh temp wav. Non-blocking — safe to call from async code."""
        path = self._tmp_dir / f"laua-voice-{uuid.uuid4().hex}.wav"
        args = build_arecord_command(
            path, sample_rate=self._sample_rate, channels=self._channels
        )
        self._proc = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._current_path = path
        return path

    def stop(self) -> Path | None:
        """Blocking — call via asyncio.to_thread. Finalizes the WAV header on terminate."""
        if self._proc is None:
            return None
        proc, path = self._proc, self._current_path
        self._proc = None
        self._current_path = None
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        return path
