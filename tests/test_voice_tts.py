"""Tests for voice TTS command builders — pure, no Piper installation needed."""

from __future__ import annotations

from pathlib import Path

from laua.voice.tts import build_piper_command


def test_build_piper_command():
    result = build_piper_command("piper", Path("/models/voice.onnx"), Path("/tmp/out.wav"))
    assert result == ["piper", "--model", "/models/voice.onnx", "--output_file", "/tmp/out.wav"]


def test_build_piper_command_custom_binary():
    result = build_piper_command(
        "/usr/local/bin/piper", Path("/models/v.onnx"), Path("/tmp/o.wav")
    )
    assert result == [
        "/usr/local/bin/piper", "--model", "/models/v.onnx", "--output_file", "/tmp/o.wav",
    ]
