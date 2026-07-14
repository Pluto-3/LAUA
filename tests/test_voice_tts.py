"""Tests for voice TTS command builders — pure, no Piper installation needed."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from laua.voice.tts import build_piper_command, resolve_piper_binary


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


# ── resolve_piper_binary — regression coverage for the venv-shebang PATH bug ──
# `laua` runs via a venv-python shebang, which does not put the venv's bin/ on
# PATH, so a bare "piper" silently failed with FileNotFoundError until this
# fallback was added (caught, but swallowed, by _speak_response's bare except).

def test_resolve_piper_binary_respects_explicit_path():
    """A user-configured non-default binary/path is never second-guessed."""
    assert resolve_piper_binary("/opt/custom/piper") == "/opt/custom/piper"


def test_resolve_piper_binary_uses_system_path_if_present():
    with patch("laua.voice.tts.shutil.which", return_value="/usr/bin/piper"):
        assert resolve_piper_binary("piper") == "piper"


def test_resolve_piper_binary_falls_back_to_venv_sibling():
    with patch("laua.voice.tts.shutil.which", return_value=None), \
         patch("laua.voice.tts.sys.executable", "/some/venv/bin/python3"), \
         patch("laua.voice.tts.Path.exists", return_value=True):
        assert resolve_piper_binary("piper") == "/some/venv/bin/piper"


def test_resolve_piper_binary_gives_up_if_nothing_found():
    with patch("laua.voice.tts.shutil.which", return_value=None), \
         patch("laua.voice.tts.Path.exists", return_value=False):
        assert resolve_piper_binary("piper") == "piper"
