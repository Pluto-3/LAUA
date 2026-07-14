"""Tests for voice audio command builders — pure, no subprocess mocking needed."""

from __future__ import annotations

from pathlib import Path

from laua.voice.audio import build_arecord_command, build_paplay_command


def test_build_arecord_command_defaults():
    assert build_arecord_command(Path("/tmp/out.wav")) == [
        "arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "/tmp/out.wav",
    ]


def test_build_arecord_command_custom_rate_channels():
    result = build_arecord_command(Path("/tmp/out.wav"), sample_rate=44100, channels=2)
    assert result == ["arecord", "-f", "S16_LE", "-r", "44100", "-c", "2", "/tmp/out.wav"]


def test_build_arecord_command_custom_format():
    result = build_arecord_command(Path("/tmp/out.wav"), audio_format="S32_LE")
    assert result == ["arecord", "-f", "S32_LE", "-r", "16000", "-c", "1", "/tmp/out.wav"]


def test_build_paplay_command():
    assert build_paplay_command(Path("/tmp/in.wav")) == ["paplay", "/tmp/in.wav"]
