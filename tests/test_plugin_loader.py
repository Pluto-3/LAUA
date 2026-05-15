"""Tests for the plugin loader."""

import pytest
from pathlib import Path
from laua.tools.registry import ToolRegistry
from laua.tools.plugin_loader import load_plugins


def _make_valid_plugin(tmp_path, name="my_plugin"):
    content = """
def register(registry, session, confirm_fn, audit_fn):
    registry._loaded_by_plugin = True
"""
    path = tmp_path / f"{name}.py"
    path.write_text(content)
    return path


def test_valid_plugin_loaded(tmp_path):
    _make_valid_plugin(tmp_path)
    reg = ToolRegistry()
    count = load_plugins(reg, None, None, None, extra_paths=[tmp_path])
    assert count == 1
    assert getattr(reg, "_loaded_by_plugin", False) is True


def test_plugin_without_register_skipped(tmp_path):
    path = tmp_path / "no_register.py"
    path.write_text("TOOL_NAME = 'something'\n")
    reg = ToolRegistry()
    count = load_plugins(reg, None, None, None, extra_paths=[tmp_path])
    assert count == 0


def test_plugin_with_syntax_error_skipped(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("def register(\n  # broken\n")
    reg = ToolRegistry()
    count = load_plugins(reg, None, None, None, extra_paths=[tmp_path])
    assert count == 0


def test_plugin_register_raises_skipped(tmp_path):
    path = tmp_path / "raises.py"
    path.write_text("def register(registry, session, confirm_fn, audit_fn):\n    raise RuntimeError('oops')\n")
    reg = ToolRegistry()
    count = load_plugins(reg, None, None, None, extra_paths=[tmp_path])
    assert count == 0


def test_nonexistent_extra_path_ignored():
    reg = ToolRegistry()
    count = load_plugins(reg, None, None, None, extra_paths=[Path("/nonexistent_plugin_dir_xyz")])
    assert count == 0


def test_multiple_valid_plugins(tmp_path):
    for i in range(3):
        _make_valid_plugin(tmp_path, name=f"plugin_{i}")
    reg = ToolRegistry()
    count = load_plugins(reg, None, None, None, extra_paths=[tmp_path])
    assert count == 3


def test_empty_directory_loads_zero(tmp_path):
    reg = ToolRegistry()
    count = load_plugins(reg, None, None, None, extra_paths=[tmp_path])
    assert count == 0
