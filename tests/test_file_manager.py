"""Tests for file manager tools."""

import os
import tempfile
from pathlib import Path
import pytest
from laua.tools.file_manager import (
    _is_restricted, _list_directory, _write_file,
    _search_files, _delete_file,
)

_NO_RESTRICTED = []
_CONFIRM_YES = lambda args, **kw: _async_true()
_CONFIRM_NO  = lambda args, **kw: _async_false()

import asyncio
async def _async_true(*a, **kw): return True
async def _async_false(*a, **kw): return False


# ── _is_restricted ────────────────────────────────────────────────────────────

def test_is_restricted_inside():
    assert _is_restricted(Path("/etc/passwd"), ["/etc"]) is True

def test_is_restricted_outside():
    assert _is_restricted(Path("/home/user/file.txt"), ["/etc"]) is False

def test_is_restricted_empty_list():
    assert _is_restricted(Path("/etc/passwd"), []) is False

def test_is_restricted_exact_match():
    assert _is_restricted(Path("/etc"), ["/etc"]) is True


# ── _list_directory ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_directory_basic(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "subdir").mkdir()
    result = await _list_directory(str(tmp_path))
    assert "a.txt" in result["files"]
    assert "subdir" in result["dirs"]

@pytest.mark.asyncio
async def test_list_directory_nonexistent():
    result = await _list_directory("/nonexistent_xyz_abc")
    assert "error" in result

@pytest.mark.asyncio
async def test_list_directory_on_file(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hi")
    result = await _list_directory(str(f))
    assert "error" in result

@pytest.mark.asyncio
async def test_list_directory_with_pattern(tmp_path):
    (tmp_path / "foo.py").write_text("")
    (tmp_path / "bar.txt").write_text("")
    result = await _list_directory(str(tmp_path), pattern="*.py")
    assert "foo.py" in result["files"]
    assert "bar.txt" not in result["files"]


# ── _write_file ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_file_creates_new(tmp_path):
    path = str(tmp_path / "new.txt")
    result = await _write_file(path, "hello", _CONFIRM_YES, _NO_RESTRICTED, 1024)
    assert result["status"] == "written"
    assert Path(path).read_text() == "hello"

@pytest.mark.asyncio
async def test_write_file_overwrite_requires_confirm_yes(tmp_path):
    path = tmp_path / "existing.txt"
    path.write_text("old")
    result = await _write_file(str(path), "new", _CONFIRM_YES, _NO_RESTRICTED, 1024)
    assert result["status"] == "written"
    assert path.read_text() == "new"

@pytest.mark.asyncio
async def test_write_file_overwrite_confirm_no(tmp_path):
    path = tmp_path / "existing.txt"
    path.write_text("original")
    result = await _write_file(str(path), "new", _CONFIRM_NO, _NO_RESTRICTED, 1024)
    assert result["status"] == "blocked"
    assert path.read_text() == "original"

@pytest.mark.asyncio
async def test_write_file_restricted_blocked(tmp_path):
    result = await _write_file("/etc/test_laua.txt", "x", _CONFIRM_YES, ["/etc"], 1024)
    assert "error" in result

@pytest.mark.asyncio
async def test_write_file_exceeds_max_bytes(tmp_path):
    path = str(tmp_path / "big.txt")
    result = await _write_file(path, "x" * 200, _CONFIRM_YES, _NO_RESTRICTED, 10)
    assert "error" in result

@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(tmp_path):
    path = str(tmp_path / "deep" / "nested" / "file.txt")
    result = await _write_file(path, "hi", _CONFIRM_YES, _NO_RESTRICTED, 1024)
    assert result["status"] == "written"


# ── _search_files ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_files_by_extension(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    result = await _search_files(str(tmp_path), extension="py")
    assert any("a.py" in m for m in result["matches"])
    assert not any("b.txt" in m for m in result["matches"])

@pytest.mark.asyncio
async def test_search_files_by_content(tmp_path):
    (tmp_path / "match.txt").write_text("find_me_here")
    (tmp_path / "no_match.txt").write_text("nothing here")
    result = await _search_files(str(tmp_path), content_pattern="find_me_here")
    assert any("match.txt" in m for m in result["matches"])
    assert not any("no_match.txt" in m for m in result["matches"])

@pytest.mark.asyncio
async def test_search_files_nonexistent_dir():
    result = await _search_files("/nonexistent_dir_xyz")
    assert "error" in result

@pytest.mark.asyncio
async def test_search_files_max_results(tmp_path):
    for i in range(10):
        (tmp_path / f"file{i}.txt").write_text("")
    result = await _search_files(str(tmp_path), max_results=3)
    assert result["count"] == 3
    assert result["truncated"] is True

@pytest.mark.asyncio
async def test_search_files_no_criteria_returns_all(tmp_path):
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text("")
    result = await _search_files(str(tmp_path))
    assert result["count"] == 3


# ── _delete_file ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_file_confirm_yes(tmp_path):
    f = tmp_path / "to_delete.txt"
    f.write_text("bye")
    result = await _delete_file(str(f), _CONFIRM_YES, _NO_RESTRICTED)
    assert result["status"] == "deleted"
    assert not f.exists()

@pytest.mark.asyncio
async def test_delete_file_confirm_no(tmp_path):
    f = tmp_path / "keep.txt"
    f.write_text("keep me")
    result = await _delete_file(str(f), _CONFIRM_NO, _NO_RESTRICTED)
    assert result["status"] == "blocked"
    assert f.exists()

@pytest.mark.asyncio
async def test_delete_file_nonexistent():
    result = await _delete_file("/nonexistent/xyz.txt", _CONFIRM_YES, _NO_RESTRICTED)
    assert "error" in result

@pytest.mark.asyncio
async def test_delete_file_restricted():
    result = await _delete_file("/etc/passwd", _CONFIRM_YES, ["/etc"])
    assert "error" in result

@pytest.mark.asyncio
async def test_delete_directory_rejected(tmp_path):
    result = await _delete_file(str(tmp_path), _CONFIRM_YES, _NO_RESTRICTED)
    assert "error" in result
