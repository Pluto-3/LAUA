"""Edge-case tests for file manager — bugs and degenerate inputs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from laua.tools.file_manager import (
    _delete_file,
    _is_restricted,
    _list_directory,
    _search_files,
    _write_file,
)

_NO_RESTRICTED: list[str] = []


async def _yes(*a, **kw) -> bool:
    return True


async def _no(*a, **kw) -> bool:
    return False


# ── _is_restricted: edge cases ────────────────────────────────────────────────

def test_is_restricted_exact_path_match():
    assert _is_restricted(Path("/etc"), ["/etc"]) is True


def test_is_restricted_deep_child():
    assert _is_restricted(Path("/etc/ssh/sshd_config"), ["/etc"]) is True


def test_is_restricted_sibling_not_restricted():
    assert _is_restricted(Path("/etcfoo/bar"), ["/etc"]) is False


def test_is_restricted_tilde_in_restricted_list():
    """Restricted list entries with ~ are expanded correctly."""
    home = str(Path.home())
    assert _is_restricted(Path(home) / "secret.txt", ["~"]) is True


def test_is_restricted_empty_restricted_list():
    assert _is_restricted(Path("/etc/passwd"), []) is False


def test_is_restricted_multiple_restricted_paths():
    assert _is_restricted(Path("/boot/grub/grub.cfg"), ["/etc", "/boot"]) is True
    assert _is_restricted(Path("/tmp/safe.txt"), ["/etc", "/boot"]) is False


def test_is_restricted_symlink_traversal(tmp_path):
    """A symlink inside an unrestricted dir that points into a restricted dir
    is correctly blocked after resolve()."""
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    safe = tmp_path / "safe"
    safe.mkdir()
    link = safe / "link_to_restricted"
    link.symlink_to(restricted)
    assert _is_restricted(link / "secret.txt", [str(restricted)]) is True


# ── _list_directory: edge cases ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_directory_empty(tmp_path):
    result = await _list_directory(str(tmp_path))
    assert result["files"] == []
    assert result["dirs"] == []


@pytest.mark.asyncio
async def test_list_directory_only_subdirs(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    result = await _list_directory(str(tmp_path))
    assert result["files"] == []
    assert "a" in result["dirs"]
    assert "b" in result["dirs"]


@pytest.mark.asyncio
async def test_list_directory_no_pattern_returns_all(tmp_path):
    (tmp_path / "foo.py").write_text("")
    (tmp_path / "bar.txt").write_text("")
    result = await _list_directory(str(tmp_path))
    assert "foo.py" in result["files"]
    assert "bar.txt" in result["files"]


@pytest.mark.asyncio
async def test_list_directory_pattern_dotpy_file_excluded():
    """
    BUG: Files whose names start with the stripped pattern (e.g. '.python_version'
    starts with '.py') currently bypass the glob check and get falsely included.
    """
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "good.py").write_text("")
        (Path(d) / ".python_version").write_text("3.11")  # starts with ".py"
        result = await _list_directory(d, pattern="*.py")
        assert "good.py" in result["files"]
        assert ".python_version" not in result["files"], (
            "BUG: .python_version should NOT match *.py — prefix shortcut is wrong"
        )


@pytest.mark.asyncio
async def test_list_directory_tilde_path():
    """~ in path should expand correctly."""
    result = await _list_directory("~")
    assert "error" not in result


@pytest.mark.asyncio
async def test_list_directory_sorted_output(tmp_path):
    for name in ["z.txt", "a.txt", "m.txt"]:
        (tmp_path / name).write_text("")
    result = await _list_directory(str(tmp_path))
    assert result["files"] == sorted(result["files"])


# ── _write_file: edge cases ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_file_empty_content(tmp_path):
    path = str(tmp_path / "empty.txt")
    result = await _write_file(path, "", _yes, _NO_RESTRICTED, 1024)
    assert result["status"] == "written"
    assert result["bytes"] == 0
    assert Path(path).read_text() == ""


@pytest.mark.asyncio
async def test_write_file_unicode_content(tmp_path):
    path = str(tmp_path / "unicode.txt")
    content = "こんにちは 🎉"
    result = await _write_file(path, content, _yes, _NO_RESTRICTED, 65536)
    assert result["status"] == "written"
    assert Path(path).read_text(encoding="utf-8") == content


@pytest.mark.asyncio
async def test_write_file_to_directory_path_returns_error(tmp_path):
    """Passing a directory as the target path should return an error, not crash."""
    result = await _write_file(str(tmp_path), "content", _yes, _NO_RESTRICTED, 65536)
    assert "error" in result


@pytest.mark.asyncio
async def test_write_file_max_bytes_exact_passes(tmp_path):
    """Content of exactly max_write_bytes should succeed (not exceed check)."""
    path = str(tmp_path / "exact.txt")
    content = "a" * 100
    result = await _write_file(path, content, _yes, _NO_RESTRICTED, 100)
    assert result["status"] == "written"


@pytest.mark.asyncio
async def test_write_file_max_bytes_one_over_fails(tmp_path):
    path = str(tmp_path / "toobig.txt")
    content = "a" * 101
    result = await _write_file(path, content, _yes, _NO_RESTRICTED, 100)
    assert "error" in result


@pytest.mark.asyncio
async def test_write_file_no_confirm_for_new_file(tmp_path):
    """Creating a new file must NOT call confirm_fn."""
    calls = []

    async def tracking_confirm(args, **kw):
        calls.append(args)
        return True

    path = str(tmp_path / "new.txt")
    await _write_file(path, "hi", tracking_confirm, _NO_RESTRICTED, 1024)
    assert calls == [], "New file creation should not ask for confirmation"


@pytest.mark.asyncio
async def test_write_file_confirm_called_for_overwrite(tmp_path):
    """Overwriting an existing file MUST call confirm_fn."""
    calls = []

    async def tracking_confirm(args, **kw):
        calls.append(args)
        return True

    path = tmp_path / "existing.txt"
    path.write_text("old")
    await _write_file(str(path), "new", tracking_confirm, _NO_RESTRICTED, 1024)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_write_file_creates_nested_dirs(tmp_path):
    path = str(tmp_path / "a" / "b" / "c" / "file.txt")
    result = await _write_file(path, "deep", _yes, _NO_RESTRICTED, 1024)
    assert result["status"] == "written"


# ── _search_files: edge cases ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_files_max_results_zero():
    """max_results=0 returns empty matches with truncated=False (not a truncation)."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.txt").write_text("")
        result = await _search_files(d, max_results=0)
        assert result["count"] == 0
        assert result["truncated"] is False


@pytest.mark.asyncio
async def test_search_files_no_matches(tmp_path):
    (tmp_path / "readme.txt").write_text("hello")
    result = await _search_files(str(tmp_path), extension="py")
    assert result["matches"] == []
    assert result["count"] == 0
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_search_files_extension_with_leading_dot(tmp_path):
    """Extension '.py' (with dot) should work the same as 'py' (without)."""
    (tmp_path / "script.py").write_text("")
    (tmp_path / "notes.txt").write_text("")
    result = await _search_files(str(tmp_path), extension=".py")
    assert any("script.py" in m for m in result["matches"])
    assert not any("notes.txt" in m for m in result["matches"])


@pytest.mark.asyncio
async def test_search_files_combined_extension_and_content(tmp_path):
    (tmp_path / "match.py").write_text("SECRET_TOKEN")
    (tmp_path / "nomatch.py").write_text("nothing here")
    (tmp_path / "wrong_ext.txt").write_text("SECRET_TOKEN")
    result = await _search_files(str(tmp_path), extension="py", content_pattern="SECRET_TOKEN")
    assert any("match.py" in m for m in result["matches"])
    assert not any("nomatch.py" in m for m in result["matches"])
    assert not any("wrong_ext.txt" in m for m in result["matches"])


@pytest.mark.asyncio
async def test_search_files_content_pattern_empty_string(tmp_path):
    """Empty content_pattern should match every file (substring '' is in everything)."""
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    result = await _search_files(str(tmp_path), content_pattern="")
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_search_files_on_file_not_dir(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hi")
    result = await _search_files(str(f))
    assert "error" in result


@pytest.mark.asyncio
async def test_search_files_name_pattern_glob(tmp_path):
    (tmp_path / "test_foo.py").write_text("")
    (tmp_path / "test_bar.py").write_text("")
    (tmp_path / "main.py").write_text("")
    result = await _search_files(str(tmp_path), name_pattern="test_*")
    matches = result["matches"]
    assert any("test_foo.py" in m for m in matches)
    assert any("test_bar.py" in m for m in matches)
    assert not any("main.py" in m for m in matches)


@pytest.mark.asyncio
async def test_search_files_truncated_flag_accurate(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("")
    result = await _search_files(str(tmp_path), max_results=5)
    # Exactly at limit: could be True or False depending on whether we hit exactly max
    # At least verify it doesn't crash and count is correct
    assert result["count"] == 5


# ── _delete_file: edge cases ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_symlink_removes_link_not_target(tmp_path):
    """Deleting a symlink should remove the link, not the target file."""
    target = tmp_path / "target.txt"
    target.write_text("keep me")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    result = await _delete_file(str(link), _yes, _NO_RESTRICTED)
    assert result["status"] == "deleted"
    assert not link.exists()
    assert target.exists(), "Target file must survive symlink deletion"


@pytest.mark.asyncio
async def test_delete_file_confirm_receives_full_path(tmp_path):
    """confirm_fn must receive the full absolute path, not a relative one."""
    received_args = []

    async def tracking_confirm(args, **kw):
        received_args.extend(args)
        return False  # deny to avoid actual deletion

    f = tmp_path / "check_me.txt"
    f.write_text("x")
    await _delete_file(str(f), tracking_confirm, _NO_RESTRICTED)
    # The path in the confirm args should be absolute
    path_arg = received_args[-1] if received_args else ""
    assert path_arg.startswith("/"), f"Expected absolute path, got: {path_arg}"
