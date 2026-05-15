"""File manager tool — scoped write/search/delete with restricted path enforcement."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

from laua.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


def _is_restricted(path: Path, restricted_paths: list[str]) -> bool:
    """Return True if the resolved path is inside any restricted directory."""
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path.absolute()
    for restricted in restricted_paths:
        try:
            restricted_resolved = Path(restricted).expanduser().resolve()
            resolved.relative_to(restricted_resolved)
            return True
        except ValueError:
            continue
    return False


async def _list_directory(
    path: str,
    pattern: str | None = None,
) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        return {"error": f"Path does not exist: {p}"}
    if not p.is_dir():
        return {"error": f"Not a directory: {p}"}
    try:
        entries = list(p.iterdir())
    except PermissionError:
        return {"error": f"Permission denied: {p}"}

    files: list[str] = []
    dirs: list[str] = []
    for entry in entries:
        if pattern and not entry.name.startswith(pattern.lstrip("*").rstrip("*")):
            # Simple pattern: check glob match
            if not entry.match(pattern):
                continue
        if entry.is_dir():
            dirs.append(entry.name)
        else:
            files.append(entry.name)

    return {"path": str(p), "files": sorted(files), "dirs": sorted(dirs)}


async def _write_file(
    path: str,
    content: str,
    confirm_fn: Callable,
    restricted_paths: list[str],
    max_write_bytes: int,
) -> dict[str, Any]:
    p = Path(path).expanduser()

    if _is_restricted(p, restricted_paths):
        return {"error": f"Path is restricted and cannot be written: {p}"}

    if len(content.encode()) > max_write_bytes:
        return {
            "error": (
                f"Content exceeds max write size of {max_write_bytes} bytes "
                f"({len(content.encode())} bytes given)."
            )
        }

    if p.exists() and p.is_file():
        approved = await confirm_fn(["overwrite", str(p)])
        if not approved:
            return {"status": "blocked", "reason": "User denied file overwrite."}

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"status": "written", "path": str(p), "bytes": len(content.encode())}
    except PermissionError:
        return {"error": f"Permission denied: {p}"}
    except Exception as exc:
        return {"error": str(exc)}


async def _search_files(
    directory: str,
    name_pattern: str | None = None,
    extension: str | None = None,
    content_pattern: str | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    base = Path(directory).expanduser()
    if not base.exists():
        return {"error": f"Directory does not exist: {base}"}
    if not base.is_dir():
        return {"error": f"Not a directory: {base}"}

    matches: list[str] = []
    ext_filter = extension.lstrip(".").lower() if extension else None

    for dirpath, _dirnames, filenames in os.walk(base):
        for fname in filenames:
            if len(matches) >= max_results:
                break

            # Extension filter
            if ext_filter and not fname.lower().endswith(f".{ext_filter}"):
                continue

            # Name pattern filter (simple substring / glob)
            if name_pattern:
                p_temp = Path(fname)
                if not p_temp.match(name_pattern) and name_pattern not in fname:
                    continue

            full_path = Path(dirpath) / fname

            # Content filter — plain substring search, no regex
            if content_pattern:
                try:
                    text = full_path.read_text(encoding="utf-8", errors="ignore")
                    if content_pattern not in text:
                        continue
                except (PermissionError, OSError):
                    continue

            matches.append(str(full_path))

        if len(matches) >= max_results:
            break

    return {"matches": matches, "count": len(matches), "truncated": len(matches) >= max_results}


async def _delete_file(
    path: str,
    confirm_fn: Callable,
    restricted_paths: list[str],
) -> dict[str, Any]:
    p = Path(path).expanduser()

    if _is_restricted(p, restricted_paths):
        return {"error": f"Path is restricted and cannot be deleted: {p}"}

    if not p.exists():
        return {"error": f"File not found: {p}"}
    if not p.is_file():
        return {"error": f"Not a regular file: {p}"}

    approved = await confirm_fn(["delete", str(p)])
    if not approved:
        return {"status": "blocked", "reason": "User denied file deletion."}

    try:
        p.unlink()
        return {"status": "deleted", "path": str(p)}
    except PermissionError:
        return {"error": f"Permission denied: {p}"}
    except Exception as exc:
        return {"error": str(exc)}


def register_file_tools(
    registry: ToolRegistry,
    confirm_fn: Callable,
    restricted_paths: list[str],
    max_search_results: int = 50,
    max_write_bytes: int = 10 * 1024 * 1024,
) -> None:
    registry.register(Tool(
        name="list_directory",
        description="List files and subdirectories. Supports optional glob pattern filter.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list."},
                "pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter filenames (e.g. '*.py').",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_list_directory,
    ))

    registry.register(Tool(
        name="write_file",
        description=(
            "Write text content to a file. Creates parent directories as needed. "
            "Requires confirmation if the file already exists. "
            "Blocked for restricted paths."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write."},
                "content": {"type": "string", "description": "Text content to write."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=lambda path, content: _write_file(
            path, content, confirm_fn, restricted_paths, max_write_bytes
        ),
    ))

    registry.register(Tool(
        name="search_files",
        description=(
            "Recursively search a directory for files matching name, extension, "
            "or content criteria."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Root directory to search."},
                "name_pattern": {
                    "type": "string",
                    "description": "Glob or substring pattern for filenames (e.g. '*.log').",
                },
                "extension": {
                    "type": "string",
                    "description": "File extension filter, with or without dot (e.g. 'py').",
                },
                "content_pattern": {
                    "type": "string",
                    "description": "Substring to search for inside file contents.",
                },
                "max_results": {
                    "type": "integer",
                    "default": 50,
                    "description": "Maximum number of results to return.",
                },
            },
            "required": ["directory"],
            "additionalProperties": False,
        },
        handler=lambda directory, name_pattern=None, extension=None,
        content_pattern=None, max_results=max_search_results: _search_files(
            directory, name_pattern, extension, content_pattern, max_results
        ),
    ))

    registry.register(Tool(
        name="delete_file",
        description=(
            "Delete a file. Always requires confirmation. Blocked for restricted paths."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to delete."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=lambda path: _delete_file(path, confirm_fn, restricted_paths),
    ))
