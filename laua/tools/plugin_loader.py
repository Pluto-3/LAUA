"""Plugin loader — scans ~/.laua/tools/ and ./tools/plugins/ for tool plugins."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SCAN_PATHS = [
    Path.home() / ".laua" / "tools",
    Path("tools") / "plugins",
]


def _load_module(path: Path) -> ModuleType | None:
    """Import a Python file as an anonymous module; return None on any error."""
    spec = importlib.util.spec_from_file_location(f"laua_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        logger.warning("Plugin %s: could not create module spec", path)
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        logger.warning("Plugin %s: error during load — %s", path, exc)
        return None
    return module


def load_plugins(
    registry: Any,
    session: Any,
    confirm_fn: Callable,
    audit_fn: Callable,
    extra_paths: list[Path] | None = None,
) -> int:
    """
    Scan plugin directories and call register() on every valid plugin.

    Returns the number of plugins successfully registered.
    """
    scan_dirs = list(_SCAN_PATHS)
    if extra_paths:
        scan_dirs.extend(extra_paths)

    loaded = 0
    for directory in scan_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for py_file in sorted(directory.glob("*.py")):
            module = _load_module(py_file)
            if module is None:
                continue
            register_fn = getattr(module, "register", None)
            if not callable(register_fn):
                logger.warning(
                    "Plugin %s: no register() function found — skipped", py_file.name
                )
                continue
            try:
                register_fn(registry, session, confirm_fn, audit_fn)
                logger.info("Loaded plugin: %s", py_file.name)
                loaded += 1
            except Exception as exc:
                logger.warning("Plugin %s: register() raised — %s", py_file.name, exc)

    return loaded
