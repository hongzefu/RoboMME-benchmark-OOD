"""Load script modules under scripts/ by file path (scripts is not a package; production code also loads them by path)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"
_CACHE: dict[str, object] = {}


def script_path(rel: str) -> Path:
    """rel is relative to scripts/, e.g. ``parity/noise_gate.py``."""
    p = SCRIPTS / rel
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_script(rel: str, *, fresh: bool = False):
    """Load a script module; the same path reuses the same module object by default, ``fresh=True`` executes a separate fresh copy.

    The module name is derived from the relative path (``_script_parity_noise_gate``); the script directory is temporarily
    prepended to sys.path so that ``import <sibling module>`` in scripts works as usual.
    """
    path = script_path(rel)
    key = str(path)
    if not fresh and key in _CACHE:
        return _CACHE[key]
    name = "_script_" + rel.removesuffix(".py").replace("/", "_").replace("-", "_")
    if fresh:
        name += f"_{len(_CACHE)}_{id(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    d = str(path.parent)
    added = d not in sys.path
    if added:
        sys.path.insert(0, d)
    try:
        spec.loader.exec_module(mod)
    finally:
        if added:
            try:
                sys.path.remove(d)
            except ValueError:
                pass
    if not fresh:
        _CACHE[key] = mod
    return mod
