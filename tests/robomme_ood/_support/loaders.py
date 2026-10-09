"""按文件路径加载 scripts/ 下的脚本模块（scripts 不是包，生产代码也按路径互相加载）。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"
_CACHE: dict[str, object] = {}


def script_path(rel: str) -> Path:
    """rel 相对 scripts/，如 ``parity/noise_gate.py``。"""
    p = SCRIPTS / rel
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_script(rel: str, *, fresh: bool = False):
    """加载脚本模块；同一路径默认复用同一模块对象，``fresh=True`` 时重新执行一份独立副本。

    模块名由相对路径派生（``_script_parity_noise_gate``），脚本目录临时加到 sys.path 头部，
    以便脚本里 ``import <同目录模块>`` 的写法照常工作。
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
