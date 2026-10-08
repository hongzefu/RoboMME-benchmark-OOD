"""T4 植入插件（tests/robomme_hard/unit/hard/mutants_plugin.py）的配套状态插件：不改该插件文件，只在它运行前后加两道核对。

- 先于 mutants_plugin 的 pytest_configure（tryfirst）：把它的 ``_redefine`` 换成「源码片段必须恰好命中 1 次」的版本
  （原版只要求至少 1 次，且只替换第一处）；不满足时记 applied=false 并抛错，会话中止、不会有任何用例被计为抓到；
- pytest_sessionstart（此时 mutants_plugin 的 pytest_configure 已走完且未抛错）：记 applied=true。

状态写到 MUT_STATUS_FILE（JSON）；未设 T4_MUTANT 时本插件不做任何事。
"""
from __future__ import annotations

import inspect
import json
import os
import textwrap

import pytest

_NAME = os.environ.get("T4_MUTANT")


def _write(applied: bool, reason: str | None) -> None:
    path = os.environ.get("MUT_STATUS_FILE")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"key": f"T4:{_NAME}", "applied": applied, "reason": reason}, fh, ensure_ascii=False)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if not _NAME:
        return
    import mutants_plugin  # 与 -p mutants_plugin 是同一个模块对象

    orig = mutants_plugin._redefine

    def checked(mod, qualname, old, new, cls=None):
        holder = cls if cls is not None else mod
        src = textwrap.dedent(inspect.getsource(getattr(holder, qualname)))
        n = src.count(old)
        if n != 1:
            reason = f"{qualname} 植入片段命中 {n} 次：{old!r}"
            _write(False, reason)
            raise AssertionError(reason)
        return orig(mod, qualname, old, new, cls=cls)

    mutants_plugin._redefine = checked
    _write(False, "植入未完成")  # 先写失败态；植入正常走完才在 pytest_sessionstart 里改成成功


def pytest_sessionstart(session):
    # mutants_plugin 的 pytest_configure 已经跑完且未抛错，植入才算生效
    if _NAME:
        _write(True, None)
