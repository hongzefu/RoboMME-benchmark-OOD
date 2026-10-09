"""Companion status plugin for the T4 mutation plugin (tests/robomme_ood/unit/hard/mutants_plugin.py): does not modify that plugin file, only adds two checks around it.

- Before mutants_plugin's pytest_configure (tryfirst): replace its ``_redefine`` with a version requiring the source fragment to match exactly once
  (the original only requires at least once and replaces only the first match); if violated, record applied=false and raise, aborting the session so no test is counted as caught;
- pytest_sessionstart (by then mutants_plugin's pytest_configure has finished without raising): record applied=true.

Status is written to MUT_STATUS_FILE (JSON); this plugin does nothing when T4_MUTANT is unset.
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
    import mutants_plugin  # same module object as -p mutants_plugin

    orig = mutants_plugin._redefine

    def checked(mod, qualname, old, new, cls=None):
        holder = cls if cls is not None else mod
        src = textwrap.dedent(inspect.getsource(getattr(holder, qualname)))
        n = src.count(old)
        if n != 1:
            reason = f"{qualname} mutation fragment matched {n} times: {old!r}"
            _write(False, reason)
            raise AssertionError(reason)
        return orig(mod, qualname, old, new, cls=cls)

    mutants_plugin._redefine = checked
    _write(False, "mutation not completed")  # write the failure state first; only switched to success in pytest_sessionstart once mutation completes


def pytest_sessionstart(session):
    # mutants_plugin's pytest_configure has finished without raising, so the mutation is in effect
    if _NAME:
        _write(True, None)
