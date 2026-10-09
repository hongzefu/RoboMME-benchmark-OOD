"""Outcome recorder plugin for the mutation runner: writes each test's per-phase outcome, exception type and collection errors to MUT_OUTCOME_FILE (JSON).

Active only when the MUT_OUTCOME_FILE environment variable is set; does not change any behavior under test.
"""
from __future__ import annotations

import json
import os

import pytest

_PATH = os.environ.get("MUT_OUTCOME_FILE")
_TESTS: dict[str, dict] = {}
_COLLECT_ERRORS: list[dict] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if not _PATH:
        return
    rep = outcome.get_result()
    rec = _TESTS.setdefault(rep.nodeid, {"phases": {}})
    exc = call.excinfo.typename if call.excinfo is not None else None
    top_file = None
    if call.excinfo is not None and rep.failed:
        # File of the top stack frame (innermost frame, i.e. the raise site), relative to the repo root; the runner uses it to exclude failures raised by the mutation plugin itself
        try:
            path = str(call.excinfo.traceback[-1].path)
            root = str(item.config.rootpath)
            top_file = os.path.relpath(path, root) if os.path.isabs(path) and path.startswith(root + os.sep) else path
        except Exception:  # leave empty when the top frame is unavailable; the runner treats it as "unknown source" and does not count it as caught
            top_file = None
    msg = None
    if rep.failed and rep.longrepr is not None:
        crash = getattr(rep.longrepr, "reprcrash", None)
        msg = (crash.message if crash is not None else str(rep.longrepr))[:400]
    rec["phases"][rep.when] = {"outcome": rep.outcome, "exc": exc, "top_file": top_file, "msg": msg}


def pytest_collectreport(report):
    if _PATH and report.failed:
        _COLLECT_ERRORS.append({"nodeid": report.nodeid, "msg": str(report.longrepr)[-600:]})


def pytest_sessionfinish(session, exitstatus):
    if not _PATH:
        return
    with open(_PATH, "w", encoding="utf-8") as fh:
        json.dump({"exitstatus": int(exitstatus), "tests": _TESTS, "collect_errors": _COLLECT_ERRORS}, fh,
                  ensure_ascii=False)
