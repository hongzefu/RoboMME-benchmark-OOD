"""L0: the hard package's three recording copies differ from upstream only by the whitelist; ``UPSTREAM.json`` shims and self-signed sha hold (C01 shim part, C18).

The upstream side is always read from git objects ``016ac1c4:src/robomme/...`` (this repo branched from upstream ``016ac1c4``),
independent of the working-tree state of ``src/robomme``; shim target sha values are likewise recomputed against ``016ac1c4``.
The whitelist lists lines exactly (removed lines, added lines); any extra difference, including comments, is out of bounds.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from tests.robomme_ood._support.loaders import REPO

HARD = REPO / "src" / "robomme_ood"
#: Upstream anchor (full 40-char sha): this repo branched from it
OFFICIAL_COMMIT = "016ac1c4ef3df2b88488abc19db08f3de83647b5"
MANIFEST = HARD / "UPSTREAM.json"

#: copy -> line-diff whitelist against upstream: [(lines removed from upstream, lines added in the copy), ...], in order of appearance.
WHITELIST: dict[str, list[tuple[list[str], list[str]]]] = {
    "env_record_wrapper/RecordWrapper.py": [
        (
            [
                "        # Force terminate episode if environment steps exceed preset safety limit (2000 steps)",
                "        fail_safe_limit = 2000",
            ],
            [
                "        # Force terminate episode if environment steps exceed preset safety limit (originally 2000 steps, 5000 since V4)",
                "        # V4 (2026-09-22, user explicitly authorized unfreezing this single spot: 'raise the recording limit from 2000 to 5000 steps'):",
                "        # with PickXtimes xhard num up to 15 the demo takes about 136+138×num≈2206 steps, so the old 2000-step cap would always kill it.",
                "        # All successful episodes of the original three tiers end within 2000 steps; relaxing the cap changes none of their outputs (verified by a local before/after comparison in V1).",
                "        fail_safe_limit = 5000",
            ],
        ),
        (
            ["                from robomme.robomme_env.utils.vqa_options import get_vqa_options"],
            ["                from robomme_ood.robomme_env.utils.vqa_options import get_vqa_options"],
        ),
    ],
    "env_record_wrapper/OraclePlannerDemonstrationWrapper.py": [
        (
            ["from robomme.robomme_env.utils.vqa_options import get_vqa_options"],
            ["from robomme_ood.robomme_env.utils.vqa_options import get_vqa_options"],
        ),
    ],
    "env_record_wrapper/DemonstrationWrapper.py": [],
}


def _manifest_raw() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _git_show(commit: str, rel: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=REPO, check=True, capture_output=True).stdout


def line_diff(official: str, copy: str) -> list[tuple[list[str], list[str]]]:
    """Line-diff blocks: [(lines removed from upstream, lines added in the copy)] (identical parts excluded)."""
    a, b = official.splitlines(), copy.splitlines()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return [(a[i1:i2], b[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


@pytest.mark.parametrize("rel", sorted(WHITELIST))
def test_copy_differs_from_official_exactly_by_whitelist(rel):
    official = _git_show(OFFICIAL_COMMIT, f"src/robomme/{rel}").decode()
    copy = (HARD / rel).read_text(encoding="utf-8")
    assert line_diff(official, copy) == WHITELIST[rel]
    if not WHITELIST[rel]:
        # Byte-identical (including line endings and the trailing newline).
        assert (HARD / rel).read_bytes() == official.encode()


def test_line_diff_detects_reverted_limit_and_extra_comment():
    """Checker negatives: changing 5000 back to 2000, or adding one extra comment line, makes the diff no longer equal the whitelist."""
    rel = "env_record_wrapper/RecordWrapper.py"
    official = _git_show(OFFICIAL_COMMIT, f"src/robomme/{rel}").decode()
    copy = (HARD / rel).read_text(encoding="utf-8")
    assert line_diff(official, copy.replace("fail_safe_limit = 5000", "fail_safe_limit = 2000")) != WHITELIST[rel]
    assert line_diff(official, copy.replace("import gymnasium", "# one extra line\nimport gymnasium", 1)) != WHITELIST[rel]
    assert line_diff(official, official) == []


# ---------------------------------------------------------------- UPSTREAM.json


def test_manifest_self_signature_holds():
    m = _manifest_raw()
    claimed = m.pop("manifest_sha256")
    canonical = json.dumps(m, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == claimed


SHIM_IMPORT = "import importlib, sys"
SHIM_ALIAS = re.compile(r'^sys\.modules\[__name__\] = importlib\.import_module\("([A-Za-z0-9_.]+)"\)$')


def _shim_alias(lines: list[str]) -> str | None:
    """Returns the alias target module if the three-line shim form (comment, import, alias) holds, otherwise None."""
    if len(lines) != 3 or not lines[0].startswith("# Borrowed: ") or lines[1] != SHIM_IMPORT:
        return None
    m = SHIM_ALIAS.match(lines[2])
    return m.group(1) if m else None


def _shim_files() -> dict[str, str]:
    """Git-tracked hard-package .py files whose content has the three-line shim form -> {path: alias target module} (derived from content, not from a path formula)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", "src/robomme_ood"], cwd=REPO, check=True,
                         capture_output=True).stdout
    found = {}
    for rel in (x.decode() for x in out.split(b"\0") if x):
        if not rel.endswith(".py"):
            continue
        target = _shim_alias((REPO / rel).read_text(encoding="utf-8").splitlines())
        if target is not None:
            found[rel] = target
    return found


def test_shim_alias_form_negatives():
    ok = ["# Borrowed: x", SHIM_IMPORT, 'sys.modules[__name__] = importlib.import_module("robomme.a")']
    assert _shim_alias(ok) == "robomme.a"
    assert _shim_alias(ok + ["X = 1"]) is None
    assert _shim_alias(ok[1:]) is None
    assert _shim_alias([ok[0], "import importlib", ok[2]]) is None


def test_shim_registry_equals_shim_files_on_disk():
    m = _manifest_raw()
    registered = {s["shim"]: s["target_module"] for s in m["shims"]}
    assert len(registered) == len(m["shims"])
    # The shim set and alias targets derived from file content match the manifest registration exactly.
    assert _shim_files() == registered


@pytest.mark.parametrize("entry", _manifest_raw()["shims"], ids=lambda e: e["target_module"])
def test_shim_form_and_target(entry):
    m = _manifest_raw()
    path = REPO / entry["shim"]
    lines = path.read_text(encoding="utf-8").splitlines()
    # Exactly three lines: one comment (pointing to this manifest), import, alias; one more or one fewer is rejected.
    assert len(lines) == 3, lines
    assert lines[0].startswith("# Borrowed: ")
    rel_manifest = lines[0].rsplit("manifest: ", 1)[1].strip()
    assert (path.parent / rel_manifest).resolve() == MANIFEST.resolve()
    assert lines[1] == SHIM_IMPORT
    assert _shim_alias(lines) == entry["target_module"]
    # The target file is registered in the upstream manifest; its sha and byte size are recomputed independently from upstream git objects.
    blob = _git_show(OFFICIAL_COMMIT, entry["target_file"])
    assert entry["target_sha256"] == hashlib.sha256(blob).hexdigest()
    assert entry["target_bytes"] == len(blob)
    assert m["robomme_files"][entry["target_file"]] == entry["target_sha256"]
