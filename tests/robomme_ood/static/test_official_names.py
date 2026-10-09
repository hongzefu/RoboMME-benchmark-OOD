"""L0: official-name residue check ``OFFICIAL_NAMES`` (1006-rename-official-names-and-stage3-eval-plan.md, part one sec. 1, part two sec. 8.2).

Repo-invented names are all replaced with official names: FrameSamp+Modulation's policy label ``perceptual-framesamp-modul``
(identifier ``framesamp_modul``), GroundSG's ``groundsg``, and the dataset interfaces ``hard-verify`` (former stage-two interface) and
``ood`` (former stage-three interface). This test scans git-tracked live code under ``scripts/``, ``src/robomme_ood/`` and ``tests/``;
any legacy name fails it.

Legacy spellings are listed independently in this file (not read from the code under test), see ``PATTERNS``; the only exemptions (whitelist) are:

- official family-name forms: ``mme-vla``, ``mme_vla``, ``MME-VLA``, ``MME_VLA``, upstream class prefix ``MMEVLAWebsocket``, ``mme_vla_suite``
  (the regexes themselves do not match these spellings);
- the alias table itself: lines between ``# >>> LEGACY_NAMES`` and ``# <<< LEGACY_NAMES`` in ``tests/robomme_ood/contract/test_constants.py``
  (this repo keeps only the two legacy dataset names the builder must reject; the evaluation-side alias table ``official_defs.py`` lives in the private evaluation repo);
- lines marked "legacy dir name" (real disk/NFS paths) or "legacy data key" (keys in published data files);
- the ``XHARD0_IN_TEST_HARD`` family (still written in negative assertions for deleted symbols; the regexes do not match it);
- this file itself.

Not scanned: the three verbatim upstream entries, in-package spec data
``src/robomme_ood/env_metadata/``, the upstream byte manifest ``UPSTREAM.json``, documentation ``*.md`` (current docs are updated separately by the main session).

Verdict line: ``OFFICIAL_NAMES=PASS|FAIL files=<n> hits=<n>``; on failure each hit is listed as ``<path>:<line>: <hit>``.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests.robomme_ood._support.loaders import REPO

ROOTS = ("scripts", "src/robomme_ood", "tests")
SKIP_PREFIX = ("src/robomme_ood/env_metadata/",)
SKIP_FILES = {"scripts/dataset_replay.py", "scripts/evaluation.py", "scripts/run_example.py",
              "src/robomme_ood/UPSTREAM.json", "tests/robomme_ood/static/test_official_names.py"}
SKIP_SUFFIX = (".md", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mkv", ".npz", ".h5", ".pkl", ".ico", ".woff", ".woff2")
ALIAS_FILE = "tests/robomme_ood/contract/test_constants.py"
ALIAS_BEGIN, ALIAS_END = "# >>> LEGACY_NAMES", "# <<< LEGACY_NAMES"
LINE_MARKERS = ("legacy dir name", "legacy data key")

#: (name, regex): all spellings of legacy names
PATTERNS = (
    ("mme label", re.compile(r"(?<![A-Za-z0-9_])mme(?![A-Za-z0-9_]|-vla)")),
    ("mme identifier", re.compile(r"(?<![A-Za-z0-9])mme_(?!vla)|(?<=[A-Za-z0-9])_mme(?![A-Za-z0-9]|_vla)")),
    ("MME name", re.compile(r"(?<![A-Za-z0-9_])MME(?![A-Za-z0-9_]|-VLA|-vla)")),
    ("MME identifier", re.compile(r"(?<![A-Za-z0-9])MME_(?!VLA)|(?<=[A-Za-z0-9])_MME(?![A-Za-z0-9]|_VLA)")),
    ("MME camel case", re.compile(r"(?<=[a-z])(?<!Robo)MME(?!VLA)")),
    ("mmesg", re.compile(r"mmesg", re.IGNORECASE)),
    ("mmevla", re.compile(r"mmevla(?!websocket)", re.IGNORECASE)),
    ("test-hard", re.compile(r"test-hard", re.IGNORECASE)),
    ("TEST_HARD constant", re.compile(r"(?<![A-Za-z0-9_])TEST_HARD0?(?![A-Za-z0-9_])")),
    ("test_hard0", re.compile(r"test_hard0", re.IGNORECASE)),
)


def scan_text(path: str, text: str) -> list[tuple[str, int, str]]:
    """Hits in one file: [(path, line number, hit excerpt)]; the alias-table block and marked lines are skipped per the whitelist."""
    hits = []
    in_alias = False
    for no, line in enumerate(text.splitlines(), 1):
        if path == ALIAS_FILE:
            if line.strip().startswith(ALIAS_BEGIN):
                in_alias = True
                continue
            if line.strip().startswith(ALIAS_END):
                in_alias = False
                continue
        if in_alias or any(m in line for m in LINE_MARKERS):
            continue
        for name, pat in PATTERNS:
            m = pat.search(line)
            if m:
                lo = max(0, m.start() - 20)
                hits.append((path, no, f"{name} …{line[lo:m.end() + 20].strip()}…"))
                break
    return hits


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", *ROOTS], cwd=REPO, capture_output=True, text=True, check=True)
    files = []
    for f in out.stdout.split():
        if f.startswith(SKIP_PREFIX) or f in SKIP_FILES or f.lower().endswith(SKIP_SUFFIX):
            continue
        if (REPO / f).is_file():
            files.append(f)
    return sorted(files)


def scan_repo() -> tuple[int, list[tuple[str, int, str]]]:
    files = tracked_files()
    hits = []
    for f in files:
        try:
            text = (REPO / f).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits += scan_text(f, text)
    return len(files), hits


def verdict_line(n_files: int, hits: list) -> str:
    return f"OFFICIAL_NAMES={'FAIL' if hits else 'PASS'} files={n_files} hits={len(hits)}"


# ---------------------------------------------------------------- positive: zero residue in the real repo


def test_official_names_no_legacy_residue():
    n, hits = scan_repo()
    assert n > 100, f"unexpected number of scanned files: {n}"
    for p, no, what in hits:
        print(f"OFFICIAL_NAMES_HIT {p}:{no}: {what}")
    print(verdict_line(n, hits))
    assert not hits, "\n".join(f"{p}:{no}: {w}" for p, no, w in hits[:50])


def test_alias_block_present_and_closed():
    """The alias-table block must exist, be paired, and appear only in test_constants.py (the only alias table in the repo)."""
    text = (REPO / ALIAS_FILE).read_text(encoding="utf-8")
    assert text.count(ALIAS_BEGIN) == 1 and text.count(ALIAS_END) == 1
    assert text.index(ALIAS_BEGIN) < text.index(ALIAS_END)
    others = [f for f in tracked_files() if f != ALIAS_FILE and ALIAS_BEGIN in (REPO / f).read_text(encoding="utf-8")]
    assert others == []


# ---------------------------------------------------------------- negatives: every legacy spelling is caught; official names and exemptions are not flagged


@pytest.mark.parametrize("line", [
    'POLICIES = ("mme", "smvla")',
    "x = load_sibling('mme_client')",
    "--mme-variant ground-sg-oracle",
    "MME_CKPT=/x",
    "WALL_MME=1",
    "x = setup_mme_run()",
    "upstream MME client",
    "route = 'mmesg/oracle/new'",
    "MMESG_ORIG_BLOCKED",
    "label mmevla",
    'dataset="test-hard"',
    "--dataset test-hard0",
    "TEST_HARD0 = 1",
    "from x import TEST_HARD",
    "def test_hard0_rejects():",
    "class FakeMMEClient:",
])
def test_scan_catches_legacy_spellings(line):
    assert scan_text("scripts/x.py", line), line


@pytest.mark.parametrize("line", [
    "third_party/mme-vla/examples/robomme/eval.py",
    "from mme_vla_suite import x",
    "MME_VLA_PY=/py",
    "_ENV_MME_VLA_PY=1; preflight_mme_vla x",
    "class MMEVLAWebsocketClientPolicy: ...",
    "MME-VLA family",
    "robomme_ood robomme RoboMME RoboMME-ood",
    "class FakeMMEVLAWebsocketClient(MMEVLAWebsocketClientPolicy): ...",
    "XHARD0_IN_TEST_HARD = os.environ.get('ROBOMME_OOD_XHARD0_IN_TEST_HARD')",
    "policy = 'perceptual-framesamp-modul'; mod = 'framesamp_modul_client'; v = 'groundsg'",
    'dataset="ood" or "hard-verify"',
    "def test_hard_parity_cli(): ...",
    "CKPT=/nfs/x/mmevla-ckpt/79999  # legacy dir name",
    "ids = ('mmevla',)  # legacy data key",
])
def test_scan_ignores_official_and_whitelisted(line):
    assert scan_text("scripts/x.py", line) == [], line


def test_alias_block_whitelist_only_in_alias_file():
    block = f"{ALIAS_BEGIN}\nA = {{'mme': 1}}\n{ALIAS_END}\nB = 'mmesg'\n"
    assert [h[1] for h in scan_text(ALIAS_FILE, block)] == [4]  # exempt inside the block, caught outside
    assert len(scan_text("scripts/other.py", block)) == 2  # the same block in other files is not exempt


def test_verdict_line_format():
    assert verdict_line(3, []) == "OFFICIAL_NAMES=PASS files=3 hits=0"
    assert verdict_line(3, [("a", 1, "x")]) == "OFFICIAL_NAMES=FAIL files=3 hits=1"
