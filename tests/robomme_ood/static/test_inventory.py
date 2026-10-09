"""L0: contract-inventory meta test (plan details 4.3, 4.5.1): turns "every part is covered" and "contracts are really executed" into two machine-checkable verdict lines.

All expectations come from independent sources, not from the code under test:
- File universe = ``git ls-files src/robomme_ood scripts challenge_interface``; every current file must be in exactly one of the master table
  ``tests/robomme_ood/contract/benchmark_contracts.json``'s ``files`` (list of owning contract ids) or ``exempt`` (exemption category and reason);
  contract ids in ``files`` must exist in ``entries``, and no longer-existing files may be registered. Keys ending in ``/`` register a whole directory
  (``src/robomme/`` is the frozen upstream package, covered as a whole by the C18 upstream byte guard), and that directory must contain git-tracked files.
- Test universe = union of the actual collection results of two subprocesses (more accurate than parsing ``def`` names: sees parametrization, conditionally skipped modules and slow tests):
  ``pytest --collect-only -q tests`` (about 7 s measured; the resource guard skips ``tests/robomme_ood/sim`` here) and
  ``pytest --collect-only -q --allow-sim-reset tests/robomme_ood/sim`` (collect only, no reset; nodeids of L4 simulation smoke entries are checked against it).
- Contract id collision: the same id appearing in several ``contracts.delta.json`` files whose source file names are disjoint is treated as two different contracts
  and reported as a conflict (not concatenated when merging into the master table, see the master table's ``merge_rule``).

Two verdict lines:
- ``TEST_INVENTORY=PASS|FAIL unclassified=<n> stale=<n> exempt=<n>``
- ``TEST_CONTRACTS=PASS|FAIL entries=<n> verified=<n> conditional=<n> blocked=<n> planned=<n> missing=<n> pending=<n>``

``missing`` is the number of nodeids of verified entries (and those registered in conditional/blocked entries) that, with the parametrization suffix removed, are not found in the collection results;
``pending`` is the number of planned entries, i.e. coverage gaps with no execution evidence yet.

**Verdict line vs. pass condition of this test**: when ``pending`` is not 0, the ``TEST_CONTRACTS`` verdict line truthfully says FAIL and lists the planned entries;
it is a gap report for the acceptance table (``pending=0``). As a daily gate, this meta test only requires ``missing=0``, ``unclassified=0``,
``stale=0`` and a valid structure (status values, verified entries have nodeids, blocked/conditional entries have a note, no collisions across deltas); it does not fail on pending.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from tests.robomme_ood._support.loaders import REPO
from tests.robomme_ood._support.resource_policy import ENV_LEDGER, ENV_MODE

TOTAL = REPO / "tests" / "robomme_ood" / "contract" / "benchmark_contracts.json"
ROOTS = ("src/robomme_ood", "scripts", "challenge_interface")
STATUSES = ("planned", "verified", "blocked", "conditional")
NOTE_KEYS = ("note",)  # note field for blocked/conditional


# ---------------------------------------------------------------- independent universes


def git_files(*paths: str) -> set[str]:
    out = subprocess.run(["git", "ls-files", "--", *paths], cwd=REPO, capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def base_nodeid(nodeid: str) -> str:
    """Strip the parametrization suffix: ``a.py::test_x[p]`` -> ``a.py::test_x``."""
    return nodeid.split("[", 1)[0]


def _collect(*args: str) -> set[str]:
    """A subprocess that only collects (``--collect-only`` runs no tests, and tests/robomme_ood/sim does not reset); returns the set of de-parametrized nodeids.

    The subprocess is an independent pytest session that installs its own resource guard: drop the parent session's guard environment
    variables so that sitecustomize and pytest_configure do not both install it and double-patch (observed recursion overflow).
    """
    env = {k: v for k, v in os.environ.items() if k not in (ENV_MODE, ENV_LEDGER)}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args],
        cwd=REPO, capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, f"collection failed {args} (exit={proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    ids = {base_nodeid(line.strip()) for line in proc.stdout.splitlines() if line.startswith("tests/") and "::" in line}
    assert ids, f"collection result is empty: {args}"
    return ids


def collect_nodeids() -> set[str]:
    """All tests (no -m, slow included) ∪ tests/robomme_ood/sim (collected only, with --allow-sim-reset)."""
    return _collect("tests/robomme_ood") | _collect("--allow-sim-reset", "tests/robomme_ood/sim")


_FILE_TOKEN = re.compile(r"[\w.-]+\.(?:py|jsonl|json|sh|html|toml)\b")


def source_files(source) -> set[str]:
    """Set of file names (basenames only) appearing in source (string or list); used to decide whether two same-id entries are the same contract."""
    items = source if isinstance(source, list) else [source]
    return {m.group(0) for x in items for m in _FILE_TOKEN.finditer(str(x))}


def delta_conflicts(deltas: dict[str, list[dict]]) -> list[tuple[str, str, str]]:
    """Same id across deltas with disjoint source file names -> (id, delta A, delta B)."""
    seen: dict[str, list[tuple[str, set[str]]]] = {}
    out = []
    for path, entries in sorted(deltas.items()):
        for e in entries:
            files = source_files(e.get("source", ""))
            for other_path, other_files in seen.get(e["id"], []):
                if other_path != path and not (files & other_files):
                    out.append((e["id"], other_path, path))
            seen.setdefault(e["id"], []).append((path, files))
    return sorted(out)


def load_deltas() -> dict[str, list[dict]]:
    out = {}
    for p in sorted((REPO / "tests" / "robomme_ood").rglob("contracts.delta.json")):
        out[str(p.relative_to(REPO))] = json.loads(p.read_text(encoding="utf-8"))
    assert out, "no contracts.delta.json found"
    return out


def nodeid_found(nodeid: str, collected: set[str], prefixes: set[str]) -> bool:
    """A registered nodeid may be function-, class- or file-level; it counts as found if it maps onto the collection results."""
    b = base_nodeid(nodeid)
    return b in collected or b in prefixes


def prefixes_of(collected: set[str]) -> set[str]:
    out = set()
    for n in collected:
        parts = n.split("::")
        for i in range(1, len(parts)):
            out.add("::".join(parts[:i]))
    return out


# ---------------------------------------------------------------- check functions (negatives feed broken copies directly)


def check_inventory(total: dict, tracked: set[str]) -> dict:
    files, exempt = total.get("files", {}), total.get("exempt", {})
    entry_ids = {e["id"] for e in total.get("entries", [])}
    file_keys = [k for k in files if not k.endswith("/")]
    dir_keys = [k for k in files if k.endswith("/")]
    registered = set(file_keys) | set(exempt)
    unclassified = sorted(tracked - registered)
    stale = sorted(registered - tracked)
    stale += sorted(d for d in dir_keys if not git_files(d))
    both = sorted(set(files) & set(exempt))
    unknown = sorted({(k, c) for k, v in files.items() for c in v if c not in entry_ids})
    empty = sorted(k for k, v in files.items() if not v)
    bad_exempt = sorted(k for k, v in exempt.items()
                        if not (isinstance(v, dict) and v.get("category") and v.get("reason")))
    ok = not (unclassified or stale or both or unknown or empty or bad_exempt)
    line = (f"TEST_INVENTORY={'PASS' if ok else 'FAIL'} unclassified={len(unclassified)} "
            f"stale={len(stale)} exempt={len(exempt)}")
    return dict(ok=ok, line=line, unclassified=unclassified, stale=stale, both=both,
                unknown=unknown, empty=empty, bad_exempt=bad_exempt)


def check_contracts(total: dict, collected: set[str], deltas: dict[str, list[dict]] | None = None) -> dict:
    entries = total.get("entries", [])
    keys = total["entry_keys"]
    domains = {d["id"] for d in total["domains"]}
    prefixes = prefixes_of(collected)
    counts = Counter(e.get("status") for e in entries)
    dup = sorted(i for i, n in Counter(e["id"] for e in entries).items() if n > 1)
    missing_keys = sorted(e["id"] for e in entries if any(k not in e for k in keys))
    bad_status = sorted(e["id"] for e in entries if e.get("status") not in STATUSES)
    bad_domain = sorted(e["id"] for e in entries if e.get("domain") not in domains)
    verified_empty = sorted(e["id"] for e in entries if e.get("status") == "verified" and not e.get("nodeids"))
    undocumented = sorted(e["id"] for e in entries if e.get("status") in ("blocked", "conditional")
                          and not any(str(e.get(k, "")).strip() for k in NOTE_KEYS))
    missing = sorted((e["id"], n) for e in entries if e.get("status") != "planned"
                     for n in e.get("nodeids", []) if not nodeid_found(n, collected, prefixes))
    pending = sorted(e["id"] for e in entries if e.get("status") == "planned")
    conflicts = delta_conflicts(deltas or {})
    structural_ok = not (dup or missing_keys or bad_status or bad_domain or verified_empty or undocumented or conflicts)
    gate_ok = structural_ok and not missing
    line_ok = gate_ok and not pending
    line = (f"TEST_CONTRACTS={'PASS' if line_ok else 'FAIL'} entries={len(entries)} "
            f"verified={counts['verified']} conditional={counts['conditional']} blocked={counts['blocked']} "
            f"planned={counts['planned']} missing={len(missing)} pending={len(pending)}")
    return dict(ok=gate_ok, line_ok=line_ok, line=line, dup=dup, missing_keys=missing_keys, bad_status=bad_status,
                bad_domain=bad_domain, verified_empty=verified_empty, undocumented=undocumented,
                conflicts=conflicts, missing=missing, pending=pending)


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def total() -> dict:
    return json.loads(TOTAL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tracked() -> set[str]:
    files = git_files(*ROOTS)
    assert files, "git ls-files is empty"
    return files


@pytest.fixture(scope="module")
def collected() -> set[str]:
    return collect_nodeids()


@pytest.fixture(scope="module")
def deltas() -> dict[str, list[dict]]:
    return load_deltas()


# ---------------------------------------------------------------- positive: real master table


def test_inventory_every_tracked_file_classified(total, tracked):
    r = check_inventory(total, tracked)
    print(r["line"])
    assert r["ok"], {k: r[k] for k in ("unclassified", "stale", "both", "unknown", "empty", "bad_exempt")}


def test_contracts_nodeids_collected(total, collected, deltas):
    r = check_contracts(total, collected, deltas)
    print(r["line"])
    if r["pending"]:
        print("planned (coverage gaps, reported only, not gating): " + ", ".join(r["pending"]))
    assert r["ok"], {k: r[k] for k in ("dup", "missing_keys", "bad_status", "bad_domain",
                                       "verified_empty", "undocumented", "conflicts", "missing")}


def test_sim_nodeids_collected_without_reset(collected):
    """Nodeids of L4 entries come from the collect-only subprocess with --allow-sim-reset; collect only, no reset is triggered."""
    assert "tests/robomme_ood/sim/test_reset_matrix.py::test_reset_cell" in collected
    assert "tests/robomme_ood/sim/test_official_one_reset.py::test_official_reset_and_unreachable_ee_step" in collected


def test_src_robomme_registered_as_directory(total):
    """The frozen upstream package is registered as one whole-directory entry attached to the upstream byte guard."""
    assert "C18-UPSTREAM-BYTES" in total["files"].get("src/robomme/", [])


# ---------------------------------------------------------------- negatives: broken tmp copies must FAIL


def _tmp_copy(total: dict, tmp_path: Path) -> dict:
    p = tmp_path / "benchmark_contracts.json"
    p.write_text(json.dumps(total, ensure_ascii=False), encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


def test_negative_unregistered_file_fails(total, tracked, tmp_path):
    bad = _tmp_copy(total, tmp_path)
    victim = sorted(k for k in bad["files"] if not k.endswith("/"))[0]
    del bad["files"][victim]
    r = check_inventory(bad, tracked)
    assert not r["ok"] and r["unclassified"] == [victim]
    assert r["line"].startswith("TEST_INVENTORY=FAIL unclassified=1 ")


def test_negative_stale_and_unknown_id_fail(total, tracked, tmp_path):
    bad = _tmp_copy(total, tmp_path)
    bad["files"]["scripts/nonexistent_file.py"] = ["C18-ENTRY-SET"]
    bad["files"]["scripts/run_example.py"] = ["C99-nonexistent"]
    bad["files"]["no/such/dir/"] = ["C18-UPSTREAM-BYTES"]
    r = check_inventory(bad, tracked)
    assert not r["ok"]
    assert "scripts/nonexistent_file.py" in r["stale"] and "no/such/dir/" in r["stale"]
    assert ("scripts/run_example.py", "C99-nonexistent") in r["unknown"]


def test_negative_nodeid_not_collected_fails(total, collected, tmp_path):
    bad = _tmp_copy(total, tmp_path)
    e = next(x for x in bad["entries"] if x["status"] == "verified")
    e["nodeids"] = e["nodeids"] + ["tests/robomme_ood/static/test_inventory.py::test_nonexistent_case[p0]"]
    r = check_contracts(bad, collected)
    assert not r["ok"] and r["missing"] == [(e["id"], "tests/robomme_ood/static/test_inventory.py::test_nonexistent_case[p0]")]
    assert r["line"].startswith("TEST_CONTRACTS=FAIL ") and " missing=1 " in r["line"]


def test_negative_verified_without_nodeids_fails(total, collected, tmp_path):
    bad = _tmp_copy(total, tmp_path)
    e = next(x for x in bad["entries"] if x["status"] == "verified")
    e["nodeids"] = []
    r = check_contracts(bad, collected)
    assert not r["ok"] and r["verified_empty"] == [e["id"]]


def test_negative_status_and_note_rules(total, collected, tmp_path):
    bad = _tmp_copy(total, tmp_path)
    cond = next(x for x in bad["entries"] if x["status"] == "conditional")
    cond.pop("note")
    odd = next(x for x in bad["entries"] if x["status"] == "verified")
    odd["status"] = "done"
    r = check_contracts(bad, collected)
    assert not r["ok"]
    assert r["undocumented"] == [cond["id"]] and r["bad_status"] == [odd["id"]]


def test_pending_only_fails_line_not_gate(total, collected):
    """With only planned entries: verdict line FAIL, gate passes; the distinction is documented in the module docstring."""
    only = copy.deepcopy(total)
    only["entries"] = [e for e in only["entries"] if e["status"] == "verified"]
    gap = copy.deepcopy(only["entries"][0])
    gap.update(id="T11-synthetic-gap", status="planned", nodeids=[])
    only["entries"].append(gap)
    r = check_contracts(only, collected)
    assert r["pending"] == ["T11-synthetic-gap"]
    assert r["ok"] and not r["line_ok"] and r["line"].startswith("TEST_CONTRACTS=FAIL ")


def test_negative_bad_exempt_and_both_fail(total, tracked, tmp_path):
    bad = _tmp_copy(total, tmp_path)
    no_reason = next(iter(bad["exempt"]))
    bad["exempt"][no_reason] = {"category": "documentation"}
    in_files = next(k for k in bad["files"] if not k.endswith("/"))
    bad["exempt"][in_files] = {"category": "documentation", "reason": "also registered in files"}
    r = check_inventory(bad, tracked)
    assert not r["ok"]
    assert r["bad_exempt"] == [no_reason] and r["both"] == [in_files]


def test_negative_delta_id_collision_fails(total, collected):
    """Same id across blocks with disjoint sources (two different contracts colliding) must be reported as a conflict; same id with overlapping sources is just a supplement to the same contract."""
    fake = {
        "tests/a/contracts.delta.json": [{"id": "C16.99", "source": ["scripts/injection-dev/site_build.py"]}],
        "tests/b/contracts.delta.json": [{"id": "C16.99", "source": "scripts/parity/noise_run_gl.sh"}],
        "tests/c/contracts.delta.json": [{"id": "C08.98", "source": ["src/robomme/x/A.py", "src/robomme/x/B.py"]}],
        "tests/d/contracts.delta.json": [{"id": "C08.98", "source": "src/robomme/x/B.py, src/robomme/x/C.py"}],
    }
    assert delta_conflicts(fake) == [("C16.99", "tests/a/contracts.delta.json", "tests/b/contracts.delta.json")]
    r = check_contracts(total, collected, fake)
    assert not r["ok"] and r["conflicts"]
