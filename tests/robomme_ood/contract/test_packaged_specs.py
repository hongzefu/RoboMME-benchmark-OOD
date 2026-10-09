"""L1 contract: the five packaged delivery specs (``src/robomme_ood/env_metadata/ood/xhard{1..5}/specs.jsonl``) are untouched and self-consistent row by row.

- file byte sha256 equals the pin table ``tests/robomme_ood/contract/packaged_specs.sha256`` (sha256sum format, the pin table is in git);
- per-file ``load_specs`` and whole-root ``load_specs_root`` pass;
- per row: ``spec.task == row.task``, ``spec.identity`` task/seed/difficulty/episode match the row,
  header ``sampling_config`` key set equals ``tasks``, rows with ``exec_steps`` do not exceed the execution-step cap;
- all 1518 row seeds are globally unique and disjoint from the official metadata (train/val/test) seeds (the hard package no longer ships train metadata).

Spec rows are read directly with stdlib json (not via the reader under test); functions under test are called only in the "passes validation" tests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.robomme_ood._support.loaders import REPO
from tests.robomme_ood.contract.test_constants import (
    EXEC_CAP,
    NEW_TIERS,
    PACKAGED_ROWS,
    PACKAGED_ROWS_TOTAL,
    PACKAGED_SELECTED,
    SPEC_KIND,
    SPECS_SCHEMA,
    TOTAL,
    V9_CELLS,
)

ROOT = REPO / "src" / "robomme_ood" / "env_metadata" / "ood"
PIN = Path(__file__).with_name("packaged_specs.sha256")
OFFICIAL_META = REPO / "src" / "robomme" / "env_metadata"


def read_pins(path: Path = PIN) -> dict[str, str]:
    """sha256sum format -> {relative path: sha256}; raises ValueError on bad format."""
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or not rel:
            raise ValueError(f"malformed pin line: {line!r}")
        if rel in pins:
            raise ValueError(f"duplicate pin line: {rel}")
        pins[rel] = digest
    return pins


def pin_mismatches(root: Path, pins: dict[str, str]) -> list[str]:
    """Compare sha256 per file against the pin table; return the mismatch list (empty = all match)."""
    bad = []
    for rel, want in sorted(pins.items()):
        path = root / rel
        got = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if got != want:
            bad.append(f"{rel}: {got} ≠ {want}")
    return bad


def read_raw(tier: str) -> tuple[dict, list[dict]]:
    records = [json.loads(line) for line in (ROOT / tier / "specs.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    return records[0], records[1:]


@pytest.fixture(scope="module")
def raw():
    return {tier: read_raw(tier) for tier in NEW_TIERS}


# -- Byte pins -------------------------------------------------------------


def test_pin_table_covers_exactly_five_files():
    assert sorted(read_pins()) == sorted(f"{tier}/specs.jsonl" for tier in NEW_TIERS)
    assert sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()) == \
        sorted(f"{tier}/specs.jsonl" for tier in NEW_TIERS)


def test_packaged_bytes_equal_pins():
    assert pin_mismatches(ROOT, read_pins()) == []


def test_pin_check_negative(tmp_path):
    """Checker negatives: changing 1 byte in a copy, deleting a file, and a malformed pin table must all be detected."""
    pins = read_pins()
    for rel in pins:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((ROOT / rel).read_bytes())
    assert pin_mismatches(tmp_path, pins) == []
    target = tmp_path / "xhard5" / "specs.jsonl"
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0x01
    target.write_bytes(bytes(data))
    assert [m.split(":")[0] for m in pin_mismatches(tmp_path, pins)] == ["xhard5/specs.jsonl"]
    target.unlink()
    assert len(pin_mismatches(tmp_path, pins)) == 1
    bad = tmp_path / "bad.sha256"
    bad.write_text("abc  xhard1/specs.jsonl\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_pins(bad)


# -- Readers pass -----------------------------------------------------------


@pytest.mark.parametrize("tier", NEW_TIERS)
def test_load_specs_each_file(tier, raw):
    from robomme_ood.env_record_wrapper import hard_specs as hs

    header, rows = hs.load_specs(ROOT / tier / "specs.jsonl", check_fingerprint=False)
    assert header["schema"] == SPECS_SCHEMA and header["difficulty"] == tier
    assert len(rows) == PACKAGED_ROWS[tier]
    assert sum(bool(r["selected"]) for r in rows) == PACKAGED_SELECTED[tier]
    assert (header, rows) == raw[tier]  # the reader does not change content


def test_load_specs_root_whole():
    from robomme_ood.env_record_wrapper import hard_specs as hs

    loaded = hs.load_specs_root(ROOT, dict(V9_CELLS), check_fingerprint=False)
    assert tuple(loaded) == NEW_TIERS
    assert sum(len(rows) for _, rows in loaded.values()) == PACKAGED_ROWS_TOTAL
    delivered = {}
    for tier, (_, rows) in loaded.items():
        for row in rows:
            if hs.delivered(row):
                delivered[(row["task"], tier)] = delivered.get((row["task"], tier), 0) + 1
    assert delivered == V9_CELLS
    assert sum(delivered.values()) == TOTAL


# -- Per-row self-consistency ------------------------------------------------


def row_problems(header: dict, row: dict) -> list[str]:
    """Self-consistency problem list for one row (empty = consistent). Criteria in the module docstring."""
    spec = row.get("spec") or {}
    ident = spec.get("identity") or {}
    out = []
    if spec.get("task") != row.get("task") or ident.get("task") != row.get("task"):
        out.append("task")
    if ident.get("seed") != row.get("seed"):
        out.append("seed")
    if ident.get("difficulty") != row.get("tier") or row.get("tier") != header.get("difficulty"):
        out.append("difficulty")
    if ident.get("episode") != row.get("episode"):
        out.append("episode")
    if spec.get("spec_kind") != SPEC_KIND:
        out.append("spec_kind")
    exec_steps = (row.get("rollout") or {}).get("exec_steps")
    if exec_steps is not None and not (isinstance(exec_steps, int) and 0 <= exec_steps <= EXEC_CAP):
        out.append("exec_steps")
    return out


@pytest.mark.parametrize("tier", NEW_TIERS)
def test_rows_self_consistent(tier, raw):
    header, rows = raw[tier]
    assert set(header["sampling_config"]) == set(header["tasks"])
    bad = {f"{r['task']}#{r['candidate']}": p for r in rows if (p := row_problems(header, r))}
    assert bad == {}
    # delivered rows all carry execution steps
    assert all(isinstance((r["rollout"] or {}).get("exec_steps"), int)
               for r in rows if r["selected"] and (r["rollout"] or {}).get("status") == "ok")


def test_row_problems_negative(raw):
    """Checker negatives: corrupt one row item by item; each is named."""
    header, rows = raw["xhard5"]
    base = next(r for r in rows if r["selected"])
    assert row_problems(header, base) == []
    cases = {
        "task": lambda r: r["spec"].__setitem__("task", "PickXtimes"),
        "seed": lambda r: r["spec"]["identity"].__setitem__("seed", r["seed"] + 1),
        "difficulty": lambda r: r["spec"]["identity"].__setitem__("difficulty", "xhard4"),
        "episode": lambda r: r["spec"]["identity"].__setitem__("episode", r["episode"] + 1),
        "spec_kind": lambda r: r["spec"].__setitem__("spec_kind", "native-newvalue/1"),
        "exec_steps": lambda r: r["rollout"].__setitem__("exec_steps", EXEC_CAP + 1),
    }
    for name, mutate in cases.items():
        row = json.loads(json.dumps(base))
        mutate(row)
        assert row_problems(header, row) == [name], name


# -- Seed uniqueness and disjointness ----------------------------------------


def official_seeds() -> set[int]:
    seeds = set()
    for path in sorted(OFFICIAL_META.rglob("*.json")):
        for record in json.loads(path.read_text(encoding="utf-8"))["records"]:
            seeds.add(int(record["seed"]))
    return seeds


def test_seeds_globally_unique_and_disjoint_from_official(raw):
    seeds = [r["seed"] for _, rows in raw.values() for r in rows]
    assert len(seeds) == PACKAGED_ROWS_TOTAL
    assert all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)
    assert len(set(seeds)) == PACKAGED_ROWS_TOTAL
    official = official_seeds()
    assert official  # official metadata was actually read
    assert set(seeds) & official == set()
