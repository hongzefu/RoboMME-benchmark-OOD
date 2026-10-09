"""L1 契约：包内五份交付规格（``src/robomme_ood/env_metadata/ood/xhard{1..5}/specs.jsonl``）没被动过且逐行自洽。

- 文件字节 sha256 等于钉值表 ``tests/robomme_ood/contract/packaged_specs.sha256``（sha256sum 格式，钉值表进 git）；
- 逐份 ``load_specs``、整根 ``load_specs_root`` 通过；
- 逐行：``spec.task == row.task``，``spec.identity`` 的 task／seed／difficulty／episode 与行一致，
  header ``sampling_config`` 键集合等于 ``tasks``，带 ``exec_steps`` 的行不超过执行步上限；
- 全部 1518 行 seed 全局唯一，且与官方元数据（train／val／test）的 seed 不相交（hard 包已不带 train 元数据）。

规格行用标准库 json 直接读（不经被测的读取函数），被测函数只在「通过校验」的用例里调用。
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
    """sha256sum 格式 → {相对路径: sha256}；格式不对即抛 ValueError。"""
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or not rel:
            raise ValueError(f"钉值行格式不对：{line!r}")
        if rel in pins:
            raise ValueError(f"钉值行重复：{rel}")
        pins[rel] = digest
    return pins


def pin_mismatches(root: Path, pins: dict[str, str]) -> list[str]:
    """按钉值表逐文件比 sha256；返回不符清单（空 = 全部一致）。"""
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


# ── 字节钉值 ─────────────────────────────────────────────────────────────


def test_pin_table_covers_exactly_five_files():
    assert sorted(read_pins()) == sorted(f"{tier}/specs.jsonl" for tier in NEW_TIERS)
    assert sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()) == \
        sorted(f"{tier}/specs.jsonl" for tier in NEW_TIERS)


def test_packaged_bytes_equal_pins():
    assert pin_mismatches(ROOT, read_pins()) == []


def test_pin_check_negative(tmp_path):
    """判定器负例：副本改 1 字节、删一个文件、钉值表格式坏，都必须被发现。"""
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


# ── 读取函数通过 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("tier", NEW_TIERS)
def test_load_specs_each_file(tier, raw):
    from robomme_ood.env_record_wrapper import hard_specs as hs

    header, rows = hs.load_specs(ROOT / tier / "specs.jsonl", check_fingerprint=False)
    assert header["schema"] == SPECS_SCHEMA and header["difficulty"] == tier
    assert len(rows) == PACKAGED_ROWS[tier]
    assert sum(bool(r["selected"]) for r in rows) == PACKAGED_SELECTED[tier]
    assert (header, rows) == raw[tier]  # 读取函数不改内容


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


# ── 逐行自洽 ─────────────────────────────────────────────────────────────


def row_problems(header: dict, row: dict) -> list[str]:
    """一行的自洽问题清单（空 = 自洽）。判据见模块文档。"""
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
    # 交付行都带执行步
    assert all(isinstance((r["rollout"] or {}).get("exec_steps"), int)
               for r in rows if r["selected"] and (r["rollout"] or {}).get("status") == "ok")


def test_row_problems_negative(raw):
    """判定器负例：逐项改坏一行，都被点名。"""
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


# ── seed 唯一与不相交 ─────────────────────────────────────────────────────


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
    assert official  # 官方元数据确实读到了
    assert set(seeds) & official == set()
