"""L1 契约：固定检查集 ``scripts/parity/gate_set.py``（V9 43 格 × 3 = 129；xhard0 16 任务 × 1 档 × 3 = 48）。

以旧 ``tests/lightweight/test_gate_set.py``（``86e5a015``）为蓝本重写：只读包内 V9 规格与仓库内两份冻结文件，
不读 ``artifacts/``、不起仿真（builder 只构造、读元数据，不 ``gym.make``）。补 F-7 负例：零字节、只有空白、
坏 JSON 的冻结文件都必须判 FAIL。
"""
from __future__ import annotations

import json

import pytest

from tests.robomme_hard._support.loaders import REPO, load_script
from tests.robomme_hard.contract.test_constants import (
    GATE_V9_PER_CELL,
    GATE_V9_SCHEMA,
    GATE_V9_TOTAL,
    GATE_X0_PER_TASK,
    GATE_X0_SCHEMA,
    GATE_X0_TOTAL,
    N_CELLS,
    N_TASKS,
    PER_TASK,
    TASKS,
    XHARD0,
    XHARD0_EPISODES,
)

FROZEN = REPO / "scripts" / "configs" / "gate-set-v9-129.json"
FROZEN_X0 = REPO / "scripts" / "configs" / "gate-set-xhard0-48.json"
X0_MANIFEST = REPO / "scripts" / "configs" / "xhard0" / "xhard0_manifest.json"


@pytest.fixture(scope="module")
def G():
    from scripts.parity import gate_set

    return gate_set


@pytest.fixture(scope="module")
def rows(G):
    return G.build_gate_set()


@pytest.fixture(scope="module")
def index(G):
    return G._delivery_index(str(G._hs().PACKAGED_SPECS_ROOT))


@pytest.fixture(scope="module")
def x0_rows(G):
    return G.load_xhard0_set(FROZEN_X0)


# ── V9 ─────────────────────────────────────────────────────────────────


def test_production_scale_constants(G):
    assert G.PER_CELL == GATE_V9_PER_CELL and G.XHARD0_PER_TASK == GATE_X0_PER_TASK
    assert G.SCHEMA == GATE_V9_SCHEMA and G.XHARD0_SCHEMA == GATE_X0_SCHEMA
    assert G.N_TASKS == N_TASKS


def test_build_is_deterministic(G, rows):
    assert G.build_gate_set() == rows
    assert G.build_payload() == G.build_payload()


def test_scale_43_cells_times_3(G, rows, index):
    table = G.allocation_table(rows)
    cells = {(t, tier) for (t, tier, _s) in index}
    assert len(cells) == N_CELLS
    assert {(t, tier) for t, cell in table.items() for tier in cell} == cells
    assert all(n == GATE_V9_PER_CELL for cell in table.values() for n in cell.values())
    assert len(rows) == GATE_V9_TOTAL and set(table) == set(TASKS)
    assert all(set(r) == set(G.ROW_KEYS) for r in rows)
    assert len({(r["task"], r["tier"], r["seed"]) for r in rows}) == GATE_V9_TOTAL


def test_rows_ordered_by_canonical_task_order(rows):
    order = {t: i for i, t in enumerate(TASKS)}
    keys = [(order[r["task"]], r["tier"], r["candidate"]) for r in rows]
    assert keys == sorted(keys)


def test_each_row_is_a_delivered_row(G, rows, index):
    hs = G._hs()
    for r in rows:
        hit = index[(r["task"], r["tier"], r["seed"])]
        assert hs.delivered(hit["row"])
        assert int(hit["row"]["candidate"]) == r["candidate"]
        assert hit["row"]["spec_sha256"] == r["spec_sha256"]
        assert hit["builder_episode"] == r["builder_episode"]
        assert 0 <= r["builder_episode"] < PER_TASK


def test_each_cell_takes_smallest_candidates(rows, index):
    for (task, tier) in {(r["task"], r["tier"]) for r in rows}:
        pool = sorted(int(h["row"]["candidate"]) for (t, tr, _s), h in index.items() if t == task and tr == tier)
        got = [r["candidate"] for r in rows if r["task"] == task and r["tier"] == tier]
        assert got == pool[:GATE_V9_PER_CELL], (task, tier)


def test_builder_episode_equals_real_builder(rows):
    """真 builder（ood，开关为 0）逐行核对 builder_episode → 身份；只构造、读元数据，不建环境。"""
    from robomme_hard.env_record_wrapper.hard_builder import BenchmarkEnvBuilder

    builders = {}
    for r in rows:
        builder = builders.get(r["task"])
        if builder is None:
            builder = builders[r["task"]] = BenchmarkEnvBuilder(env_id=r["task"], dataset="ood")
            assert builder.get_episode_num() == PER_TASK
        ident = builder.resolve_identity(r["builder_episode"])
        assert (ident["tier"], ident["candidate"], ident["seed"], ident["spec_sha256"]) == \
            (r["tier"], r["candidate"], r["seed"], r["spec_sha256"]), r


def test_frozen_file_equals_rebuild_and_check_passes(G, rows):
    assert FROZEN.read_text(encoding="utf-8") == G.dumps_payload(G.build_payload())
    assert G.load_gate_set(FROZEN) == rows
    assert json.loads(FROZEN.read_text(encoding="utf-8"))["schema"] == GATE_V9_SCHEMA
    ok, line = G.check(FROZEN)
    assert ok, line
    assert line.startswith(f"GATE_SET=PASS cells={N_CELLS} per_cell={GATE_V9_PER_CELL} total={GATE_V9_TOTAL} "
                           f"in_delivery={GATE_V9_TOTAL} sha=")


def test_tampered_frozen_value_fails(G, tmp_path):
    obj = json.loads(FROZEN.read_text(encoding="utf-8"))
    obj["rows"][5]["seed"] += 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(G.GateSetError):
        G.load_gate_set(bad)
    ok, line = G.check(bad)
    assert not ok and line.startswith("GATE_SET=FAIL")


def test_resigned_but_swapped_rows_fail_check(G, tmp_path):
    """两行交换顺序并重签（sha 自洽、总数不变）：逐字节比与重算结果不同 → FAIL。"""
    obj = json.loads(FROZEN.read_text(encoding="utf-8"))
    obj["rows"][0], obj["rows"][-1] = obj["rows"][-1], obj["rows"][0]
    obj["sha256"] = G.payload_sha256(obj)
    bad = tmp_path / "swapped.json"
    bad.write_text(G.dumps_payload(obj), encoding="utf-8")
    assert len(G.load_gate_set(bad)) == GATE_V9_TOTAL
    ok, line = G.check(bad)
    assert not ok and line.startswith("GATE_SET=FAIL")


@pytest.mark.parametrize("content", [b"", b"  \n", b"{not json", b"[]"])
def test_f7_empty_or_broken_frozen_file_fails(G, tmp_path, content):
    path = tmp_path / "frozen.json"
    path.write_bytes(content)
    ok, line = G.check(path)
    assert not ok and line.startswith("GATE_SET=FAIL")


def test_f7_missing_frozen_file_fails(G, tmp_path):
    ok, line = G.check(tmp_path / "missing.json")
    assert not ok and line.startswith("GATE_SET=FAIL")


def test_exports(G, rows, tmp_path):
    from scripts.parity import hard_parity

    gen = G.to_generate_identities(rows)
    assert gen[0] == {"task": rows[0]["task"], "tier": rows[0]["tier"], "seed": rows[0]["seed"]}
    dig = G.to_env_digest_identities(rows)
    assert all(d["source_episode"] == -1 and set(d) == {"task", "source_episode", "seed", "builder_episode"} for d in dig)
    assert len(G.to_eval_keys(rows)) == GATE_V9_TOTAL
    with pytest.raises(G.GateSetError):
        G.to_legacy_identities(rows)
    path = G.write_generate_identities(rows, tmp_path / "g9.jsonl")
    assert hard_parity.read_identity_subset(path) == {(r["task"], r["tier"], r["seed"]) for r in rows}
    with pytest.raises(G.GateSetError):
        G.write_generate_identities(rows, tmp_path / "g9.json")


def test_switch_on_refuses_build(G, monkeypatch):
    monkeypatch.setenv(G.XHARD0_ENV, "1")
    with pytest.raises(G.GateSetError):
        G.build_gate_set()


def test_cli_check_and_export(G, tmp_path, capsys):
    assert G.main(["check", "--path", str(FROZEN)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("GATE_SET=PASS")
    out = tmp_path / "g9.jsonl"
    assert G.main(["export", "--set", "v9", "--kind", "generate", "--out", str(out)]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == GATE_V9_TOTAL


# ── xhard0 ─────────────────────────────────────────────────────────────


def test_xhard0_frozen_self_consistent(G, x0_rows):
    obj = json.loads(FROZEN_X0.read_text(encoding="utf-8"))
    assert obj["schema"] == GATE_X0_SCHEMA
    assert obj["sha256"] == G.payload_sha256(obj)
    assert FROZEN_X0.read_text(encoding="utf-8") == G.dumps_payload(obj)
    assert len(x0_rows) == GATE_X0_TOTAL
    counts: dict[str, int] = {}
    for r in x0_rows:
        assert set(r) == set(G.XHARD0_ROW_KEYS) and r["tier"] == XHARD0
        assert r["source_episode"] in XHARD0_EPISODES
        counts[r["task"]] = counts.get(r["task"], 0) + 1
    assert counts == {task: GATE_X0_PER_TASK for task in TASKS}
    order = {t: i for i, t in enumerate(TASKS)}
    keys = [(order[r["task"]], r["source_episode"]) for r in x0_rows]
    assert keys == sorted(keys)
    ok, line = G.check_xhard0(FROZEN_X0)
    assert ok, line
    assert line.startswith(f"GATE_SET_XHARD0=PASS tasks={N_TASKS} per_task={GATE_X0_PER_TASK} total={GATE_X0_TOTAL}")


def test_xhard0_rows_in_official_manifest(x0_rows):
    manifest = json.loads(X0_MANIFEST.read_text(encoding="utf-8"))["rows"]
    official = {(r["task"], int(r["episode"]), int(r["seed"])) for r in manifest}
    assert all((r["task"], r["source_episode"], r["seed"]) in official for r in x0_rows)


def test_xhard0_builder_episode_matches_switch_on_builder(x0_rows, monkeypatch):
    """开关打开时 ood builder 前 12 局是 xhard0：冻结行的 builder_episode 解析回同一身份。"""
    from robomme_hard.env_record_wrapper import hard_specs
    from robomme_hard.env_record_wrapper.hard_builder import BenchmarkEnvBuilder

    monkeypatch.setattr(hard_specs, "XHARD0_IN_TEST_HARD", True)
    builders = {}
    for r in x0_rows:
        builder = builders.setdefault(r["task"], BenchmarkEnvBuilder(env_id=r["task"], dataset="ood"))
        ident = builder.resolve_identity(r["builder_episode"])
        assert (ident["tier"], ident["seed"], ident["source_episode"]) == (XHARD0, r["seed"], r["source_episode"])


def test_xhard0_tamper_fails(G, tmp_path):
    obj = json.loads(FROZEN_X0.read_text(encoding="utf-8"))
    obj["rows"][7]["seed"] += 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(G.GateSetError):
        G.load_xhard0_set(bad)
    ok, line = G.check_xhard0(bad)
    assert not ok and line.startswith("GATE_SET_XHARD0=FAIL")


@pytest.mark.parametrize("content", [b"", b"{not json"])
def test_xhard0_f7_empty_or_broken_fails(G, tmp_path, content):
    path = tmp_path / "x0.json"
    path.write_bytes(content)
    ok, line = G.check_xhard0(path)
    assert not ok and line.startswith("GATE_SET_XHARD0=FAIL")


def test_xhard0_build_from_source_format(G, x0_rows, tmp_path):
    src = [{"builder_episode": r["builder_episode"], "e0_framesamp_modul_order": 0, "e0_shard": i, "e0_smvla_order": 0,
            "seed": r["seed"], "source_episode": r["source_episode"], "task": r["task"]}
           for i, r in enumerate(reversed(x0_rows))]
    path = tmp_path / "identities-small48.json"
    path.write_text(json.dumps(src), encoding="utf-8")
    assert G.build_xhard0_set(path) == x0_rows
    out = tmp_path / "x0.json"
    assert G.main(["build-xhard0", "--source", str(path), "--out", str(out)]) == 0
    ok, _ = G.check_xhard0(out, path)
    assert ok
    path.write_text(json.dumps([s for s in src if s["task"] != TASKS[-1]]), encoding="utf-8")
    with pytest.raises(G.GateSetError):
        G.build_xhard0_set(path)
    broken = [dict(s) for s in src]
    broken[0]["builder_episode"] += 1  # builder_episode 与 source_episode 不自洽
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(G.GateSetError):
        G.build_xhard0_set(path)


def test_xhard0_exports(G, x0_rows, tmp_path):
    legacy = G.to_legacy_identities(x0_rows)
    assert legacy == [{"task": r["task"], "source_episode": r["source_episode"], "seed": r["seed"],
                       "builder_episode": r["builder_episode"]} for r in x0_rows]
    out = tmp_path / "legacy.json"
    assert G.main(["export", "--set", "xhard0", "--kind", "legacy", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == legacy
    assert G.load_gate_set(FROZEN_X0) == x0_rows  # gate-set-xhard0 schema 转交
