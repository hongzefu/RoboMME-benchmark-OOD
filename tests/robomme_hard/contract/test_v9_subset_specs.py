"""L1 契约：``scripts/injection-dev/v9_subset_specs.py`` 现存的两个子命令 ``assemble``、``link``。

以旧 ``tests/lightweight/test_v9_subset_specs.py``（``86e5a015``）为蓝本重写（derive／extend／verify 已随 V8 表删除）。
纯 CPU 合成夹具，不碰真实 artifacts、不 reset：真实的 ``_freeze.freeze``（/4）封冻结根 → ``_rollout.split_v8`` 切片
→ ``_rollout.run_continue_v8``（替身 runner 在显式局目录写假 h5／mp4）→ 带结果段的规格根与 ``delivery.json``。

合成三个来源（格表都是 V9_CELLS 的子表）：

- 子集根：StopCube xhard1～5 各 2 局 + InsertPeg xhard4 3 局（V8 复用局）；
- MoveCube 片根：MoveCube xhard4 2 局（V9 新局）；
- InsertPeg 片根：子集根 xhard4 里 InsertPeg 的行原样抽出、单任务重签（本例不追加新局）。

整合后：2×5 + 3 + 2 = 15 局，复用 13、新 2、7 格。另补两条拒绝分支（输出根与子集根相同、输出已存在）与
sampling 不一致即 FAIL；link 幂等、目标存在且内容不同即 FAIL。
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest

from tests.robomme_hard._support.loaders import load_script
from tests.robomme_hard.contract.test_constants import NEW_TIERS

IP = ("InsertPeg", "xhard4")
MC = ("MoveCube", "xhard4")
SUBSET_CELLS = {**{("StopCube", t): 2 for t in NEW_TIERS}, IP: 3}
MC_CELLS = {MC: 2}
ASM_CELLS = {**SUBSET_CELLS, **MC_CELLS}


@pytest.fixture(scope="module")
def V():
    return load_script("injection-dev/v9_subset_specs.py")


@pytest.fixture(scope="module")
def R(V):
    return V._rollout


@pytest.fixture(scope="module")
def F(V):
    return importlib.import_module("_freeze")


def _drafts(H, task: str, tier: str, candidates: int) -> list[dict]:
    rule = H.seed_rule_for(tier, "v8")
    rows = []
    for episode in range(candidates):
        spec = {"spec_kind": "native-newvalue/2", "task": task, "tier": tier, "layout": {"x": episode}}
        if task == "MoveCube":
            spec["initializations"] = {"0": {"way_idx": 0}, "1": {"way_idx": episode % 3}}
        rows.append({"task": task, "difficulty": tier, "episode": episode, "attempt": 0,
                     "seed": H.seed_for(task, episode, 0, rule), "reset_ok": True, "spec": spec,
                     "spec_sha256": H.spec_sha256(spec)})
    return rows


def build_frozen(V, R, F, root: Path, cells: dict, spare: int, tag: str) -> Path:
    H = V.H
    for tier in R.cell_tiers(cells):
        tasks = [t for t in H.ALL_TASKS if (t, tier) in cells]
        cands = {t: cells[(t, tier)] + spare for t in tasks}
        drafts = [row for t in tasks for row in _drafts(H, t, tier, cands[t])]
        parts = {"difficulty": tier, "tasks": tasks, "seed_rule": H.seed_rule_for(tier, "v8"),
                 "sampling_config": {t: {"decision": {"k": t, "tier": tier, "tag": tag}, "native": {}} for t in tasks},
                 "recovery_rule": {"rule": "off"}, "identity_source": "formula", "run_id": f"t2-specs-{tier}",
                 "draw_stats": {}, "provenance": {}}
        select = {t: tuple(range(cells[(t, tier)])) for t in tasks}
        header, rows = F.freeze(drafts, parts, select, cands, schema=H.SCHEMA)
        F.write_jsonl_exclusive(root / tier / "specs.jsonl", [header, *rows])
    return root


class FakeRunner:
    """代替 ``run_batch``：在显式局目录写假 h5 与主视频；``fails`` 里的 (tier, task, candidate) 记规划失败。"""

    def __init__(self, R, out_dir: Path, fails=()):
        self.R, self.out_dir, self.fails = R, Path(out_dir), set(fails)
        self.ran: list[tuple] = []

    def __call__(self, batch, header, round_index):
        tier = header["difficulty"]
        out = []
        for row in batch:
            key = (tier, row["task"], row["candidate"])
            self.ran.append(key)
            ok = key not in self.fails
            record = {"task": row["task"], "tier": tier, "candidate": row["candidate"], "episode": row["episode"],
                      "seed": row["seed"], "attempt": row["attempt"], "spec_sha256": row["spec_sha256"], "ok": ok,
                      "error_type": None if ok else "ScrewPlanFailure", "error": None if ok else "规划失败",
                      "spec_binding": None, "env_package": "robomme_hard",
                      "env_module": f"robomme_hard.robomme_env.{row['task']}", "wrapper_modules": None, "h5": None,
                      "round": round_index, "role": row.get("_role", "selected")}
            if ok:
                wdir = self.R.episode_dir(self.out_dir, row)
                h5 = wdir / "hdf5_files" / f"{row['task']}_ep{row['episode']}_seed{row['seed']}.h5"
                h5.parent.mkdir(parents=True, exist_ok=True)
                payload = f"h5-{self.out_dir.name}-{tier}-{row['task']}-{row['candidate']}".encode()
                h5.write_bytes(payload)
                (wdir / "videos").mkdir(exist_ok=True)
                (wdir / "videos" / f"{row['task']}_ep{row['episode']}_seed{row['seed']}_demo.mp4").write_bytes(
                    b"mp4-" + payload)
                record.update(h5=str(h5), h5_sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                              frames=105, exec_steps=100)
            out.append(record)
        return out


def build_world(V, R, F, base: Path, cells: dict, tag: str, fails=()) -> Path:
    """冻结根 → 单片 → continue：返回片根（含 specs/<tier>/specs.jsonl 与 delivery.json）。"""
    frozen = build_frozen(V, R, F, base / "frozen", cells, spare=2, tag=tag)
    shard = base / "shard"
    R.split_v8(frozen, cells, shard, label="t2")
    summary = R.run_continue_v8(shard / "specs", cells, shard, src_root=Path("."), workers=1, gpu="0",
                                pkg="robomme_hard", code_baseline="test", batch_runner=FakeRunner(R, shard, fails))
    assert summary["delivery_set"].startswith("V8_DELIVERY_SET=PASS"), summary["delivery_set"]
    return shard


def read_file(path: Path) -> tuple[dict, list[dict]]:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[0], records[1:]


def write_file(path: Path, header: dict, rows: list[dict], H) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(H.canonical_json(r) + "\n" for r in [header, *rows]), encoding="utf-8")


def insertpeg_root(V, subset: Path, out: Path) -> Path:
    """从子集根 xhard4 抽出 InsertPeg 行做成单任务片根（行逐字不动，header 只留 InsertPeg 并重签）。"""
    H = V.H
    header, rows = read_file(subset / "specs" / "xhard4" / "specs.jsonl")
    task = IP[0]
    header = copy.deepcopy(header)
    header["tasks"] = [task]
    for key in ("per_env", "select_rule", "delivery_per_cell", "sampling_config"):
        header[key] = {task: header[key][task]}
    rows = [r for r in rows if r["task"] == task]
    header = V.resign(header, rows, H.V9_CELLS)
    write_file(out / "specs" / "xhard4" / "specs.jsonl", header, rows, H)
    (out / "delivery.json").write_bytes((subset / "delivery.json").read_bytes())
    return out


@pytest.fixture
def world(V, R, F, tmp_path):
    subset = build_world(V, R, F, tmp_path / "v8", SUBSET_CELLS, "v8", fails={("xhard4", "InsertPeg", 1)})
    mc = build_world(V, R, F, tmp_path / "mc", MC_CELLS, "v9-region")
    ip = insertpeg_root(V, subset, tmp_path / "ip")
    return {"subset": subset, "mc": mc, "ip": ip, "v8_delivery": subset / "delivery.json", "tmp": tmp_path}


def assemble(V, world, out: Path, **kw):
    return V.assemble(world["subset"], world["mc"], world["ip"], world["v8_delivery"], out / "specs-root",
                      out / "delivery" / "delivery.local.json", cells=dict(ASM_CELLS), **kw)


# ── assemble ───────────────────────────────────────────────────────────


def test_assemble_pass_and_sources(V, world):
    H = V.H
    out = world["tmp"] / "v9"
    result = assemble(V, world, out)
    assert not result["problems"], result["problems"]
    total = sum(ASM_CELLS.values())
    new = MC_CELLS[MC]
    assert result["line"] == f"V9_ASSEMBLE=PASS rows={total} reused={total - new} new={new} cells={len(ASM_CELLS)}"
    loaded = H.load_specs_root(out / "specs-root", ASM_CELLS, cell_table=H.V9_CELLS, check_fingerprint=False)
    assert list(loaded) == list(NEW_TIERS)
    header, rows = loaded["xhard4"]
    assert header["tasks"] == ["StopCube", "InsertPeg", "MoveCube"]
    assert header["sampling_config"]["MoveCube"]["decision"]["tag"] == "v9-region"
    for tier in ("xhard1", "xhard2", "xhard3", "xhard5"):  # 其余四档与子集根逐字节相同
        assert (out / "specs-root" / tier / "specs.jsonl").read_bytes() == \
            (world["subset"] / "specs" / tier / "specs.jsonl").read_bytes()
    data = json.loads((out / "delivery" / "delivery.local.json").read_text(encoding="utf-8"))
    assert data["schema"] == "v8-delivery/1" and len(data["rows"]) == total
    src = {(r["task"], r["tier"], r["candidate"]): r["source"] for r in data["rows"]}
    assert {s for (t, _, _), s in src.items() if t == "MoveCube"} == {"v9-new"}
    assert {s for (t, _, _), s in src.items() if t != "MoveCube"} == {"v8-reuse"}
    # InsertPeg 候选 1 规划失败 → 递补 3；交付 0、2、3
    assert sorted(c for (t, _, c) in src if t == "InsertPeg") == [0, 2, 3]
    assert data["line"].startswith(f"V9_DELIVERY_SET=PASS tasks=3 cells={len(ASM_CELLS)} total={total}")
    for row in data["rows"]:
        assert set(V.DELIVERY_ROW_KEYS) <= set(row)
        assert Path(row["h5"]).is_file() and Path(row["video"]).is_file()
        assert row["video_sha256"] == hashlib.sha256(Path(row["video"]).read_bytes()).hexdigest()
        assert (out / "delivery" / row["path"]).resolve() == Path(row["h5"]).resolve()


def test_assemble_rejects_out_specs_equal_subset(V, world):
    with pytest.raises(V.SubsetError, match="不能与 --subset 相同"):
        V.assemble(world["subset"], world["mc"], world["ip"], world["v8_delivery"], world["subset"] / "specs",
                   world["tmp"] / "d.json", cells=dict(ASM_CELLS))


def test_assemble_rejects_existing_output(V, world):
    out = world["tmp"] / "v9"
    target = out / "delivery" / "delivery.local.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(V.SubsetError, match="拒绝覆盖"):
        assemble(V, world, out)
    assert target.read_text(encoding="utf-8") == "{}"
    assert not (out / "specs-root").exists()


def test_assemble_rejects_wrong_movecube_root_tasks(V, world):
    """MoveCube 根里混进别的任务即拒绝（子集根本身当 MoveCube 根传）。"""
    with pytest.raises(V.SubsetError, match="MoveCube 根"):
        V.assemble(world["subset"], world["subset"], world["ip"], world["v8_delivery"], world["tmp"] / "o" / "s",
                   world["tmp"] / "o" / "d.json", cells=dict(ASM_CELLS))


def test_assemble_fails_on_sampling_mismatch_and_writes_nothing(V, world):
    H = V.H
    path = world["ip"] / "specs" / "xhard4" / "specs.jsonl"
    header, rows = read_file(path)
    header["sampling_config"]["InsertPeg"]["decision"]["tag"] = "drifted"
    write_file(path, V.resign(header, rows, H.V9_CELLS), rows, H)
    out = world["tmp"] / "v9b"
    result = assemble(V, world, out)
    assert result["line"] is None and any("sampling_config" in p for p in result["problems"])
    assert not out.exists()


def test_assemble_fails_when_v8_delivery_disagrees(V, world):
    """V8 清单里一个复用局的 h5_sha256 与规格结果段不符 → FAIL，不写交付清单。"""
    data = json.loads(world["v8_delivery"].read_text(encoding="utf-8"))
    row = next(r for r in data["rows"] if r["task"] == "StopCube")
    row["h5_sha256"] = "0" * 64
    bad = world["tmp"] / "bad-delivery.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    out = world["tmp"] / "v9c"
    result = V.assemble(world["subset"], world["mc"], world["ip"], bad, out / "specs-root",
                        out / "delivery" / "delivery.local.json", cells=dict(ASM_CELLS))
    assert result["line"].startswith("V9_ASSEMBLE=FAIL") and any("h5_sha256" in p for p in result["problems"])
    assert not (out / "delivery" / "delivery.local.json").exists()


# ── link ───────────────────────────────────────────────────────────────


def test_link_idempotent_and_refuses_overwrite(V, world, capsys):
    out = world["tmp"] / "v9"
    assert not assemble(V, world, out)["problems"]
    delivery = out / "delivery" / "delivery.local.json"
    tree = out / "delivery"
    n_rows = sum(ASM_CELLS.values())
    assert V.main(["link", "--delivery", str(delivery), "--out", str(tree)]) == 0
    assert f"V9_LINK=PASS rows={n_rows} files={2 * n_rows} linked={2 * n_rows} skipped=0" in capsys.readouterr().out
    data = json.loads(delivery.read_text(encoding="utf-8"))
    for row in data["rows"]:
        wdir = tree / "episodes" / row["tier"] / f"{row['task']}_episode_{row['candidate']}"
        assert os.path.samefile(wdir / "hdf5_files" / Path(row["h5"]).name, row["h5"])
        assert os.path.samefile(wdir / "videos" / Path(row["video"]).name, row["video"])
    assert V.main(["link", "--delivery", str(delivery), "--out", str(tree)]) == 0
    assert f"linked=0 skipped={2 * n_rows}" in capsys.readouterr().out
    row = data["rows"][0]
    target = tree / "episodes" / row["tier"] / f"{row['task']}_episode_{row['candidate']}" / "hdf5_files" / \
        Path(row["h5"]).name
    target.unlink()
    target.write_bytes(b"not the same h5")
    assert V.main(["link", "--delivery", str(delivery), "--out", str(tree)]) == 1
    text = capsys.readouterr().out
    assert "V9_LINK=FAIL" in text and "不覆盖" in text
    assert target.read_bytes() == b"not the same h5"


def test_link_refuses_out_inside_source_shard(V, world, capsys):
    out = world["tmp"] / "v9"
    assert not assemble(V, world, out)["problems"]
    delivery = out / "delivery" / "delivery.local.json"
    assert V.main(["link", "--delivery", str(delivery), "--out", str(world["mc"] / "tree")]) == 1
    assert "重叠" in capsys.readouterr().out
