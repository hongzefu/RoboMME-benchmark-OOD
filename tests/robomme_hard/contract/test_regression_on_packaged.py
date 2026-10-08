"""L1 契约：对包内真实规格进程内跑 ``scripts/parity/hard_regression.py`` 的四个纯 CPU 子命令。

- ``delivery-set``：V9 交付形态、seed 按档隔离、布局独立三行全 PASS（43 格、800 局）；
- ``tier-values``：14 任务 41 格逐局取值等于表 1；
- ``movecube-layout``：MoveCube xhard4 50 局 × 2 段 × 3 物体 = 300 点全在新区域内、运动方式 17／17／16；
- ``step-headroom``：交付清单由包内 800 个交付行组成；逐局 h5 的步数由 ``_h5_steps`` 的替身按该行
  ``rollout.frames``／``exec_steps`` 给出（800 个真实 h5 不在仓库里；h5 读取本身由对拍块 T6 测），
  判定器本身照常走：全部 ≤ 1600、候选池（包内规格根）里 exec_over_cap 为 0 与清单计数一致。

每个判定都配负例：在 tmp 副本或替身数据上造一处错，对应判定行必须 FAIL。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.robomme_hard._support.loaders import REPO, load_script
from tests.robomme_hard.contract.test_constants import (
    EXEC_CAP,
    MOVECUBE_POINTS,
    MOVECUBE_WAYS,
    N_CELLS,
    NEW_TIERS,
    PACKAGED_EXEC_OVER_CAP,
    TOTAL,
    V9_CELLS,
    XHARD0,
    XHARD4_ONLY,
)

ROOT = REPO / "src" / "robomme_hard" / "env_metadata" / "ood"


@pytest.fixture(scope="module")
def hr():
    return load_script("parity/hard_regression.py")


def run(hr, argv, capsys) -> tuple[int, dict[str, str]]:
    rc = hr.main(argv)
    out = capsys.readouterr().out
    lines = {}
    for line in out.splitlines():
        name, sep, _ = line.partition("=")
        if sep and name.isupper():
            lines[name] = line
    return rc, lines


def fields(line: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in line.split() if "=" in item)


def copy_root(tmp_path: Path) -> Path:
    dst = tmp_path / "root"
    shutil.copytree(ROOT, dst)
    return dst


def rewrite_rows(path: Path, mutate) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    mutate(records[0], records[1:])
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def resign(path: Path) -> None:
    from robomme_hard.env_record_wrapper import hard_specs as hs

    def _do(header, rows):
        header["sampling_config_sha256"] = hs.digest(header["sampling_config"])
        header["identity_sha256"] = hs.identity_sha256(header, rows)
        header["delivery_sha256"] = hs.delivery_sha256(rows)

    rewrite_rows(path, _do)


# ── delivery-set ───────────────────────────────────────────────────────


def test_delivery_set_pass_on_packaged(hr, capsys):
    rc, lines = run(hr, ["delivery-set", "--specs-root", str(ROOT), "--cells", "full"], capsys)
    assert rc == 0
    f = fields(lines["V9_DELIVERY_SET"])
    assert lines["V9_DELIVERY_SET"].startswith("V9_DELIVERY_SET=PASS")
    assert (int(f["cells"]), int(f["total"]), int(f["expected_total"])) == (N_CELLS, TOTAL, TOTAL)
    assert int(f["cell_mismatch"]) == 0 and int(f["load_errors"]) == 0
    assert lines["V9_SEED_DISJOINT"].startswith("V9_SEED_DISJOINT=PASS") and fields(lines["V9_SEED_DISJOINT"])["shared"] == "0"
    assert lines["V9_LAYOUT_INDEPENDENT"].startswith("V9_LAYOUT_INDEPENDENT=PASS")
    assert fields(lines["V9_LAYOUT_INDEPENDENT"])["layout_equal_pairs"] == "0"


def test_delivery_set_fails_when_one_delivered_row_dropped(hr, capsys, tmp_path):
    """负例：xhard5 一个交付行改成 selected=False 并重签（规格自身仍合法），逐格计数对不上格表 → FAIL。"""
    root = copy_root(tmp_path)

    def drop(header, rows):
        row = next(r for r in rows if r["selected"])
        row["selected"] = False

    rewrite_rows(root / "xhard5" / "specs.jsonl", drop)
    resign(root / "xhard5" / "specs.jsonl")
    rc, lines = run(hr, ["delivery-set", "--specs-root", str(root)], capsys)
    assert rc == 1 and lines["V9_DELIVERY_SET"].startswith("V9_DELIVERY_SET=FAIL")
    assert int(fields(lines["V9_DELIVERY_SET"])["total"]) == TOTAL - 1


def test_delivery_set_fails_on_missing_tier_file(hr, capsys, tmp_path):
    root = copy_root(tmp_path)
    (root / "xhard3" / "specs.jsonl").unlink()
    rc, lines = run(hr, ["delivery-set", "--specs-root", str(root)], capsys)
    assert rc == 1
    assert fields(lines["V9_DELIVERY_SET"])["missing_files"] == "1"


# ── tier-values ────────────────────────────────────────────────────────


def test_tier_values_pass_on_packaged(hr, capsys):
    rc, lines = run(hr, ["tier-values", "--specs-root", str(ROOT)], capsys)
    assert rc == 0 and lines["V8_TIER_VALUES"].startswith("V8_TIER_VALUES=PASS")
    f = fields(lines["V8_TIER_VALUES"])
    assert f["mismatches"] == "0" and f["missing_files"] == "0"
    # 逐局核对的行数 = 交付格里有取值维度的局数（800 − MoveCube 50 − InsertPeg 50）
    valued = sum(n for (task, _), n in V9_CELLS.items() if task not in XHARD4_ONLY)
    assert int(f["rows"]) == valued


def test_tier_values_fail_on_changed_value(hr, capsys, tmp_path):
    root = copy_root(tmp_path)

    def bump(header, rows):
        row = next(r for r in rows if r["task"] == "StopCube" and r["selected"])
        row["spec"]["actions"]["stop_time"] += 1

    rewrite_rows(root / "xhard5" / "specs.jsonl", bump)
    rc, lines = run(hr, ["tier-values", "--specs-root", str(root)], capsys)
    assert rc == 1 and fields(lines["V8_TIER_VALUES"])["mismatches"] == "1"


# ── movecube-layout ────────────────────────────────────────────────────


def test_movecube_layout_pass_on_packaged(hr, capsys):
    rc, lines = run(hr, ["movecube-layout", "--specs-root", str(ROOT)], capsys)
    assert rc == 0
    layout = fields(lines["V9_MOVECUBE_LAYOUT"])
    assert lines["V9_MOVECUBE_LAYOUT"].startswith("V9_MOVECUBE_LAYOUT=PASS")
    assert int(layout["episodes"]) == V9_CELLS[("MoveCube", "xhard4")]
    assert int(layout["points"]) == int(layout["in_region"]) == MOVECUBE_POINTS
    assert layout["region_mismatch"] == "0" and layout["bad_rows"] == "0"
    ways = fields(lines["V9_MOVECUBE_WAYS"])
    assert lines["V9_MOVECUBE_WAYS"].startswith("V9_MOVECUBE_WAYS=PASS")
    assert ways["ways"] == "/".join(str(MOVECUBE_WAYS[k]) for k in sorted(MOVECUBE_WAYS))


def test_movecube_layout_fail_on_point_outside_region(hr, capsys, tmp_path):
    """负例：一局 execution 段 goal 挪到远处（重签，规格仍合法）→ 该点不在 U 内 → FAIL。"""
    from robomme_hard.env_record_wrapper import hard_specs as hs

    root = copy_root(tmp_path)

    def move(header, rows):
        row = next(r for r in rows if r["task"] == "MoveCube" and r["selected"])
        row["spec"]["layout"]["execution"]["goal_xy"] = [5.0, 5.0]
        row["spec_sha256"] = hs.spec_sha256(row["spec"])

    path = root / "xhard4" / "specs.jsonl"
    rewrite_rows(path, move)
    resign(path)
    rc, lines = run(hr, ["movecube-layout", "--specs-root", str(root)], capsys)
    assert rc == 1 and lines["V9_MOVECUBE_LAYOUT"].startswith("V9_MOVECUBE_LAYOUT=FAIL")
    assert int(fields(lines["V9_MOVECUBE_LAYOUT"])["in_region"]) == MOVECUBE_POINTS - 1


# ── step-headroom ──────────────────────────────────────────────────────


def packaged_delivery(base: Path) -> tuple[Path, dict[str, tuple[int, int]]]:
    """包内 800 个交付行 → v8-delivery/1 清单（行 path 指向占位文件名）与 {路径: (演示帧, 执行步)}。"""
    rows, steps = [], {}
    for tier in NEW_TIERS:
        lines = (ROOT / tier / "specs.jsonl").read_text(encoding="utf-8").splitlines()
        for row in (json.loads(line) for line in lines[1:]):
            rollout = row["rollout"] or {}
            if not (row["selected"] and rollout.get("status") == "ok"):
                continue
            name = f"{tier}/{row['task']}_c{row['candidate']}.h5"
            rows.append({"task": row["task"], "tier": tier, "episode": row["episode"], "seed": row["seed"],
                         "candidate": row["candidate"], "path": name, "exec_steps": rollout["exec_steps"],
                         "frames": rollout["frames"]})
            steps[str(base / name)] = (rollout["frames"] - rollout["exec_steps"], rollout["exec_steps"])
    payload = {"schema": "v8-delivery/1", "rows": rows,
               "counts": {"exec_over_cap": PACKAGED_EXEC_OVER_CAP, "backfills": 0, "infra_retries": 0, "failed": 0}}
    path = base / "delivery.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, steps


def xhard0_source(base: Path, steps: dict) -> Path:
    path = base / "x0" / "identities.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"tier": XHARD0, "path": "x0.h5", "success": True}) + "\n", encoding="utf-8")
    steps[str(path.parent / "x0.h5")] = (0, 1)  # xhard0 上限待定（Q16），给一个任何上限都放行的步数
    return path


def patch_steps(hr, monkeypatch, steps):
    def fake(path):
        return steps[str(path)]

    monkeypatch.setattr(hr, "_h5_steps", fake)


def test_step_headroom_pass_on_packaged(hr, capsys, tmp_path, monkeypatch):
    delivery, steps = packaged_delivery(tmp_path)
    x0 = xhard0_source(tmp_path, steps)
    patch_steps(hr, monkeypatch, steps)
    rc, lines = run(hr, ["step-headroom", "--delivery", str(delivery), "--pool", str(ROOT), "--xhard0", str(x0)],
                    capsys)
    assert rc == 0, lines
    f = fields(lines["V9_STEP_CAP"])
    assert lines["V9_STEP_CAP"].startswith("V9_STEP_CAP=PASS")
    assert int(f["rows"]) == TOTAL and int(f["cells"]) == N_CELLS
    assert int(f["max"]) <= EXEC_CAP and int(f["cap"]) == EXEC_CAP
    assert int(f["filtered"]) == PACKAGED_EXEC_OVER_CAP and f["over"] == "0"


def test_step_headroom_negatives(hr, capsys, tmp_path, monkeypatch):
    delivery, steps = packaged_delivery(tmp_path)
    x0 = xhard0_source(tmp_path, steps)
    patch_steps(hr, monkeypatch, steps)
    payload = json.loads(delivery.read_text())
    # 一局执行步超上限（清单与 h5 同时改）
    over = dict(payload, rows=[dict(payload["rows"][0], exec_steps=EXEC_CAP + 1), *payload["rows"][1:]])
    first = str(tmp_path / payload["rows"][0]["path"])
    steps_over = dict(steps, **{first: (steps[first][0], EXEC_CAP + 1)})
    (tmp_path / "over.json").write_text(json.dumps(over))
    patch_steps(hr, monkeypatch, steps_over)
    rc, lines = run(hr, ["step-headroom", "--delivery", str(tmp_path / "over.json"), "--pool", str(ROOT),
                         "--xhard0", str(x0)], capsys)
    assert rc == 1 and fields(lines["V9_STEP_CAP"])["over"] == "1"
    patch_steps(hr, monkeypatch, steps)
    # 计数键缺一个
    absent = dict(payload, counts={k: v for k, v in payload["counts"].items() if k != "backfills"})
    (tmp_path / "absent.json").write_text(json.dumps(absent))
    rc, lines = run(hr, ["step-headroom", "--delivery", str(tmp_path / "absent.json"), "--pool", str(ROOT),
                         "--xhard0", str(x0)], capsys)
    assert rc == 1 and fields(lines["V9_STEP_CAP"])["count_keys_absent"] == "1"
    # 清单报的 exec_over_cap 与候选池实数不符
    wrong = dict(payload, counts=dict(payload["counts"], exec_over_cap=PACKAGED_EXEC_OVER_CAP + 1))
    (tmp_path / "wrong.json").write_text(json.dumps(wrong))
    rc, lines = run(hr, ["step-headroom", "--delivery", str(tmp_path / "wrong.json"), "--pool", str(ROOT),
                         "--xhard0", str(x0)], capsys)
    assert rc == 1 and fields(lines["V9_STEP_CAP"])["filtered_mismatch"] == "1"
    # --skip-xhard0 只给 INFO，不出 PASS
    rc, lines = run(hr, ["step-headroom", "--delivery", str(delivery), "--pool", str(ROOT), "--skip-xhard0"], capsys)
    assert rc == 0 and lines["V9_STEP_CAP"].startswith("V9_STEP_CAP=INFO")


def test_step_headroom_fails_when_h5_steps_disagree_with_delivery(hr, capsys, tmp_path, monkeypatch):
    """负例：清单不动、h5 替身给出的执行步比清单 ``exec_steps`` 少 1（仍在上限内）→ ``exec_steps_mismatch=1`` 判 FAIL。"""
    delivery, steps = packaged_delivery(tmp_path)
    x0 = xhard0_source(tmp_path, steps)
    first = str(tmp_path / json.loads(delivery.read_text())["rows"][0]["path"])
    demo, exec_steps = steps[first]
    patch_steps(hr, monkeypatch, dict(steps, **{first: (demo, exec_steps - 1)}))
    rc, lines = run(hr, ["step-headroom", "--delivery", str(delivery), "--pool", str(ROOT), "--xhard0", str(x0)],
                    capsys)
    f = fields(lines["V9_STEP_CAP"])
    assert rc == 1 and lines["V9_STEP_CAP"].startswith("V9_STEP_CAP=FAIL")
    assert f["exec_steps_mismatch"] == "1" and f["missing_h5"] == "0" and f["over"] == "0"


def test_step_headroom_counts_unreadable_h5_as_missing(hr, capsys, tmp_path, monkeypatch):
    """负例：读某一局 h5 时替身抛异常（缺文件、打不开）→ 该局不计入、``missing_h5=1`` 判 FAIL。"""
    delivery, steps = packaged_delivery(tmp_path)
    x0 = xhard0_source(tmp_path, steps)
    first = str(tmp_path / json.loads(delivery.read_text())["rows"][0]["path"])

    def fake(path):
        if str(path) == first:
            raise OSError(f"打不开 {path}")
        return steps[str(path)]

    monkeypatch.setattr(hr, "_h5_steps", fake)
    rc, lines = run(hr, ["step-headroom", "--delivery", str(delivery), "--pool", str(ROOT), "--xhard0", str(x0)],
                    capsys)
    f = fields(lines["V9_STEP_CAP"])
    assert rc == 1 and lines["V9_STEP_CAP"].startswith("V9_STEP_CAP=FAIL")
    assert f["missing_h5"] == "1" and f["exec_steps_mismatch"] == "0"
