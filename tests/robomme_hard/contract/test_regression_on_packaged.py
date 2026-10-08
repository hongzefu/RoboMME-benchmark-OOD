"""L1 契约：对包内真实规格跑三项数据检查（交付集、档位取值、MoveCube 布局），各配负例。

旧仓这里调用 ``scripts/parity/hard_regression.py`` 的 ``delivery-set``／``tier-values``／``movecube-layout``
子命令；该工具随私有评估仓走，本仓改用测试侧等价判定 ``packaged_checks``（只用标准库 json 读规格行，判据数值与
``fd0017d6`` 的工具实现一致）。``step-headroom`` 需要逐局 h5 与交付清单，属工具行为，不在本仓。

- 交付集：V9 43 格逐格交付行（selected 且 rollout ok）数等于格表、格表外无交付、selected 而未成功为 0、
  交付行执行步 ≤ ``EXEC_CAP``，合计 800 局；同任务跨档 seed 不交；布局独立（header ``layout_rule``、
  ``layout_parent`` 全空、跨档交付行位置叶子不照抄）；
- 档位取值：14 任务 41 格逐局取值等于 ``test_tier_table.TABLE``（行数 = 800 − MoveCube 50 − InsertPeg 50）；
- MoveCube 布局：xhard4 50 局 × 2 段 × 3 物体 = 300 点全在 V9 区域内、至少一点在 V8 旧区域外、段内 region
  与源码和 V9 三个数一致、运动方式 17／17／16。

负例都在 tmp 副本或内存副本上造一处错，对应计数必须变化。
"""
from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import pytest

from tests.robomme_hard._support.loaders import REPO
from tests.robomme_hard.contract.packaged_checks import (
    delivered,
    delivery_set_check,
    movecube_layout_check,
    read_tier,
    tier_values_check,
)
from tests.robomme_hard.contract.test_constants import (
    EXEC_CAP,
    LAYOUT_RULE,
    MOVECUBE_POINTS,
    MOVECUBE_REGION_V8,
    MOVECUBE_REGION_V9,
    MOVECUBE_WAYS,
    N_CELLS,
    NEW_TIERS,
    TOTAL,
    V9_CELLS,
    XHARD4_ONLY,
)
from tests.robomme_hard.contract.test_tier_table import TABLE

ROOT = REPO / "src" / "robomme_hard" / "env_metadata" / "ood"


def copy_root(tmp_path: Path) -> Path:
    dst = tmp_path / "root"
    shutil.copytree(ROOT, dst)
    return dst


def rewrite_rows(path: Path, mutate) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mutate(records[0], records[1:])
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def check(root: Path) -> dict:
    return delivery_set_check(root, NEW_TIERS, dict(V9_CELLS), exec_cap=EXEC_CAP, layout_rule=LAYOUT_RULE)


# ── 交付集 ─────────────────────────────────────────────────────────────


def test_delivery_set_pass_on_packaged():
    r = check(ROOT)
    assert (r["cells"], r["total"]) == (N_CELLS, TOTAL)
    assert r["cell_mismatch"] == [] and r["extra_cells"] == [] and r["missing_files"] == []
    assert r["selected_not_ok"] == 0 and r["delivered_over_cap"] == 0
    assert r["shared"] == 0 and r["seed_tasks"] == len({t for t, _ in V9_CELLS})
    assert r["mode_bad"] == 0 and r["parent_non_null"] == 0
    assert r["layout_equal_pairs"] == 0 and r["no_position"] == 0, r["layout_detail"]
    print(f"V9_DELIVERY_SET=PASS cells={r['cells']} total={r['total']} cell_mismatch=0 "
          f"selected_not_ok=0 delivered_over_cap=0 shared=0 layout_equal_pairs=0")


def test_delivery_set_fails_when_one_delivered_row_dropped(tmp_path):
    """负例：xhard5 一个交付行改成 selected=False → 该格少一局、总数 799、cell_mismatch=1。"""
    root = copy_root(tmp_path)

    def drop(header, rows):
        next(r for r in rows if delivered(r))["selected"] = False

    rewrite_rows(root / "xhard5" / "specs.jsonl", drop)
    r = check(root)
    assert r["total"] == TOTAL - 1 and len(r["cell_mismatch"]) == 1


def test_delivery_set_fails_on_missing_tier_file(tmp_path):
    root = copy_root(tmp_path)
    (root / "xhard3" / "specs.jsonl").unlink()
    r = check(root)
    assert r["missing_files"] == ["xhard3"] and r["cell_mismatch"]


def test_delivery_set_negatives_selected_not_ok_over_cap_seed_layout(tmp_path):
    """负例：selected 而 rollout failed、交付行执行步超上限、跨档 seed 撞号、layout_parent 非空、跨档照抄一个浮点
    位置叶子、header layout_rule 改掉，各自对应计数变为非零。"""
    root = copy_root(tmp_path)
    _, x2 = read_tier(root, "xhard2")
    stop_x2 = next(r for r in x2 if r["task"] == "StopCube" and delivered(r))

    def mutate_x1(header, rows):
        header["layout_rule"] = {"mode": "derived"}
        stop = [r for r in rows if r["task"] == "StopCube" and delivered(r)]
        stop[0]["rollout"]["status"] = "failed"
        stop[1]["rollout"]["exec_steps"] = EXEC_CAP + 1
        stop[2]["seed"] = stop_x2["seed"]
        stop[3]["layout_parent"] = "xhard2/0"
        stop[4]["spec"]["layout"] = json.loads(json.dumps(stop_x2["spec"]["layout"]))

    rewrite_rows(root / "xhard1" / "specs.jsonl", mutate_x1)
    r = check(root)
    assert r["selected_not_ok"] == 1 and r["delivered_over_cap"] == 1
    assert r["shared"] >= 1 and r["parent_non_null"] == 1 and r["mode_bad"] == 1
    assert r["layout_equal_pairs"] >= 1, r


# ── 档位取值 ───────────────────────────────────────────────────────────


def test_tier_values_pass_on_packaged():
    r = tier_values_check(ROOT, NEW_TIERS, TABLE)
    assert r["mismatches"] == [] and r["missing_files"] == []
    # 逐局核对的行数 = 交付格里有取值维度的局数（800 − MoveCube 50 − InsertPeg 50）
    assert r["rows"] == sum(n for (task, _), n in V9_CELLS.items() if task not in XHARD4_ONLY)


def test_tier_values_fail_on_changed_value(tmp_path):
    root = copy_root(tmp_path)

    def bump(header, rows):
        row = next(r for r in rows if r["task"] == "StopCube" and r["selected"])
        row["spec"]["actions"]["stop_time"] += 1

    rewrite_rows(root / "xhard5" / "specs.jsonl", bump)
    assert len(tier_values_check(root, NEW_TIERS, TABLE)["mismatches"]) == 1


# ── MoveCube 布局 ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def movecube_module():
    # 包 __init__ 把同名类导出到 robomme_hard.robomme_env.MoveCube 属性上，按模块路径取模块本身
    return importlib.import_module("robomme_hard.robomme_env.MoveCube")


def movecube_rows(root: Path = ROOT) -> list[dict]:
    _, rows = read_tier(root, "xhard4")
    return [r for r in rows if r["task"] == "MoveCube" and delivered(r)]


def run_movecube(rows, module) -> dict:
    return movecube_layout_check(rows, module=module, v9_region=MOVECUBE_REGION_V9, v8_region=MOVECUBE_REGION_V8)


def test_movecube_layout_pass_on_packaged(movecube_module):
    r = run_movecube(movecube_rows(), movecube_module)
    assert r["episodes"] == V9_CELLS[("MoveCube", "xhard4")]
    assert r["points"] == r["in_region"] == MOVECUBE_POINTS, r["out_detail"]
    assert r["outside_old"] >= 1 and r["region_mismatch"] == 0 and r["bad_rows"] == []
    assert r["ways"] == MOVECUBE_WAYS
    print(f"V9_MOVECUBE_LAYOUT=PASS episodes={r['episodes']} in_region={r['in_region']} points={r['points']} "
          f"outside_old={r['outside_old']} region_mismatch=0 ways={'/'.join(str(r['ways'][k]) for k in sorted(r['ways']))}")


def test_movecube_layout_fail_on_point_outside_region(movecube_module):
    """负例：一局 execution 段 goal 挪到远处 → 该点不在区域内（in_region 少 1）。"""
    rows = json.loads(json.dumps(movecube_rows()))
    rows[0]["spec"]["layout"]["execution"]["goal_xy"] = [5.0, 5.0]
    r = run_movecube(rows, movecube_module)
    assert r["in_region"] == MOVECUBE_POINTS - 1


def test_movecube_layout_negatives_region_way_bad_row(movecube_module):
    """负例：段内 region 的 r_out 改掉 → region_mismatch；一局 way_idx 换方式 → 配额不等于 17/17/16；
    删掉一局 peg_offsets → 坏行。"""
    rows = json.loads(json.dumps(movecube_rows()))
    rows[0]["spec"]["layout"]["demo"]["region"]["r_out"] = MOVECUBE_REGION_V9["r_out"] + 0.01
    inits = rows[1]["spec"]["initializations"]
    last = max(inits, key=int)
    inits[last]["way_idx"] = (inits[last]["way_idx"] + 1) % len(MOVECUBE_WAYS)
    del rows[2]["spec"]["layout"]["execution"]["peg_offsets"]
    r = run_movecube(rows, movecube_module)
    assert r["region_mismatch"] >= 1
    assert r["ways"] != MOVECUBE_WAYS
    assert len(r["bad_rows"]) == 1
