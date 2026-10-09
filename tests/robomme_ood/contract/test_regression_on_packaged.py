"""L1 contract: run three data checks on the real packaged specs (delivery set, tier values, MoveCube layout), each with negatives.

The old repo called the ``delivery-set``/``tier-values``/``movecube-layout`` subcommands of ``scripts/parity/hard_regression.py`` here;
that tool moved to the private evaluation repo, so this repo uses the test-side equivalent checks in ``packaged_checks`` (reading spec rows with stdlib json only; criteria and values match
the tool implementation at ``fd0017d6``). ``step-headroom`` needs per-episode h5 and the delivery manifest; it is tool behavior and not in this repo.

- delivery set: delivered rows (selected and rollout ok) per cell over the 43 V9 cells equal the cell table, no delivery outside the table, selected-but-failed is 0,
  delivered rows' execution steps <= ``EXEC_CAP``, 800 episodes in total; same-task seeds disjoint across tiers; layouts independent (header ``layout_rule``,
  ``layout_parent`` all null, position leaves of delivered rows not copied across tiers);
- tier values: per-episode values over 14 tasks and 41 cells equal ``test_tier_table.TABLE`` (row count = 800 - MoveCube 50 - InsertPeg 50);
- MoveCube layout: xhard4 50 episodes x 2 segments x 3 objects = 300 points all in the V9 region, at least one point outside the old V8 region, in-segment region
  consistent with the source and the three V9 numbers, motion modes 17/17/16.

Negatives each introduce one error in a tmp copy or in-memory copy; the corresponding count must change.
"""
from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import pytest

from tests.robomme_ood._support.loaders import REPO
from tests.robomme_ood.contract.packaged_checks import (
    delivered,
    delivery_set_check,
    movecube_layout_check,
    read_tier,
    tier_values_check,
)
from tests.robomme_ood.contract.test_constants import (
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
from tests.robomme_ood.contract.test_tier_table import TABLE

ROOT = REPO / "src" / "robomme_ood" / "env_metadata" / "ood"


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


# -- Delivery set -------------------------------------------------------------


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
    """Negative: one xhard5 delivered row set to selected=False -> that cell is short by one, total 799, cell_mismatch=1."""
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
    """Negatives: selected but rollout failed, delivered row execution steps over the cap, cross-tier seed collision, non-null layout_parent, one float
    position leaf copied across tiers, header layout_rule changed; each makes the corresponding count non-zero."""
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


# -- Tier values -------------------------------------------------------------


def test_tier_values_pass_on_packaged():
    r = tier_values_check(ROOT, NEW_TIERS, TABLE)
    assert r["mismatches"] == [] and r["missing_files"] == []
    # rows checked per episode = episodes in delivery cells that have value dimensions (800 - MoveCube 50 - InsertPeg 50)
    assert r["rows"] == sum(n for (task, _), n in V9_CELLS.items() if task not in XHARD4_ONLY)


def test_tier_values_fail_on_changed_value(tmp_path):
    root = copy_root(tmp_path)

    def bump(header, rows):
        row = next(r for r in rows if r["task"] == "StopCube" and r["selected"])
        row["spec"]["actions"]["stop_time"] += 1

    rewrite_rows(root / "xhard5" / "specs.jsonl", bump)
    assert len(tier_values_check(root, NEW_TIERS, TABLE)["mismatches"]) == 1


# -- MoveCube layout --------------------------------------------------------


@pytest.fixture(scope="module")
def movecube_module():
    # the package __init__ exports the same-named class as the robomme_ood.robomme_env.MoveCube attribute; fetch the module itself by module path
    return importlib.import_module("robomme_ood.robomme_env.MoveCube")


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
    """Negative: move one episode's execution-segment goal far away -> that point is outside the region (in_region drops by 1)."""
    rows = json.loads(json.dumps(movecube_rows()))
    rows[0]["spec"]["layout"]["execution"]["goal_xy"] = [5.0, 5.0]
    r = run_movecube(rows, movecube_module)
    assert r["in_region"] == MOVECUBE_POINTS - 1


def test_movecube_layout_negatives_region_way_bad_row(movecube_module):
    """Negatives: change r_out of an in-segment region -> region_mismatch; switch one episode's way_idx -> quota no longer 17/17/16;
    delete one episode's peg_offsets -> bad row."""
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
