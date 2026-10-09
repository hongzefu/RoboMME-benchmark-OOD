"""L1 contract: the single V9 tier value table (v8 plan part one table 1, reused by V9), checked against three places at once:

1. the in-process ``decision`` (per-tier new values) returned by each task's ``native_blocks(cls)``;
2. ``sampling_config[task]`` in each packaged tier header (verbatim equal to 1's ``{"decision", "native"}``, and valued per the table);
3. the packaged per-row specs: the episode's actual values in delivered rows' ``spec`` (via the test-side reader ``packaged_checks.tier_dims``, fields identical item by item to the old repo's
   ``scripts/parity/hard_regression.py::tier_dims``; that tool lives in the private evaluation repo, so this repo carries an equivalent check).
Fixed values in the table are integers, intervals are ``(lo, hi)`` closed intervals (RouteStick segment count, PatternLock node count).
MoveCube and InsertPeg have no value dimensions (MoveCube's region and motion modes are guarded by movecube-layout in ``test_regression_on_packaged``).
This table is an expectation written independently on the test side; it does not read ``hard_regression.V8_TIER_TABLE``.
"""
from __future__ import annotations

import importlib
import json

import pytest

from tests.robomme_ood._support.loaders import REPO
from tests.robomme_ood.contract.packaged_checks import row_mismatches, tier_dims
from tests.robomme_ood.contract.test_constants import NEW_TIERS, V9_CELLS, XHARD4_ONLY

ROOT = REPO / "src" / "robomme_ood" / "env_metadata" / "ood"

#: {task: {tier: {dimension: fixed value or (lo, hi)}}}, delivery cells only
TABLE = {
    "PickXtimes": {"xhard1": {"times": 6, "distractors": 1}, "xhard2": {"times": 7, "distractors": 2},
                   "xhard3": {"times": 8, "distractors": 3}},
    "SwingXtimes": {"xhard1": {"rounds": 4, "distractors": 1}, "xhard2": {"rounds": 5, "distractors": 2},
                    "xhard3": {"rounds": 6, "distractors": 3}, "xhard4": {"rounds": 7, "distractors": 4},
                    "xhard5": {"rounds": 8, "distractors": 4}},
    "StopCube": {"xhard1": {"stop_time": 6, "move_interval": 60}, "xhard2": {"stop_time": 7, "move_interval": 60},
                 "xhard3": {"stop_time": 8, "move_interval": 60}, "xhard4": {"stop_time": 9, "move_interval": 60},
                 "xhard5": {"stop_time": 10, "move_interval": 60}},
    "VideoUnmask": {"xhard1": {"pick": 2, "distractor_bins": 4, "distractor_cubes": 2},
                    "xhard2": {"pick": 3, "distractor_bins": 4, "distractor_cubes": 2},
                    "xhard3": {"pick": 3, "distractor_bins": 8, "distractor_cubes": 4},
                    "xhard4": {"pick": 3, "distractor_bins": 12, "distractor_cubes": 6}},
    "ButtonUnmask": {"xhard1": {"pick": 2, "distractor_bins": 4, "distractor_cubes": 2},
                     "xhard2": {"pick": 3, "distractor_bins": 4, "distractor_cubes": 2},
                     "xhard3": {"pick": 3, "distractor_bins": 8, "distractor_cubes": 4},
                     "xhard4": {"pick": 3, "distractor_bins": 12, "distractor_cubes": 6}},
    "BinFill": {"xhard1": {"put_in": 6}, "xhard2": {"put_in": 7}},
    "VideoUnmaskSwap": {"xhard1": {"swap": 5, "pick": 2, "outer": 2}, "xhard2": {"swap": 7, "pick": 3, "outer": 4}},
    "ButtonUnmaskSwap": {"xhard1": {"swap": 3, "pick": 2, "outer": 2}, "xhard2": {"swap": 5, "pick": 3, "outer": 4}},
    "VideoPlaceButton": {"xhard1": {"placements": 3}, "xhard2": {"placements": 4}},
    "VideoPlaceOrder": {"xhard1": {"visits": 5}, "xhard2": {"visits": 6}},
    "PickHighlight": {"xhard1": {"pick": 4, "total": 7}, "xhard2": {"pick": 5, "total": 8}},
    "VideoRepick": {"xhard1": {"cubes": 4, "swap": 4, "repick": 2}, "xhard2": {"cubes": 5, "swap": 6, "repick": 3}},
    "RouteStick": {"xhard1": {"segments": (8, 10)}, "xhard2": {"segments": (11, 13)},
                   "xhard3": {"segments": (14, 16)}},
    "PatternLock": {"xhard1": {"nodes": (9, 12)}, "xhard2": {"nodes": (13, 15)}, "xhard3": {"nodes": (16, 18)}},
}


def _fixed(pair) -> int | tuple[int, int]:
    """A ``[a, b]`` closed interval in the config -> fixed value a when a == b, otherwise (a, b)."""
    lo, hi = pair
    return lo if lo == hi else (lo, hi)


def _vpb_placements(cfg) -> int:
    return importlib.import_module("robomme_ood.robomme_env.VideoPlaceButton").vpb_target_placement_count(cfg)


#: Accessors reading a tier's value from the decision block (field reading and interval normalization only; sampling logic is not re-implemented)
DECISION_READERS = {
    "PickXtimes": lambda d, t: {"times": _fixed(d["number_range"][t]), "distractors": len(d[t]["distractor"]["colors"])},
    "SwingXtimes": lambda d, t: {"rounds": _fixed(d["number_range"][t]), "distractors": len(d[t]["distractor"]["colors"])},
    "StopCube": lambda d, t: {
        "stop_time": _fixed((d[t]["stop_time_range"]["low"], d[t]["stop_time_range"]["high_exclusive"] - 1)),
        "move_interval": _fixed((min(d[t]["move_interval_choices"]), max(d[t]["move_interval_choices"])))},
    "VideoUnmask": lambda d, t: {"pick": d["pick_count"][t], "distractor_bins": d[t]["distractor"]["count"],
                                 "distractor_cubes": _fixed(d[t]["distractor"]["cube_count_range"])},
    "ButtonUnmask": lambda d, t: {"pick": d["pick_count"][t], "distractor_bins": d[t]["distractor"]["count"],
                                  "distractor_cubes": _fixed(d[t]["distractor"]["cube_count_range"])},
    "BinFill": lambda d, t: {"put_in": _fixed(d["configs"][t]["put_in_numbers"])},
    "VideoUnmaskSwap": lambda d, t: {"swap": _fixed(d["swap_count_range"][t]), "pick": _fixed(d["pick_count_range"][t]),
                                     "outer": d[t]["distractor"]["count"]},
    "ButtonUnmaskSwap": lambda d, t: {"swap": _fixed(d["swap_count_range"][t]), "pick": _fixed(d["pick_count_range"][t]),
                                      "outer": d[t]["distractor"]["count"]},
    "VideoPlaceButton": lambda d, t: {"placements": _vpb_placements(d[t])},
    "VideoPlaceOrder": lambda d, t: {"visits": sum(d[t]["visit_counts"])},
    "PickHighlight": lambda d, t: {"pick": _fixed(d["highlight_count"][t]), "total": _fixed(d["spawn_count"][t])},
    "VideoRepick": lambda d, t: {
        "cubes": d[t]["layout"]["cube_count"],
        "swap": _fixed((d["swap"][t]["swap_min"], d["swap"][t]["swap_max"])),
        "repick": _fixed((d["num_repeats_range"][t]["low"], d["num_repeats_range"][t]["high_exclusive"] - 1))},
    "RouteStick": lambda d, t: {"segments": _fixed(d[t]["segment_count_range"])},
    "PatternLock": lambda d, t: {"nodes": _fixed(d["path_length_range"][t])},
}

VALUED_CELLS = sorted((task, tier) for task, tiers in TABLE.items() for tier in tiers)


def _norm(value):
    return json.loads(json.dumps(value))


def native_blocks(task):
    module = importlib.import_module(f"robomme_ood.robomme_env.{task}")
    decision, native = module.native_blocks(getattr(module, task))
    return _norm(decision), _norm(native)


def headers():
    return {tier: json.loads((ROOT / tier / "specs.jsonl").read_text(encoding="utf-8").splitlines()[0])
            for tier in NEW_TIERS}


def test_table_covers_exactly_valued_delivery_cells():
    """Table = V9 delivery cells minus the two tasks delivered only in xhard4 with no value dimensions (41 cells)."""
    assert set(VALUED_CELLS) == {key for key in V9_CELLS if key[0] not in XHARD4_ONLY}
    assert set(DECISION_READERS) == set(TABLE)


@pytest.mark.parametrize("task,tier", VALUED_CELLS)
def test_native_blocks_decision_matches_table(task, tier):
    decision, _ = native_blocks(task)
    assert DECISION_READERS[task](decision, tier) == TABLE[task][tier]


@pytest.mark.parametrize("tier", NEW_TIERS)
def test_header_sampling_config_equals_native_blocks_and_table(tier):
    header = headers()[tier]
    for task in header["tasks"]:
        decision, native = native_blocks(task)
        assert header["sampling_config"][task] == {"decision": decision, "native": native}, task
        if task in TABLE:
            assert DECISION_READERS[task](header["sampling_config"][task]["decision"], tier) == TABLE[task][tier]


@pytest.mark.parametrize("tier", NEW_TIERS)
def test_packaged_rows_match_table(tier):
    lines = (ROOT / tier / "specs.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines[1:] if line.strip()]
    assert row_mismatches(rows, tier, TABLE) == []
    checked = {(r["task"], tier) for r in rows if r["task"] in TABLE and r["selected"]}
    assert checked == {key for key in VALUED_CELLS if key[1] == tier}


def test_row_mismatches_negative():
    """Checker negatives: changing one PickXtimes episode's count to a value outside the table, moving a RouteStick segment count out of its interval, and deleting a value field are all caught."""
    lines = (ROOT / "xhard1" / "specs.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines[1:]]
    pick = next(r for r in rows if r["task"] == "PickXtimes" and r["selected"])
    route = next(r for r in rows if r["task"] == "RouteStick" and r["selected"])
    stop = next(r for r in rows if r["task"] == "BinFill" and r["selected"])
    pick, route, stop = _norm(pick), _norm(route), _norm(stop)
    assert row_mismatches([pick, route, stop], "xhard1", TABLE) == []
    assert tier_dims("PickXtimes", pick["spec"])["times"] == TABLE["PickXtimes"]["xhard1"]["times"]
    pick["spec"]["objects"]["num_repeats"] = TABLE["PickXtimes"]["xhard1"]["times"] + 1
    route["spec"]["objects"]["L"] = TABLE["RouteStick"]["xhard1"]["segments"][1] + 1
    del stop["spec"]["objects"]["target_numbers"]
    assert len(row_mismatches([pick, route, stop], "xhard1", TABLE)) == 3


def test_decision_reader_negative():
    """Checker negatives: changing one tier's value in the decision block makes the read result differ from the table."""
    decision, _ = native_blocks("StopCube")
    decision["xhard3"]["stop_time_range"]["low"] += 1
    assert DECISION_READERS["StopCube"](decision, "xhard3") != TABLE["StopCube"]["xhard3"]
