"""L1 契约：唯一一份 V9 档位取值表（v8 方案第一部分表 1，V9 沿用），对两处核对：

1. 进程内各任务 ``native_blocks(cls)`` 给出的 ``decision``（按档位的新值取值）；
2. 包内各档 header 的 ``sampling_config[task]``（逐字等于 1 的 ``{"decision", "native"}``，并按表取值）。

逐行规格实际取值的核对依赖 ``scripts/parity/hard_regression.py`` 的 ``tier_dims``，该工具与其断言随私有评估仓，
本仓不测。
表里定值写成整数，区间写成 ``(lo, hi)`` 闭区间（RouteStick 的段数、PatternLock 的节点数）。
MoveCube、InsertPeg 不计取值维度。
本表是测试侧独立写下的期望，不读 ``hard_regression.V8_TIER_TABLE``。
"""
from __future__ import annotations

import importlib
import json

import pytest

from tests.robomme_hard._support.loaders import REPO
from tests.robomme_hard.contract.test_constants import NEW_TIERS, V9_CELLS, XHARD4_ONLY

ROOT = REPO / "src" / "robomme_hard" / "env_metadata" / "ood"

#: {task: {tier: {维度: 定值 或 (lo, hi)}}}，只含交付格
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
    """配置里的 ``[a, b]`` 闭区间 → a == b 时为定值 a，否则 (a, b)。"""
    lo, hi = pair
    return lo if lo == hi else (lo, hi)


def _vpb_placements(cfg) -> int:
    return importlib.import_module("robomme_hard.robomme_env.VideoPlaceButton").vpb_target_placement_count(cfg)


#: 从 decision 块读某档取值的访问器（只做字段读取与区间归一，不复刻采样逻辑）
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
    module = importlib.import_module(f"robomme_hard.robomme_env.{task}")
    decision, native = module.native_blocks(getattr(module, task))
    return _norm(decision), _norm(native)


def headers():
    return {tier: json.loads((ROOT / tier / "specs.jsonl").read_text(encoding="utf-8").splitlines()[0])
            for tier in NEW_TIERS}


def test_table_covers_exactly_valued_delivery_cells():
    """表 = V9 交付格去掉只在 xhard4 交付、不计取值的两个任务（41 格）。"""
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


def test_decision_reader_negative():
    """判定器负例：decision 块里改一档的取值，读出的结果不再等于表。"""
    decision, _ = native_blocks("StopCube")
    decision["xhard3"]["stop_time_range"]["low"] += 1
    assert DECISION_READERS["StopCube"](decision, "xhard3") != TABLE["StopCube"]["xhard3"]
