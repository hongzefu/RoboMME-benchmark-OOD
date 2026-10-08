"""InsertPeg 新值档（V9 只交付 xhard4）：离线 ``_load_scene`` + 两次 ``_initialize_episode`` 的杆数、杆位与间隔、
抓取端与插入端绑定、包内规格回放与自导出；杆轮廓间隔 ``footprint_gap`` 的手算算例。"""
from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from . import cells as C
from . import offline_scene as O

TASK = "InsertPeg"
IP = importlib.import_module("robomme_hard.robomme_env.InsertPeg")


def _decision(tier):
    header, _ = O.delivered_rows(TASK, tier, 0)
    return header["sampling_config"][TASK]["decision"][tier]


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK))
def test_packaged_spec_replays_with_zero_mismatch(task, tier, k):
    C.check_packaged_replay(task, tier, k)


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK))
def test_offline_export_equals_package_and_replays(task, tier, k):
    C.check_self_export(task, tier, k)


@pytest.mark.parametrize("tier", O.tiers_of(TASK))
def test_tampered_spec_is_detected(tier):
    C.check_tamper_detected(TASK, tier)


@pytest.mark.parametrize("tier", O.tiers_of(TASK))
@pytest.mark.parametrize("k", range(C.REPLAY_ROWS))
def test_pegs_count_gaps_and_target_binding(tier, k):
    row, env = C.replayed(TASK, tier, k)
    dec = _decision(tier)
    assert len(env.pegs) == dec["peg_count"] == len(env.peg_heads) == len(env.peg_tails)
    init = row["spec"]["initializations"][str(1)]  # 评估用的是第二次初始化
    # 实测最小间隔（生产 record）必须严格大于本档下限（规则是严格不等号）
    assert init["min_pair_gap_m"] > dec["peg_min_pair_gap_m"]
    assert init["min_box_gap_m"] > dec["peg_box_min_gap_m"]
    # 杆根 x 不超过本档上界；杆位就是冻结值（根随 set_pose 移到规格位置）
    for i, peg in enumerate(env.pegs):
        (x, y), _yaw = init["pegs"][str(i)]
        assert x <= dec["peg_x_max_m"] + 1e-9
    # 被抓的目标恒为第一根杆；抓取端与插入端是同一根杆的两端
    assert env.peg is env.pegs[0]
    assert {env.grasp_target, env.insert_target} == {env.peg_head, env.peg_tail}
    assert env.grasp_target is not env.insert_target
    assert env.insert_way in ("left", "right")
    assert env.grasp_target_distance in ("near", "far")


# ── footprint_gap 手算算例（独立几何：两条平行轴对齐矩形之间的空隙）────────────────


def test_footprint_gap_parallel_rectangles():
    # 两根沿 x 的杆，长 0.1、半宽 0.01，根相距 y=0.05：轮廓之间空隙 = 0.05 − 2×0.01 = 0.03
    a = IP.peg_footprint(np.array([0.0, 0.0]), 0.0, 0.1, 0.01)
    b = IP.peg_footprint(np.array([0.0, 0.05]), 0.0, 0.1, 0.01)
    assert math.isclose(IP.footprint_gap(a, b), 0.03, abs_tol=1e-9)
    assert math.isclose(IP.footprint_gap(b, a), 0.03, abs_tol=1e-9)


def test_footprint_gap_overlap_and_far_apart():
    a = IP.peg_footprint(np.array([0.0, 0.0]), 0.0, 0.1, 0.01)
    same = IP.peg_footprint(np.array([0.0, 0.0]), 0.0, 0.1, 0.01)
    assert IP.footprint_gap(a, same) <= 0.0
    far = IP.peg_footprint(np.array([1.0, 0.0]), 0.0, 0.1, 0.01)
    assert IP.footprint_gap(a, far) > 0.5
