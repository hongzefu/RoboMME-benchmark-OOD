"""VPB／VPO 演示落点工具（``utils/xhard_home_site.py``）：演示计划守卫、放置序列占台守卫、放回掩码、
goal_site 区域落点网格与贪心选点。手写小表做输入。"""
from __future__ import annotations

import math

import pytest

from robomme_hard.robomme_env.utils import xhard_home_site as H
from robomme_hard.robomme_env.utils.SceneGenerationError import SceneGenerationError


# ── validate_demo_plan ────────────────────────────────────────────────────────


def test_newvalue_tiers_require_return_to_origin_and_count_range():
    assert H.validate_demo_plan(2, H.RETURN_TO_ORIGIN, "xhard1", 3) == (2, H.RETURN_TO_ORIGIN)
    for bad in [(2, H.RETURN_LAST_ONLY, "xhard2", 3), (0, H.RETURN_TO_ORIGIN, "xhard1", 3),
                (4, H.RETURN_TO_ORIGIN, "xhard1", 3)]:
        with pytest.raises(SceneGenerationError):
            H.validate_demo_plan(*bad)


def test_native_tiers_only_accept_native_plan():
    assert H.validate_demo_plan(*H.NATIVE_DEMO_PLAN, "hard", 3) == H.NATIVE_DEMO_PLAN
    with pytest.raises(SceneGenerationError):
        H.validate_demo_plan(2, H.NATIVE_DEMO_PLAN[1], "easy", 3)


# ── validate_place_sequence ───────────────────────────────────────────────────


def test_place_sequence_moves_cube_between_targets():
    occ = H.validate_place_sequence([(0, 0), (1, 2), (0, 1)], 4)
    assert occ == {0: None, 1: 0, 2: 1, 3: None}


def test_two_cubes_on_one_target_rejected():
    with pytest.raises(SceneGenerationError, match="两块同台"):
        H.validate_place_sequence([(0, 1), (1, 1)], 3)


def test_placing_onto_own_target_is_idle_and_rejected():
    with pytest.raises(SceneGenerationError, match="原地空转"):
        H.validate_place_sequence([(0, 1), (0, 1)], 3)


def test_out_of_range_target_and_empty_board_rejected():
    with pytest.raises(SceneGenerationError):
        H.validate_place_sequence([(0, 3)], 3)
    with pytest.raises(SceneGenerationError):
        H.validate_place_sequence([], 0)


# ── returned_mask ─────────────────────────────────────────────────────────────


def test_returned_mask_by_policy():
    assert H.returned_mask(H.RETURN_TO_ORIGIN, 3) == [True, True, True]
    assert H.returned_mask(H.RETURN_LAST_ONLY, 3) == [False, False, True]
    assert H.returned_mask(H.NO_RETURN, 2) == [False, False]
    with pytest.raises(SceneGenerationError):
        H.returned_mask("whatever", 2)


# ── goal_drop_candidates／plan_goal_drop_xy ─────────────────────────────────────


def test_candidates_sorted_by_distance_then_xy():
    c = H.goal_drop_candidates((0.0, 0.0), half=0.02, step=0.01)
    assert len(c) == 25 and c[0] == (0.0, 0.0)
    d = [math.hypot(x, y) for x, y in c]
    assert d == sorted(d)
    ring1 = c[1:5]  # 四个距离 0.01 的点按 (x, y) 字典序
    assert ring1 == sorted(ring1)


def test_plan_takes_centre_when_free_and_respects_clearance():
    assert H.plan_goal_drop_xy((0.1, 0.2), 1, []) == [(0.1, 0.2)]
    # 中心被方块占着：第一个落点必须离方块至少 cube 间隙
    chosen = H.plan_goal_drop_xy((0.0, 0.0), 2, [("cube", (0.0, 0.0))])
    gap = H.GOAL_DROP_CLEARANCE
    for x, y in chosen:
        assert math.hypot(x, y) >= gap["cube"] - 1e-9
    (x1, y1), (x2, y2) = chosen
    assert math.hypot(x1 - x2, y1 - y2) >= gap["drop"] - 1e-9


def test_plan_raises_when_region_blocked():
    obstacles = [("button", (x / 10, y / 10)) for x in range(-3, 4) for y in range(-3, 4)]
    with pytest.raises(SceneGenerationError):
        H.plan_goal_drop_xy((0.0, 0.0), 1, obstacles)
