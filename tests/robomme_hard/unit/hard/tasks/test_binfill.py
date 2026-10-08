"""BinFill 新值档真值表（xhard1、xhard2）：按语言序列逐色投入正确数量的方块，再按按钮为成功。

错误与边界：少投一块就按按钮、按钮阶段多投一块、投错颜色（数量对但颜色不对）、提前按按钮均为失败；
投入后方块被移出场景，计数只加一次。颜色与配额取自本局（包内规格回放）。
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world

TASK = "BinFill"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _cubes(w, color):
    return list(getattr(w.env, f"{color}_cubes"))


def _drop_in(w, cube):
    """抓起 → 放进孔板（落在孔板中心、松手、夹爪抬开）。"""
    w.grasp(cube)
    a = w.tick()
    w.release_onto(cube, w.xyz(w.env.board_with_hole)[:2])
    b = w.tick()
    return a, b


def _fill(w, plan):
    for color, n in plan:
        for cube in _cubes(w, color)[:n]:
            assert _drop_in(w, cube) == (OK, OK)


@pytest.mark.parametrize("tier", TIERS)
def test_quota_then_button_succeeds(world, tier):
    w = world(tier)
    plan = list(w.env.binfill_language_sequence)
    _fill(w, plan)
    for color, n in plan:
        assert getattr(w.env, f"{color}_cubes_in_bin") == n
    w.press(w.env.button)
    assert w.tick() == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS)
def test_one_short_then_button_fails(world, tier):
    w = world(tier)
    plan = list(w.env.binfill_language_sequence)
    color, n = plan[-1]
    _fill(w, plan[:-1] + [(color, n - 1)])
    w.press(w.env.button)
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS)
def test_extra_cube_in_button_stage_fails(world, tier):
    w = world(tier)
    plan = list(w.env.binfill_language_sequence)
    _fill(w, plan)
    color, n = plan[0]
    spare = _cubes(w, color)[n]  # 同色多一块
    w.grasp(spare)
    w.tick()
    w.release_onto(spare, w.xyz(w.env.board_with_hole)[:2])
    assert w.tick()["fail"] is True


@pytest.mark.parametrize("tier", TIERS[:1])
def test_wrong_colour_right_total_fails_at_button(world, tier):
    """总块数对、颜色不对：用多余颜色的方块顶替最后一种颜色的一块 → 按按钮时计数不符即失败。"""
    w = world(tier)
    plan = list(w.env.binfill_language_sequence)
    color, n = plan[-1]
    other = next(c for c, _ in plan if c != color)
    used = dict(plan)
    _fill(w, plan[:-1] + [(color, n - 1)])
    substitute = _cubes(w, other)[used[other]]
    assert _drop_in(w, substitute) == (OK, OK)
    w.press(w.env.button)
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_early_button_fails_and_binned_cube_counts_once(world, tier):
    w = world(tier)
    color, _ = w.env.binfill_language_sequence[0]
    cube = _cubes(w, color)[0]
    _drop_in(w, cube)
    assert getattr(w.env, f"{color}_cubes_in_bin") == 1
    # 投入的方块被移出孔板，再 evaluate 不会重复计数
    w.tick()
    w.tick()
    assert getattr(w.env, f"{color}_cubes_in_bin") == 1
    w.press(w.env.button)
    assert w.tick() == {"success": False, "fail": True}
