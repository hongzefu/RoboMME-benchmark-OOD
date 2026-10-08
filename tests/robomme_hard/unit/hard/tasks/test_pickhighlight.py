"""PickHighlight 新值档真值表（xhard1、xhard2）：按按钮 → 依次拾放每个高亮方块 → 末尾再按按钮为成功。

错误与边界：漏一个目标或不按末按钮不成功；抓未高亮方块即失败；首按钮之前抓任何方块即失败；
重复抓同一目标不推进也不失败。高亮目标与顺序取自本局（包内规格回放）。
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world

TASK = "PickHighlight"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _press_release(w):
    w.press(w.env.button)
    out = w.tick()
    w.unpress(w.env.button)
    return out


def _pick_place(w, cube):
    w.grasp(cube)
    a = w.tick()
    w.release_onto(cube, w.xyz(cube)[:2])
    b = w.tick()
    return a, b


@pytest.mark.parametrize("tier", TIERS)
def test_all_highlighted_then_final_button_succeeds(world, tier):
    w = world(tier)
    assert _press_release(w) == OK
    for cube in w.env.target_cubes:
        assert _pick_place(w, cube) == (OK, OK)
    assert w.tick() == OK, "不按末按钮不算成功"
    w.press(w.env.button)
    assert w.tick() == {"success": True, "fail": False}
    assert all(v == 1 for v in w.env.target_cube_pickup_counts.values())


@pytest.mark.parametrize("tier", TIERS)
def test_missing_last_target_never_succeeds(world, tier):
    w = world(tier)
    _press_release(w)
    for cube in w.env.target_cubes[:-1]:
        _pick_place(w, cube)
    w.press(w.env.button)
    for _ in range(3):
        assert w.tick() == OK


@pytest.mark.parametrize("tier", TIERS)
def test_unhighlighted_cube_fails(world, tier):
    w = world(tier)
    _press_release(w)
    other = next(c for c in w.env.all_cubes if c not in w.env.target_cubes)
    w.grasp(other)
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_pick_before_first_button_fails(world, tier):
    w = world(tier)
    w.grasp(w.env.target_cubes[0])
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_repeating_same_target_does_not_advance(world, tier):
    w = world(tier)
    _press_release(w)
    first = w.env.target_cubes[0]
    _pick_place(w, first)
    stage = w.stage
    assert _pick_place(w, first) == (OK, OK)
    assert w.stage == stage, "重复抓第一个目标不应推进到下一项"
    assert w.env.target_cube_pickup_counts[first.name] == 2
