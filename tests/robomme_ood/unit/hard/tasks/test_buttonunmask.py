"""ButtonUnmask 新值档真值表（xhard1～xhard4）：先按按钮揭示 → 依次抓起「藏着第 k 种颜色方块的容器」并放下。

错误与边界：跳过按钮直接抓对的容器不推进（按钮项不判失败）；按钮之后抓错容器、抬干扰容器即失败；
按钮按下深度为零不算按下，按到行程底才算。抓取次数取自本档。
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import unmask_driver as D

TASK = "ButtonUnmask"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _bins_in_order(w):
    cubes = D.colour_cubes(w.env)[: w.env.xhard_pick_count]
    return [D.bin_hiding(w, c, w.env.spawned_bins) for c in cubes]


def _press(w):
    w.press(w.env.button)
    out = w.tick()
    w.unpress(w.env.button)
    return out


@pytest.mark.parametrize("tier", TIERS)
def test_button_then_each_hiding_container_succeeds(world, tier):
    w = world(tier)
    assert _press(w) == OK and w.stage == 1
    bins = _bins_in_order(w)
    out = None
    for k, b in enumerate(bins):
        assert w.env.color_names[k] in w.env.task_list[w.stage]["name"]
        D.lift(w, b)
        out = w.tick()
        if k < len(bins) - 1:
            assert out == OK
            D.put_down(w, b)
            assert w.tick() == OK
    assert out == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_skipping_button_does_not_advance(world, tier):
    w = world(tier)
    D.lift(w, _bins_in_order(w)[0])
    for _ in range(3):
        assert w.tick() == OK
    assert w.stage == 0


@pytest.mark.parametrize("tier", TIERS)
def test_wrong_or_distractor_container_after_button_fails(world, tier):
    w = world(tier)
    _press(w)
    right = _bins_in_order(w)[0]
    D.lift(w, next(b for b in w.env.spawned_bins if b is not right))
    assert w.tick() == {"success": False, "fail": True}
    w = World.build(TASK, tier)
    _press(w)
    D.lift(w, w.env.distractor_bins[-1])
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_button_depth_zero_is_not_a_press(world, tier):
    w = world(tier)
    w.press(w.env.button, depth=0.0)
    assert w.tick() == OK and w.stage == 0
    w.press(w.env.button)
    assert w.tick() == OK and w.stage == 1
