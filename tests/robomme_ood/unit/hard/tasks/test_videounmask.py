"""VideoUnmask 新值档真值表（xhard1～xhard4）：演示后静止 → 依次抓起「藏着第 k 种颜色方块的容器」、放下、再抓下一个。

错误与边界：抓错容器（另一个区域内容器）即失败；两次抓取之间不放下而直接抬下一个即失败；
抬起任一干扰容器即失败；静止不满 64 步不推进。抓取次数（单／双／三选）取自本档（包内规格回放）。
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import unmask_driver as D

TASK = "VideoUnmask"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _settle(w):
    """第一项静止检查：机械臂静止若干步后推进（步数由生产 static_check 判，测试只推进到阶段变化）。"""
    w.still()
    for _ in range(200):
        out = w.tick()
        if w.stage >= 1:
            return out
    raise AssertionError("静止 200 步仍未推进")


def _bins_in_order(w):
    cubes = D.colour_cubes(w.env)[: w.env.xhard_pick_count]
    return [D.bin_hiding(w, c, w.env.spawned_bins) for c in cubes]


@pytest.mark.parametrize("tier", TIERS)
def test_pick_each_hiding_container_in_order_succeeds(world, tier):
    w = world(tier)
    assert _settle(w) == OK
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


@pytest.mark.parametrize("tier", TIERS)
def test_wrong_container_fails(world, tier):
    w = world(tier)
    _settle(w)
    right = _bins_in_order(w)[0]
    wrong = next(b for b in w.env.spawned_bins if b is not right)
    D.lift(w, wrong)
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS)
def test_distractor_container_fails(world, tier):
    w = world(tier)
    _settle(w)
    D.lift(w, w.env.distractor_bins[0])
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_second_pick_without_putting_down_fails(world, tier):
    w = world(tier)
    _settle(w)
    first, second = _bins_in_order(w)[:2]
    D.lift(w, first)
    assert w.tick() == OK
    D.lift(w, second)  # 第一个还抬着
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_moving_robot_delays_static_stage(world, tier):
    w = world(tier)
    w.moving()
    for _ in range(100):
        assert w.tick() == OK
    assert w.stage == 0
