"""PickXtimes 新值档真值表：N 次完整拾放后按按钮为成功；N−1／N+1、持续抓住、错对象、持物按按钮为失败或不成功。

CPU 世界替身驱动真实任务表与 ``evaluate``；N 取自本局 ``num_repeats``（包内规格回放），不在测试里写死。
（阈值等号的边界用例放在 SwingXtimes：其阈值来自 sampling 参数，可以换成 float32 精确可表示的值。）
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world

TASK = "PickXtimes"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


def _disc_xy(w):
    return w.xyz(w.env.target)[:2]


def _cycle(w, n):
    """完整拾放 n 次：抓起目标方块 → 放到圆盘上，各 evaluate 一次。"""
    for _ in range(n):
        w.grasp(w.env.target_cube)
        assert w.tick() == OK
        w.release_onto(w.env.target_cube, _disc_xy(w))
        assert w.tick() == OK


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_n_cycles_then_button_succeeds(world, tier):
    w = world(tier)
    n = w.env.num_repeats
    _cycle(w, n)
    w.press(w.env.button)
    assert w.tick() == {"success": True, "fail": False}
    # 成功后终态稳定
    assert w.tick()["success"] is True


@pytest.mark.parametrize("tier", TIERS)
def test_one_cycle_short_then_button_fails(world, tier):
    w = world(tier)
    _cycle(w, w.env.num_repeats - 1)
    w.press(w.env.button)
    assert w.tick() == {"success": False, "fail": True}
    w.unpress(w.env.button)
    assert w.tick()["fail"] is True, "失败终态必须保持"


@pytest.mark.parametrize("tier", TIERS)
def test_extra_cycle_fails(world, tier):
    w = world(tier)
    _cycle(w, w.env.num_repeats)
    w.grasp(w.env.target_cube)  # 第 N+1 次抓起：按钮阶段任何方块被抓起都算失败
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS)
def test_pressing_button_while_holding_fails(world, tier):
    w = world(tier)
    _cycle(w, w.env.num_repeats)
    w.grasp(w.env.target_cube)
    w.press(w.env.button)
    assert w.tick() == {"success": False, "fail": True}, "失败判定优先于完成判定"


@pytest.mark.parametrize("tier", TIERS)
def test_holding_forever_never_completes(world, tier):
    w = world(tier)
    w.grasp(w.env.target_cube)
    for _ in range(5):
        assert w.tick() == OK
    assert w.stage == 1, "一直抓着只完成第一项「抓起」，放置项不推进"


@pytest.mark.parametrize("tier", TIERS)
def test_grasping_distractor_mid_task_fails(world, tier):
    w = world(tier)
    _cycle(w, 1)
    w.grasp(w.env.distractor_cubes[0])
    assert w.tick() == {"success": False, "fail": True}
