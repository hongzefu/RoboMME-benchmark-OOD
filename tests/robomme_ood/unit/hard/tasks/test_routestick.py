"""RouteStick 新值档真值表（xhard1～xhard3）：执行段按演示的落点序列、每段从演示规定的一侧绕过杆。

方向判据：本段 TCP 轨迹相对「上一落点 → 本落点」连线的平均叉积，正为 clockwise、负为 counterclockwise
（生产 ``direction_fail``）。测试在连线中点外侧放一个途经点（高于全部高度阈值），侧向由手算法向量给出。
错误与边界：落点对但绕向反即失败；途经点恰在连线上（叉积为零）即失败；失败锁存；跳到别的落点即失败。
"""
from __future__ import annotations

import numpy as np
import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import stick_driver as SD

TASK = "RouteStick"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier, k=0: World.build(TASK, tier, k)


def _via(w, prev, cur, side):
    """途经点：连线中点沿左法向 (−ly, lx) 偏移 side×0.1 m；side=+1 叉积为正（clockwise）。"""
    p, c = w.xyz(prev)[:2], w.xyz(cur)[:2]
    line = c - p
    n = np.array([-line[1], line[0]]) / np.linalg.norm(line)
    return (p + c) / 2 + side * 0.1 * n


VIA_STEPS = 8  # 途经点停留步数：轨迹点以途经点为主（复位那一刻的 TCP 点也在第一段轨迹里，见交回说明）


def _segment(w, prev, cur, side):
    SD.hover(w, _via(w, prev, cur, side))
    for _ in range(VIA_STEPS):
        out = w.step()
        if out["fail"]:
            return out
    SD.touch(w, cur)
    return w.step()


def _sign(direction):
    return {"clockwise": +1, "counterclockwise": -1}[direction]


@pytest.mark.parametrize("tier", TIERS)
def test_correct_nodes_and_sides_succeed(world, tier):
    w = world(tier)
    SD.run_demo(w)
    path, dirs = w.env.selected_buttons, w.env.swing_directions
    out = None
    for i, cur in enumerate(path[1:]):
        out = _segment(w, path[i], cur, _sign(dirs[i]))
        if i < len(dirs) - 1:
            assert out == OK, i
    assert out == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS)
def test_reversed_side_fails_and_latches(world, tier):
    w = world(tier)
    SD.run_demo(w)
    path, dirs = w.env.selected_buttons, w.env.swing_directions
    assert _segment(w, path[0], path[1], -_sign(dirs[0]))["fail"] is True
    SD.hover(w, w.xyz(path[1])[:2])
    assert w.step()["fail"] is True, "失败锁存"


@pytest.mark.parametrize("tier", TIERS[:1])
def test_path_on_the_line_fails(world, tier):
    """第二段整段沿连线走（叉积为零）→ 失败。用第二段：第一段的轨迹缓存里还留着复位那一刻的 TCP 点
    （生产 ``_gripper_xy_trace`` 在执行段开始时不清空，见交回的生产问题），第一段之后缓存才被清空。"""
    w = world(tier)
    SD.run_demo(w)
    path, dirs = w.env.selected_buttons, w.env.swing_directions
    assert _segment(w, path[0], path[1], _sign(dirs[0])) == OK
    assert _segment(w, path[1], path[2], 0)["fail"] is True


@pytest.mark.parametrize("tier", TIERS[:1])
def test_jumping_to_a_wrong_node_fails(world, tier):
    w = world(tier)
    SD.run_demo(w)
    path = w.env.selected_buttons
    wrong = next(b for b in w.env.buttons_grid if b is not path[0] and b is not path[1])
    SD.touch(w, wrong)
    assert w.step()["fail"] is True
