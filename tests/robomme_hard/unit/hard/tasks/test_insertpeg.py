"""InsertPeg 新值档真值表（V9 只交付 xhard4）：执行段抓起演示里那根杆的同一端，从演示的那一侧把另一端插进孔。

演示段（抓起、插入、复位、静止 100 步）在 CPU 世界里按任务表对象走完；执行段：
正例 = 抬起抓取端（离夹爪更近）→ 插入端到孔中心、夹爪在要求的一侧 → 成功；
错误与边界：抓另一端即失败；夹爪在反方向一侧插入即失败；抓别的杆即失败；两端离夹爪等距不算抓起（严格 ``<``）。
"""
from __future__ import annotations

import pytest

from robomme_hard.robomme_env.utils import reset_panda

from .. import offline_scene as O
from ..world import World, cpu_world

TASK = "InsertPeg"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}
LIFT = 0.15


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier, k=0: World.build(TASK, tier, k)


def _lift_end(w, end, other_z=0.12):
    """抬起杆的一端（夹爪就在这一端），另一端略低。"""
    x, y, _ = w.xyz(end)
    w.move(end, (x, y, LIFT))
    w.agent.held = end
    w.tcp_to((x, y, LIFT))


def _insert(w, side_sign):
    """插入：插入端移到孔中心，抓取端在孔外 0.1 m；夹爪在孔的 y 侧 ``side_sign`` 方向。"""
    env = w.env
    bx, by, bz = w.xyz(env.box)
    w.move(env.insert_target, (bx, by, bz))
    w.move(env.grasp_target, (bx, by + side_sign * 0.1, bz))
    w.tcp_to((bx, by + side_sign * 0.1, bz))


def _demo(w):
    env = w.env
    w.still()
    for _ in range(600):
        task = env.task_list[w.stage]
        if not task["demonstration"]:
            return
        name = task["name"]
        if name.startswith("Pick up the peg"):
            _lift_end(w, env.grasp_target)
        elif name.startswith("Insert the peg"):
            _insert(w, -env.direction)
        else:  # 复位与静止
            w.agent.held = None
            w.agent.robot.set_qpos(reset_panda.get_reset_panda_param("qpos"))
        assert w.step()["fail"] is False, name
    raise AssertionError("演示段未走完")


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("k", range(3))
def test_same_end_correct_side_succeeds(world, tier, k):
    w = world(tier, k)
    _demo(w)
    _lift_end(w, w.env.grasp_target)
    assert w.step() == OK
    _insert(w, -w.env.direction)
    assert w.step() == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS)
def test_grasping_the_other_end_fails(world, tier):
    w = world(tier)
    _demo(w)
    _lift_end(w, w.env.insert_target)
    assert w.step() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS)
def test_inserting_from_the_wrong_side_fails(world, tier):
    w = world(tier)
    _demo(w)
    _lift_end(w, w.env.grasp_target)
    w.step()
    _insert(w, +w.env.direction)
    assert w.step() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS)
def test_other_peg_and_equidistant_ends(world, tier):
    w = world(tier)
    _demo(w)
    other = next(i for i, p in enumerate(w.env.pegs) if p is not w.env.peg)
    _lift_end(w, w.env.peg_heads[other])
    assert w.step() == {"success": False, "fail": True}
    w = World.build(TASK, tier)
    _demo(w)
    # 两端关于夹爪对称、坐标取 float32 精确可表示的 ±1/16 m：两端到夹爪等距，严格 < 不成立 → 不算抓起
    w.move(w.env.grasp_target, (2.0 ** -4, 0.0, LIFT))
    w.move(w.env.insert_target, (-(2.0 ** -4), 0.0, LIFT))
    w.tcp_to((0.0, 0.0, LIFT))
    stage = w.stage
    assert w.step() == OK and w.stage == stage
