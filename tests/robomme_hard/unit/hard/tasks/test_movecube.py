"""MoveCube 新值档真值表（V9 只交付 xhard4）：执行段用演示指定的那一种方式（抓放／夹爪推／用杆推）把方块移到目标。

三种方式各取包内第一个用到它的正式局（方式由规格 ``initializations.1.way_idx`` 选出，与环境实际 ``way`` 核对）。
演示段在 CPU 世界里走完，复位时方块与目标换到执行段布局（生产 step 的复位逻辑）。
错误与边界：抓放方式下直接推到目标、抓杆即失败；推的方式下抓起方块即失败；杆推方式下先推方块即失败；
推到判定含夹爪须张开；距离阈值内外。
"""
from __future__ import annotations

import functools

import pytest

from robomme_hard.robomme_env.utils import reset_panda

from .. import offline_scene as O
from ..world import World, cpu_world

TASK = "MoveCube"
TIER = O.tiers_of(TASK)[0]
OK = {"success": False, "fail": False}


@functools.lru_cache(maxsize=None)
def _row_for_way(way):
    header, rows = O.delivered_rows(TASK, TIER)
    with cpu_world():
        ways = World.build(TASK, TIER, 0).env.ways
    for k, row in enumerate(rows):
        if ways[row["spec"]["initializations"]["1"]["way_idx"]] == way:
            return k
    raise AssertionError(f"包内没有用 {way} 的正式局")


@pytest.fixture
def world():
    with cpu_world():
        def make(way):
            w = World.build(TASK, TIER, _row_for_way(way))
            assert w.env.way == way
            return w
        yield make


def _onto_goal(w, obj, offset=0.0):
    gx, gy, _ = w.xyz(w.env.goal_site)
    w.move(obj, (gx + offset, gy, 0.02))


def _demo(w):
    """演示段：按本局方式走一遍，然后模拟复位（reset_in_proecess 一步），停在执行段第一项。"""
    env = w.env
    w.still()
    for _ in range(400):
        task = env.task_list[w.stage] if hasattr(env, "task_list") else None
        if task is not None and not task["demonstration"]:
            return
        name = task["name"] if task else ""
        if name == "Pick up the cube":
            w.grasp(env.cube)
        elif name == "place the cube onto the target":
            gx, gy, _ = w.xyz(env.goal_site)
            w.release_onto(env.cube, (gx, gy))
        elif name == "Pick up the peg":
            w.grasp(env.grasp_target)
        elif name.startswith(("Hook the cube", "Close the gripper")):
            w.agent.held = None
            _onto_goal(w, env.cube)
        elif name == "NO RECORD":
            w.agent.robot.set_qpos(reset_panda.get_reset_panda_param("qpos"))
            env.reset_in_proecess = True
            w.step()
            env.reset_in_proecess = False
            continue
        assert w.step()["fail"] is False, name
    raise AssertionError("演示段未走完")


def test_grasp_putdown_way(world):
    w = world("grasp_putdown")
    _demo(w)
    w.grasp(w.env.cube)
    assert w.step() == OK
    gx, gy, _ = w.xyz(w.env.goal_site)
    w.release_onto(w.env.cube, (gx, gy))
    assert w.step() == {"success": True, "fail": False}


def test_grasp_putdown_rejects_push_and_peg(world):
    w = world("grasp_putdown")
    _demo(w)
    _onto_goal(w, w.env.cube)  # 没抓就推到目标
    assert w.step() == {"success": False, "fail": True}
    w = world("grasp_putdown")
    _demo(w)
    w.grasp(w.env.grasp_target)
    assert w.step() == {"success": False, "fail": True}


def test_gripper_push_way_and_open_gripper_requirement(world):
    w = world("gripper_push")
    _demo(w)
    q = w.agent.robot.get_qpos().clone()
    q[0, -2:] = 0.0  # 夹爪闭合：推到目标也不算（must_gripper_open）
    w.agent.robot.set_qpos(q)
    _onto_goal(w, w.env.cube)
    assert w.step() == OK
    w.agent.robot.set_qpos(reset_panda.get_reset_panda_param("qpos"))
    assert w.step() == {"success": True, "fail": False}


def test_gripper_push_rejects_grasping_cube(world):
    w = world("gripper_push")
    _demo(w)
    w.grasp(w.env.cube)
    assert w.step() == {"success": False, "fail": True}


def test_peg_push_way(world):
    w = world("peg_push")
    _demo(w)
    w.grasp(w.env.grasp_target)
    assert w.step() == OK
    w.agent.held = None
    _onto_goal(w, w.env.cube)
    assert w.step() == {"success": True, "fail": False}


def test_peg_push_rejects_cube_first_and_distance_boundary(world):
    w = world("peg_push")
    _demo(w)
    _onto_goal(w, w.env.cube)  # 先推方块（没拿杆）
    assert w.step() == {"success": False, "fail": True}
    w = world("peg_push")
    _demo(w)
    w.grasp(w.env.grasp_target)
    w.step()
    w.agent.held = None
    # 距离内外（不复刻生产阈值公式）：离目标中心两个方块边长不算推到，半个方块边长算推到
    edge = 2 * w.env.cube_half_size
    _onto_goal(w, w.env.cube, offset=2 * edge)
    assert w.step() == OK
    _onto_goal(w, w.env.cube, offset=0.5 * edge)
    assert w.step() == {"success": True, "fail": False}
