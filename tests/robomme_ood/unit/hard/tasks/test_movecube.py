"""MoveCube new-value tier truth table (V9 delivers only xhard4): in the execution segment move the cube to the goal using the one method the demonstration specified (pick-and-place/gripper push/stick push).

Each of the three methods uses the first packaged formal episode that uses it (method selected by the spec's ``initializations.1.way_idx``, cross-checked with the env's actual ``way``).
The demonstration segment is walked through in the CPU world; on reset the cube and goal switch to the execution layout (production step's reset logic).
Errors and boundaries: under pick-and-place, pushing directly to the goal or grasping the peg fails; under push, picking up the cube fails; under stick push, pushing the cube first fails;
reaching the goal requires the gripper open; inside/outside the distance threshold.
"""
from __future__ import annotations

import functools

import pytest

from robomme_ood.robomme_env.utils import reset_panda

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
    raise AssertionError(f"no packaged formal episode uses {way} as its method")


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
    """Demonstration segment: walk through once with this episode's method, then emulate the reset (one reset_in_proecess step), stopping at the first execution item."""
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
    raise AssertionError("demonstration segment not finished")


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
    _onto_goal(w, w.env.cube)  # pushed to the goal without grasping
    assert w.step() == {"success": False, "fail": True}
    w = world("grasp_putdown")
    _demo(w)
    w.grasp(w.env.grasp_target)
    assert w.step() == {"success": False, "fail": True}


def test_gripper_push_way_and_open_gripper_requirement(world):
    w = world("gripper_push")
    _demo(w)
    q = w.agent.robot.get_qpos().clone()
    q[0, -2:] = 0.0  # gripper closed: reaching the goal does not count either (must_gripper_open)
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
    _onto_goal(w, w.env.cube)  # push the cube first (without taking the stick)
    assert w.step() == {"success": False, "fail": True}
    w = world("peg_push")
    _demo(w)
    w.grasp(w.env.grasp_target)
    w.step()
    w.agent.held = None
    # inside/outside distance (without replicating production's threshold formula): two cube edge lengths from the goal center does not count as reached, half a cube edge length does
    edge = 2 * w.env.cube_half_size
    _onto_goal(w, w.env.cube, offset=2 * edge)
    assert w.step() == OK
    _onto_goal(w, w.env.cube, offset=0.5 * edge)
    assert w.step() == {"success": True, "fail": False}
