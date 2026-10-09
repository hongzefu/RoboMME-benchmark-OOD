"""InsertPeg new-value tier truth table (V9 delivers only xhard4): in the execution segment pick up the same end of the peg used in the demonstration and insert the other end into the hole from the demonstrated side.

The demonstration segment (pick up, insert, reset, stay still 100 steps) is walked through in the CPU world following the task table objects; execution segment:
positive = lift the grasp end (closer to the gripper) → insertion end to the hole center with the gripper on the required side → success;
errors and boundaries: grasping the other end fails; inserting with the gripper on the opposite side fails; grasping another peg fails; both ends equidistant from the gripper does not count as picked up (strict ``<``).
"""
from __future__ import annotations

import pytest

from robomme_ood.robomme_env.utils import reset_panda

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
    """Lift one end of the peg (the gripper is at this end), the other end slightly lower."""
    x, y, _ = w.xyz(end)
    w.move(end, (x, y, LIFT))
    w.agent.held = end
    w.tcp_to((x, y, LIFT))


def _insert(w, side_sign):
    """Insert: move the insertion end to the hole center, the grasp end 0.1 m outside the hole; the gripper is on the ``side_sign`` y side of the hole."""
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
        else:  # reset and stay still
            w.agent.held = None
            w.agent.robot.set_qpos(reset_panda.get_reset_panda_param("qpos"))
        assert w.step()["fail"] is False, name
    raise AssertionError("demonstration segment not finished")


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
    # both ends symmetric about the gripper, coordinates ±1/16 m exactly representable in float32: both ends equidistant from the gripper, strict < does not hold → not picked up
    w.move(w.env.grasp_target, (2.0 ** -4, 0.0, LIFT))
    w.move(w.env.insert_target, (-(2.0 ** -4), 0.0, LIFT))
    w.tcp_to((0.0, 0.0, LIFT))
    stage = w.stage
    assert w.step() == OK and w.stage == stage
