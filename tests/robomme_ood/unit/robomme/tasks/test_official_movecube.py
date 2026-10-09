"""MoveCube native three-tier truth table (C05): move the cube to the target the demonstrated way -- one of peg hook-push, gripper push, or pick-and-place.

Expectations (the online segment of each of the three ways):
- gripper_push: push the cube to the target (horizontal distance <= T.PUSH_ONTO_XY) with the gripper open (both fingers > T.GRIPPER_OPEN) -> success;
  picking up the cube or the peg -> failure;
- peg_push: first pick up the peg (either end), then hook the cube to the target with the gripper open -> success; cube already at the target before picking the peg -> failure;
- grasp_putdown: pick up the cube and place it on the target (within T.DROP_ONTO_XY) -> success; pushing it to the target without picking up -> failure.
After the demo, the reset_in_proecess branch of the real step moves the cube and target to the second set of poses (demo data does not pollute the online segment).
"""
from __future__ import annotations

import numpy as np
import pytest

from _official_world import OfficialWorld, find_seed
from tests.robomme_ood.unit.robomme import official_thresholds as T

TASK = "MoveCube"
DIFFS = ("easy", "medium", "hard")


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _online(world, diff, way):
    seed = find_seed(TASK, diff, lambda e: e.way == way)
    ep = world.make(diff, seed=seed)
    env = ep.env
    assert env.cube_half_size == pytest.approx(T.CUBE_HALF)  # prerequisite of the pin PUSH_ONTO_XY
    ep.skip_demo()
    env.reset_in_proecess = True  # during solve_strong_reset: the real step moves cube/target to the second set of poses
    ep.step()
    env.reset_in_proecess = False
    np.testing.assert_allclose(env.cube.xyz, env.cube_init_pose_2.p[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(env.goal_site.xyz, env.goal_site_2_pose_p[0], atol=1e-6)
    assert not ep.fail
    return ep, env


def _push_to(ep, env, dx=0.0):
    gx, gy, _ = env.goal_site.xyz
    env.cube.move_to(gx + dx, gy, env.cube_half_size)
    ep.tcp_to(gx + dx - 2 * T.CUBE_HALF, gy, T.TABLE_Z)


@pytest.mark.parametrize("diff", DIFFS)
def test_gripper_push(world, diff):
    ep, env = _online(world, diff, "gripper_push")
    ep.close_gripper()
    _push_to(ep, env)
    ep.step()
    assert not ep.success  # does not count as pushed while the gripper is closed
    ep.open_gripper()
    ep.step()
    assert ep.success and not ep.fail


@pytest.mark.parametrize("dx, ok", [(T.PUSH_ONTO_XY - T.EPS, True), (T.PUSH_ONTO_XY + T.EPS, False)])
def test_push_distance_threshold(world, dx, ok):
    ep, env = _online(world, "easy", "gripper_push")
    _push_to(ep, env, dx=dx)
    ep.step()
    assert ep.success is ok


@pytest.mark.parametrize("diff", DIFFS)
def test_gripper_push_picking_cube_fails(world, diff):
    ep, env = _online(world, diff, "gripper_push")
    ep.grasp(env.cube)
    ep.step()
    assert ep.fail and not ep.success


def test_gripper_push_picking_peg_fails(world):
    ep, env = _online(world, "easy", "gripper_push")
    ep.grasp(env.peg_tail)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_peg_push(world, diff):
    ep, env = _online(world, diff, "peg_push")
    ep.grasp(env.peg_head)  # either end counts as picking up the peg
    ep.step()
    assert ep.task_index == ep.first_online_index() + 1
    ep.close_gripper()
    _push_to(ep, env)
    ep.step()
    assert not ep.success  # still holding the peg (gripper closed): not complete
    ep.open_gripper()
    ep.step()
    assert ep.success and not ep.fail


def test_peg_push_cube_on_goal_before_peg_fails(world):
    ep, env = _online(world, "easy", "peg_push")
    _push_to(ep, env)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_grasp_putdown(world, diff):
    ep, env = _online(world, diff, "grasp_putdown")
    ep.grasp(env.cube)
    ep.step()
    ep.place_on(env.cube, env.goal_site)
    ep.step()
    assert ep.success and not ep.fail


def test_grasp_putdown_pushing_fails(world):
    ep, env = _online(world, "easy", "grasp_putdown")
    _push_to(ep, env)
    ep.step()
    assert ep.fail and not ep.success


def test_grasp_putdown_picking_peg_fails(world):
    ep, env = _online(world, "easy", "grasp_putdown")
    ep.grasp(env.peg_head)
    ep.step()
    assert ep.fail and not ep.success


def test_three_ways_reachable_in_every_difficulty():
    for diff in DIFFS:
        for way in ("peg_push", "gripper_push", "grasp_putdown"):
            find_seed(TASK, diff, lambda e, w=way: e.way == w)
