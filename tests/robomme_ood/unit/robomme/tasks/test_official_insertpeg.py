"""InsertPeg native three-tier truth table (C05, C07 geometry).

Geometric criteria (hand-computed placements):
- "pick up by the demonstrated end": grasp-end height > T.PEG_PICKUP_Z, and the tcp is closer to the grasp end than to the insertion end (strict <);
- "insert from the demonstrated side": insertion end to box < T.INSERT_XY and closer than the grasp end, and (tcp_y - box_y)*direction < 0;
  when |tcp_y - box_y| < 1e-3 the side is judged by the grasp end's y instead;
- switching peg, switching end, inserting the wrong end, inserting in reverse -> failure. The demo segment (pick up -> insert -> reset -> still for 100 steps) is also really driven with these placements,
  and the reset step uses the reset_in_proecess branch of the real step to put the three pegs back to their initial poses. All three tiers share the same logic in the official implementation.
"""
from __future__ import annotations

import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

TASK = "InsertPeg"
DIFFS = ("easy", "medium", "hard")
LIFT = T.PEG_LIFT_Z


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _lift_by(ep, grab, other):
    """Pick up: both ends raised, tcp attached to the grab end."""
    gx, gy, _ = grab.xyz
    ox, oy, _ = other.xyz
    grab.move_to(z=LIFT)
    other.move_to(z=LIFT)
    ep.agent.held = grab
    ep.tcp_to(gx, gy, LIFT)


def _insert(ep, insert_end, grab_end, side_sign, gap=0.0):
    """Put the insertion end at the box center (offset by gap), the grasp end T.PEG_END_OFFSET on the side_sign side in y, tcp following the grasp end."""
    bx, by, bz = ep.env.box.xyz
    insert_end.move_to(bx + gap, by, bz)
    grab_end.move_to(bx, by + side_sign * T.PEG_END_OFFSET, bz)
    ep.tcp_to(bx, by + side_sign * T.PEG_END_OFFSET, bz + T.PEG_END_OFFSET)


def _correct_side(env):
    return -env.direction  # (tcp_y − box_y)·direction < 0


def _drive_demo(ep):
    env = ep.env
    _lift_by(ep, env.grasp_target, env.insert_target)
    ep.step()
    assert ep.task_index == 1
    _insert(ep, env.insert_target, env.grasp_target, _correct_side(env))
    ep.step()
    assert ep.task_index == 2
    ep.agent.held = None
    env.reset_in_proecess = True  # flag during solve_strong_reset: the real step puts the pegs back to their initial poses
    ep.step()
    env.reset_in_proecess = False
    guard = 0
    while ep.task_index < ep.first_online_index():
        ep.step()
        guard += 1
        assert guard < 300


@pytest.mark.parametrize("diff", DIFFS)
def test_demo_then_correct_end_and_side_succeeds(world, diff):
    ep = world.make(diff, seed=8)
    env = ep.env
    assert env.insert_way == ("left" if env.direction == -1 else "right")
    assert f"{env.insert_way} side" in env.task_list[-1]["name"]
    assert "same side of the box" in goal_text(env)[0]
    _drive_demo(ep)
    assert not ep.success
    _lift_by(ep, env.grasp_target, env.insert_target)
    ep.step()
    _insert(ep, env.insert_target, env.grasp_target, _correct_side(env))
    ep.step()
    assert ep.success and not ep.fail


def _online(world, diff="easy", seed=8):
    ep = world.make(diff, seed=seed)
    ep.skip_demo()
    return ep, ep.env


@pytest.mark.parametrize("diff", DIFFS)
def test_wrong_end_grasp_fails(world, diff):
    ep, env = _online(world, diff)
    _lift_by(ep, env.insert_target, env.grasp_target)
    ep.step()
    assert ep.fail and not ep.success


def test_other_peg_fails(world):
    ep, env = _online(world)
    other = next(p for p in env.pegs if p is not env.peg)
    _lift_by(ep, other.head, other.tail)
    ep.step()
    assert ep.fail and not ep.success


def test_wrong_side_insert_fails(world):
    ep, env = _online(world)
    _lift_by(ep, env.grasp_target, env.insert_target)
    ep.step()
    _insert(ep, env.insert_target, env.grasp_target, -_correct_side(env))
    ep.step()
    assert ep.fail and not ep.success


def test_wrong_end_insert_fails(world):
    ep, env = _online(world)
    _lift_by(ep, env.grasp_target, env.insert_target)
    ep.step()
    _insert(ep, env.grasp_target, env.insert_target, _correct_side(env))
    ep.step()
    assert ep.fail and not ep.success


def test_equidistant_tcp_does_not_count_as_grasp(world):
    ep, env = _online(world)
    a, b = env.grasp_target, env.insert_target
    a.move_to(z=LIFT)
    b.move_to(z=LIFT)
    mid = (a.xyz + b.xyz) / 2
    ep.tcp_to(*mid)
    ep.step()
    assert ep.task_index == ep.first_online_index() and not ep.fail


@pytest.mark.parametrize("gap, inserted", [(T.INSERT_XY - T.EPS, True), (T.INSERT_XY + T.EPS, False)])
def test_insert_distance_threshold(world, gap, inserted):
    ep, env = _online(world)
    _lift_by(ep, env.grasp_target, env.insert_target)
    ep.step()
    _insert(ep, env.insert_target, env.grasp_target, _correct_side(env), gap=gap)
    ep.step()
    assert ep.success is inserted


@pytest.mark.parametrize("grip_side_ok", [True, False])
def test_direction_near_zero_falls_back_to_grip_end(world, grip_side_ok):
    ep, env = _online(world)
    _lift_by(ep, env.grasp_target, env.insert_target)
    ep.step()
    side = _correct_side(env) if grip_side_ok else -_correct_side(env)
    _insert(ep, env.insert_target, env.grasp_target, side)
    bx, by, bz = env.box.xyz
    ep.tcp_to(bx + T.PEG_END_OFFSET, by + T.DIRECTION_NEAR_ZERO / 2, bz)  # |tcp_y - box_y| < threshold
    ep.step()
    assert ep.success is grip_side_ok and ep.fail is (not grip_side_ok)


def test_reset_branch_restores_peg_poses(world):
    ep = world.make("easy", seed=8)
    env = ep.env
    init = [p.xyz.copy() for p in env.pegs]
    for p in env.pegs:
        p.set_pose(((T.CARRY_HIGH_Z,) * 3, (1, 0, 0, 0)))  # any position far from the initial pose
    env.reset_in_proecess = True
    ep.step()
    for p, x in zip(env.pegs, init):
        assert abs(p.xyz - x).max() < 1e-6
