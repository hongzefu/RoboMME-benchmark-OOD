"""SwingXtimes native three-tier truth table (C05): pick up -> swing right->left N rounds -> put down -> button.

Key verdicts (expectations from the task definition and hand-written event sequences):
- "above the target" = horizontal distance <= T.SWING_ENTER_XY and height < T.SWING_ENTER_Z;
- each "entry" into the right/left target counts one swing, lingering is not recounted (hysteresis of the exit threshold T.SWING_EXIT_XY); total swings > 2N fails;
- reversed left/right order does not advance the subtask; not putting down before the button, or picking a distractor cube -> failure.
"""
from __future__ import annotations

import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

TASK = "SwingXtimes"
DIFFS = ("easy", "medium", "hard")
SWING_Z = T.LIFT_Z  # < T.SWING_ENTER_Z


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _over(ep, env, target, dx=0.0, z=SWING_Z):
    x, y, _ = target.xyz
    ep.carry(env.target_cube, x + dx, y, z)
    ep.step()


def _away(ep, env):
    ep.carry(env.target_cube, 0.0, 0.0, T.CARRY_HIGH_Z)
    ep.step()


def _full_rounds(ep, env, rounds):
    for _ in range(rounds):
        _over(ep, env, env.target_right)
        _over(ep, env, env.target_left)


def _finish(ep, env):
    _away(ep, env)
    ep.release(env.target_cube, 0.0, 0.0)
    ep.step()
    ep.press(env.button)
    ep.step()


@pytest.mark.parametrize("diff", DIFFS)
def test_n_round_trips_then_putdown_and_button_succeeds(world, diff):
    ep = world.make(diff, seed=5)
    env = ep.env
    assert env.target_right.xyz[1] < env.target_left.xyz[1]  # "right" is the one with smaller y
    n = env.num_repeats
    if n > 1:
        assert any(f"{['', 'one', 'two', 'three'][n]} times" in g for g in goal_text(env))
    ep.grasp(env.target_cube)
    ep.step()
    _full_rounds(ep, env, n)
    assert env.swing_count == 2 * n
    assert not ep.success and not ep.fail
    _finish(ep, env)
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", DIFFS)
def test_left_before_right_does_not_advance_and_overflows(world, diff):
    ep = world.make(diff, seed=5)
    env = ep.env
    ep.grasp(env.target_cube)
    ep.step()
    idx = ep.task_index
    _over(ep, env, env.target_left)  # reversed: left first
    assert ep.task_index == idx and not ep.fail
    _away(ep, env)
    _full_rounds(ep, env, env.num_repeats)  # then complete N rounds as usual: total swings 2N+1
    assert env.swing_count == 2 * env.num_repeats + 1
    assert env.swing_over_limit
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_dwelling_counts_once(world, diff):
    ep = world.make(diff, seed=5)
    env = ep.env
    ep.grasp(env.target_cube)
    ep.step()
    for _ in range(5):
        _over(ep, env, env.target_right)
    assert env.swing_count == 1
    # beyond the entry threshold but wobbling within the exit threshold: still counts as lingering
    _over(ep, env, env.target_right, dx=T.SWING_EXIT_XY - T.EPS)
    assert env.swing_count == 1
    # really leaving and coming back: counted again
    _away(ep, env)
    _over(ep, env, env.target_right)
    assert env.swing_count == 2


@pytest.mark.parametrize("dx, z, advances", [
    (T.SWING_ENTER_XY - T.EPS, SWING_Z, True), (T.SWING_ENTER_XY + T.EPS, SWING_Z, False),
    (0.0, T.SWING_ENTER_Z - T.EPS, True), (0.0, T.SWING_ENTER_Z + T.EPS, False),
])
def test_swing_thresholds(world, dx, z, advances):
    ep = world.make("easy", seed=5)
    env = ep.env
    ep.grasp(env.target_cube)
    ep.step()
    idx = ep.task_index
    _over(ep, env, env.target_right, dx=dx, z=z)
    assert (ep.task_index == idx + 1) is advances


@pytest.mark.parametrize("diff", DIFFS)
def test_button_without_putdown_fails(world, diff):
    ep = world.make(diff, seed=5)
    env = ep.env
    ep.grasp(env.target_cube)
    ep.step()
    _full_rounds(ep, env, env.num_repeats)
    ep.press(env.button)  # still in the "put down" subtask; the button is a failure condition
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", ("medium", "hard"))
def test_pick_distractor_fails(world, diff):
    ep = world.make(diff, seed=5)
    env = ep.env
    ep.grasp(env.non_target_cubes[0])
    ep.step()
    assert ep.fail and not ep.success


def test_too_many_rounds_fails(world):
    ep = world.make("easy", seed=5)
    env = ep.env
    ep.grasp(env.target_cube)
    ep.step()
    _full_rounds(ep, env, env.num_repeats + 1)
    ep.step()
    assert ep.fail and not ep.success
