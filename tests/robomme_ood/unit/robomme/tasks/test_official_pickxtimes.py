"""PickXtimes native three-tier truth table (C05): success only after N complete pick-and-place cycles followed by a button press.

Expectations come from the task definition itself (goal language "repeating this action N times, then press the button"):
N pick-and-place + button -> success; pressing after N-1, an extra N+1th pick, grasping the wrong object, pressing while holding -> failure; N done without pressing -> ongoing.
N is read from this episode's env.num_repeats and cross-checked against the count word in the goal language (language and verdict bound to the same episode).
"""
from __future__ import annotations

import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

TASK = "PickXtimes"
DIFFS = ("easy", "medium", "hard")
# independent expectation: English cardinal numbers (only for checking the count in the goal language; source: English)
WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _cycle(ep, env):
    ep.grasp(env.target_cube)
    ep.step()
    ep.place_on(env.target_cube, env.target)
    ep.step()


@pytest.mark.parametrize("diff", DIFFS)
def test_n_cycles_then_button_succeeds(world, diff):
    ep = world.make(diff, seed=3)
    env = ep.env
    n = env.num_repeats
    goals = goal_text(env)
    if n > 1:
        assert all(WORDS[n] in g for g in goals), goals
    else:
        assert all("repeating" not in g for g in goals), goals
    for _ in range(n):
        _cycle(ep, env)
        assert not ep.success and not ep.fail
    ep.step()
    assert not ep.success and not ep.fail  # N done without pressing the button: not finished, no success
    ep.press(env.button)
    ep.step()
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", DIFFS)
def test_button_after_n_minus_1_cycles_fails(world, diff):
    ep = world.make(diff, seed=3)
    env = ep.env
    for _ in range(env.num_repeats - 1):
        _cycle(ep, env)
    ep.press(env.button)
    ep.step()
    assert ep.fail and not ep.success
    ep.unpress(env.button)
    ep.step(3)
    assert ep.fail and not ep.success  # failure terminal state is stable


@pytest.mark.parametrize("diff", DIFFS)
def test_extra_pickup_after_n_cycles_fails(world, diff):
    ep = world.make(diff, seed=3)
    env = ep.env
    for _ in range(env.num_repeats):
        _cycle(ep, env)
    ep.grasp(env.target_cube)  # the N+1th pickup: the failure condition of the button subtask
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", ("medium", "hard"))
def test_wrong_object_fails(world, diff):
    ep = world.make(diff, seed=3)
    env = ep.env
    assert env.non_target_cubes, "medium/hard have distractor cubes"
    ep.grasp(env.non_target_cubes[0])
    ep.step()
    assert ep.fail and not ep.success


def test_easy_has_single_color_no_distractor(world):
    ep = world.make("easy", seed=3)
    assert ep.env.non_target_cubes == []
    assert len(ep.env.all_cubes) == 1


@pytest.mark.parametrize("diff", DIFFS)
def test_press_button_while_holding_fails(world, diff):
    ep = world.make(diff, seed=3)
    env = ep.env
    for _ in range(env.num_repeats - 1):
        _cycle(ep, env)
    ep.grasp(env.target_cube)
    ep.step()
    assert not ep.fail
    ep.press(env.button)  # pressing the button while holding (in the "place on target" subtask)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_place_off_target_does_not_advance(world, diff):
    ep = world.make(diff, seed=3)
    env = ep.env
    ep.grasp(env.target_cube)
    ep.step()
    before = ep.task_index
    tx, ty, _ = env.target.xyz
    ep.release(env.target_cube, tx + 4 * T.DROP_ONTO_XY, ty)  # far outside the placement threshold
    ep.step()
    assert ep.task_index == before and not ep.success and not ep.fail


def test_drop_distance_boundary(world):
    """Horizontal distance threshold T.DROP_ONTO_XY of is_obj_dropped_onto (<= check): advances just inside, not just outside."""
    for offset, advances in ((T.DROP_ONTO_XY - T.EPS, True), (T.DROP_ONTO_XY + T.EPS, False)):
        ep = world.make("easy", seed=3)
        env = ep.env
        ep.grasp(env.target_cube)
        ep.step()
        before = ep.task_index
        tx, ty, _ = env.target.xyz
        ep.release(env.target_cube, tx + offset, ty)
        ep.step()
        assert (ep.task_index == before + 1) is advances, offset


def test_rebuild_clears_episode_state(world):
    """After rebuilding (a new episode) the subtask pointer and failure flag start from zero."""
    ep = world.make("easy", seed=3)
    ep.press(ep.env.button)
    ep.step()
    assert ep.fail
    ep2 = world.make("easy", seed=3)
    assert ep2.task_index == 0
    ep2.step()
    assert not ep2.fail and not ep2.success
