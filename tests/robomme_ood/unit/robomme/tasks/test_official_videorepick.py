"""VideoRepick native three-tier truth table (C05, C06): the cube picked and placed in the demo is picked and placed N more times by identity after swaps, then the button is pressed.

The whole demo segment is driven by the real evaluate/step: demo pick up -> put down -> (easy/medium) still 20 steps -> swap (real swap_flat_two_lane)
-> reset check; hard has no swap. Expectations:
- in the online segment, N times "pick up the same cube -> put down" then press the button -> success; the one in the demo does not count toward N;
- picking another cube (including the one now occupying its original position after swaps) -> failure;
- pressing the button during the pick-and-place subtask: within the timing window T.REPICK_BUTTON_WINDOW -> failure; before the window starts it does not fail (current official behavior).
"""
from __future__ import annotations

import numpy as np
import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

TASK = "VideoRepick"
DIFFS = ("easy", "medium", "hard")
WORDS = {2: "two", 3: "three"}


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _drive_demo(ep):
    env = ep.env
    target = env.target_cube_1
    ep.grasp(target)
    ep.step()
    ep.release(target)
    ep.step()
    drop_xy = target.xyz[:2].copy()
    guard = 0
    while ep.task_index < ep.first_online_index():
        ep.step()
        guard += 1
        assert guard < 1000
    ep.step()  # past the positioning of the last swap at the start of the next step
    return drop_xy


def _cycle(ep, cube):
    ep.grasp(cube)
    ep.step()
    ep.release(cube)
    ep.step()


@pytest.mark.parametrize("diff", DIFFS)
def test_repick_same_cube_n_times_then_button_succeeds(world, diff):
    ep = world.make(diff, seed=6)
    env = ep.env
    n = env.num_repeats
    goals = goal_text(env)
    if n > 1:
        assert any(f"{WORDS[n]} times" in g or (n == 2 and "twice" in g) for g in goals), goals
    else:
        assert all("again" in g for g in goals), goals
    _drive_demo(ep)
    assert not ep.success and not ep.fail
    for _ in range(n):
        _cycle(ep, env.target_cube_1)
    assert not ep.success
    ep.press(env.button_left)
    ep.step()
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", ("easy", "medium"))
def test_swap_happens_and_involves_target(world, diff):
    ep = world.make(diff, seed=6)
    env = ep.env
    assert env.swap_times >= 1
    _drive_demo(ep)
    a, b = env.swap_schedule[0][0], env.swap_schedule[0][1]
    assert a is env.target_cube_1 and b is not None and b is not a  # the first swap pair always includes the target cube
    # after several swaps the target cube may be swapped back, so "swapped away" is picked by layout only in the next test


@pytest.mark.parametrize("diff", ("easy", "medium"))
def test_cube_at_original_position_is_wrong(world, diff):
    for seed in range(40):
        ep = world.make(diff, seed=seed)
        env = ep.env
        drop_xy = _drive_demo(ep)
        impostor = min(env.spawned_cubes, key=lambda c: np.linalg.norm(c.xyz[:2] - drop_xy))
        if impostor is not env.target_cube_1:
            break
    else:
        pytest.fail("the target cube was never swapped away within 40 seeds")
    ep.grasp(impostor)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_demo_pick_not_counted(world, diff):
    """Only N-1 online pick-and-place cycles: pressing the button within the timing window -> failure (the demo one does not count)."""
    ep = world.make(diff, seed=6)
    env = ep.env
    _drive_demo(ep)
    start = int(env.elapsed_steps)
    for _ in range(env.num_repeats - 1):
        _cycle(ep, env.target_cube_1)
    ep.step(max(0, T.REPICK_BUTTON_WINDOW[0] + 10 - (int(env.elapsed_steps) - start)))
    ep.press(env.button_left)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("wait, fails", [(T.REPICK_BUTTON_WINDOW[0] - 10, False), (T.REPICK_BUTTON_WINDOW[0] + 10, True)])
def test_button_timewindow_during_repick(world, wait, fails):
    ep = world.make("hard", seed=6)
    env = ep.env
    _drive_demo(ep)
    ep.step(wait)
    ep.press(env.button_left)
    ep.step()
    assert ep.fail is fails and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_wrong_cube_fails(world, diff):
    ep = world.make(diff, seed=6)
    env = ep.env
    _drive_demo(ep)
    other = next(c for c in env.spawned_cubes if c is not env.target_cube_1)
    ep.grasp(other)
    ep.step()
    assert ep.fail and not ep.success


def test_hard_has_no_swap_and_15_cubes(world):
    ep = world.make("hard", seed=6)
    env = ep.env
    assert env.swap_times == 0
    assert len(env.spawned_cubes) == 15  # 5 rounds x 3 colors
    assert [t.get("specialflag") for t in env.task_list].count("swap") == 0
