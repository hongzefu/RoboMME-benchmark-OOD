"""VideoUnmaskSwap and ButtonUnmaskSwap native three-tier truth tables (C05, C06 scene motion).

- the swap animation uses the real ``swap_flat_two_lane`` (actor doubles only provide pose/set_pose): at the end of each swap the two containers have exactly exchanged positions,
  the other containers stay put; after all swaps the container positions are a permutation of the initial positions.
- verdicts are by "container identity", not "original position": only following the container that hides the cube is correct; picking up the other one now occupying the original position -> failure.
- ButtonUnmaskSwap: each of the two buttons pressed once in any order (pressed ones are removed from the list); pressing the same button twice does not advance;
  both buttons pressed on the same step -> the first subtask consumes both buttons at once and the second can never complete (current official behavior, registered conditional).
"""
from __future__ import annotations

import numpy as np
import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

DIFFS = ("easy", "medium", "hard")
BIN_UP = T.BIN_UP_Z


def _xy(a):
    return a.xyz[:2].copy()


def _run_until(ep, pre_elapsed):
    """Step one at a time until "elapsed at the start of the next step" == pre_elapsed."""
    while int(ep.env.elapsed_steps) < pre_elapsed:
        ep.step()


def _demo_done_vus(ep):
    """Run through the demo segment (watching still until the last swap ends); then step further until elapsed > swap end step,
    so that the last swap's positioning "at the start of the next step" lands and online actions are not overwritten by it."""
    guard = 0
    while ep.task_index < 1:
        ep.step()
        guard += 1
        assert guard < 600
    assert int(ep.env.elapsed_steps) >= _swap_end(ep.env)  # exact switch timing see test_vus_still_demo_one_step_before_swap_end
    _run_until(ep, _swap_end(ep.env) + 1)


def _swap_end(env):
    return int(env.swap_schedule[-1][3])


def _pick_sequence(ep, env):
    ep.grasp(env.selected_bins[0], z=BIN_UP)
    ep.step()
    if env.pick_times == 2:
        ep.release(env.selected_bins[0])
        ep.step()
        ep.grasp(env.selected_bins[1], z=BIN_UP)
        ep.step()


# --------------------------------------------------------------------------- VideoUnmaskSwap


@pytest.fixture
def vus():
    with OfficialWorld("VideoUnmaskSwap") as w:
        yield w


@pytest.mark.parametrize("diff", DIFFS)
def test_vus_swap_mechanics(vus, diff):
    ep = vus.make(diff, seed=4)
    env = ep.env
    n_swaps = len(env.swap_schedule)
    assert n_swaps == env.swap_times
    origin = {id(b): _xy(b) for b in env.spawned_bins}
    start1, end1 = int(env.swap_schedule[0][2]), int(env.swap_schedule[0][3])
    _run_until(ep, start1)
    snap = {id(b): _xy(b) for b in env.spawned_bins}
    for b in env.spawned_bins:  # all back in place after the reveal animation ends
        np.testing.assert_allclose(snap[id(b)], origin[id(b)], atol=1e-6)
    _run_until(ep, end1 + 1)
    a, b = env.swap_schedule[0][0], env.swap_schedule[0][1]
    assert a is not None and b is not None and a is not b
    np.testing.assert_allclose(_xy(a), snap[id(b)], atol=1e-5)
    np.testing.assert_allclose(_xy(b), snap[id(a)], atol=1e-5)
    if n_swaps == 1:
        for o in env.spawned_bins:
            if o is not a and o is not b:
                np.testing.assert_allclose(_xy(o), snap[id(o)], atol=1e-6)
    _demo_done_vus(ep)
    final = sorted(map(tuple, np.round([_xy(o) for o in env.spawned_bins], 5)))
    initial = sorted(map(tuple, np.round(list(origin.values()), 5)))
    # swaps are only a permutation (endpoint capture of consecutive swaps drifts sub-millimeter; container spacing >= 0.1 m, so a 1 mm tolerance suffices)
    np.testing.assert_allclose(final, initial, atol=1e-3)


@pytest.mark.parametrize("diff", DIFFS)
def test_vus_follow_identity_succeeds(vus, diff):
    ep = vus.make(diff, seed=4)
    env = ep.env
    goals = goal_text(env)
    assert all(f"hiding the {env.color_names[0]} cube" in g for g in goals)
    _demo_done_vus(ep)
    _pick_sequence(ep, env)
    assert ep.success and not ep.fail


def test_vus_original_position_is_wrong():
    """Find a layout where the target container was swapped away: picking up the one now occupying its original position -> failure."""
    with OfficialWorld("VideoUnmaskSwap") as world:
        for seed in range(40):
            ep = world.make("hard", seed=seed)
            env = ep.env
            target = env.selected_bins[0]
            start_xy = _xy(target)
            _demo_done_vus(ep)
            impostor = min(env.spawned_bins, key=lambda o: np.linalg.norm(_xy(o) - start_xy))
            if impostor is not target:
                break
        else:
            pytest.fail("the target container was never swapped away within 40 seeds")
        ep.grasp(impostor, z=BIN_UP)
        ep.step()
        assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", ("medium", "hard"))
def test_vus_empty_distractor_container_fails(vus, diff):
    ep = vus.make(diff, seed=4)
    env = ep.env
    assert len(env.spawned_bins) == 4
    empty = [b for b in env.spawned_bins if all(b is not s for s in env.selected_bins)]
    assert len(empty) == 1  # 4 containers hide only 3 cubes
    _demo_done_vus(ep)
    ep.grasp(empty[0], z=BIN_UP)
    ep.step()
    assert ep.fail and not ep.success


def test_vus_two_picks_reversed_fails(vus):
    ep = vus.make("hard", seed=4)
    env = ep.env
    assert env.pick_times == 2
    _demo_done_vus(ep)
    ep.grasp(env.selected_bins[1], z=BIN_UP)
    ep.step()
    assert ep.fail and not ep.success


def test_vus_still_demo_one_step_before_swap_end(vus):
    ep = vus.make("easy", seed=4)
    env = ep.env
    _run_until(ep, _swap_end(env) - 1)
    assert ep.task_index == 0  # still in the demo segment before the last swap ends
    ep.step()
    assert ep.task_index == 1


# --------------------------------------------------------------------------- ButtonUnmaskSwap


@pytest.fixture
def bus():
    with OfficialWorld("ButtonUnmaskSwap") as w:
        yield w


REVEAL_END = T.REVEAL_END_STEP + 1  # past the reveal animation window


def _press_both(ep, env, order=("button_left", "button_right")):
    for name in order:
        btn = getattr(env, name)
        ep.press(btn)
        ep.step()
        ep.unpress(btn)


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("order", [("button_left", "button_right"), ("button_right", "button_left")])
def test_bus_two_buttons_then_follow_identity_succeeds(bus, diff, order):
    ep = bus.make(diff, seed=4)
    env = ep.env
    ep.step(REVEAL_END)
    _press_both(ep, env, order)
    assert ep.task_index == 2
    _run_until(ep, _swap_end(env) + 2)
    _pick_sequence(ep, env)
    assert ep.success and not ep.fail


def test_bus_same_button_twice_does_not_advance(bus):
    ep = bus.make("easy", seed=4)
    env = ep.env
    ep.step(REVEAL_END)
    _press_both(ep, env, ("button_left", "button_left"))
    assert ep.task_index == 1
    assert env.button_list == [env.button_right]


def test_bus_both_buttons_same_step_blocks_second_subgoal(bus):
    """Current official behavior: both buttons pressed on the same step, the first subtask removes both from the list and the second subtask never completes."""
    ep = bus.make("easy", seed=4)
    env = ep.env
    ep.step(REVEAL_END)
    ep.press(env.button_left)
    ep.press(env.button_right)
    ep.step()
    ep.unpress(env.button_left)
    ep.unpress(env.button_right)
    assert ep.task_index == 1 and env.button_list == []
    _run_until(ep, _swap_end(env) + 2)
    _press_both(ep, env)
    _pick_sequence(ep, env)
    assert not ep.success and ep.task_index == 1


def test_bus_one_button_then_pick_is_not_success(bus):
    ep = bus.make("medium", seed=4)
    env = ep.env
    ep.step(REVEAL_END)
    _press_both(ep, env, ("button_left",))
    _run_until(ep, _swap_end(env) + 2)
    ep.grasp(env.selected_bins[0], z=BIN_UP)
    ep.step(2)
    assert not ep.success and not ep.fail and ep.task_index == 1


def test_bus_button_list_only_restored_by_rebuild(bus):
    """The button list is built in _load_scene and not reset by _initialize_episode: rerunning only _initialize_episode leaves the list stale (current official behavior);
    the benchmark re-runs gym.make per episode (rerunning _load_scene), so both buttons are present in a new episode."""
    ep = bus.make("easy", seed=4)
    env = ep.env
    ep.step(REVEAL_END)
    _press_both(ep, env)
    assert env.button_list == []
    import torch

    env._initialize_episode(torch.arange(1), {})
    assert env.button_list == []
    ep2 = bus.make("easy", seed=4)
    assert ep2.env.button_list == [ep2.env.button_left, ep2.env.button_right]
