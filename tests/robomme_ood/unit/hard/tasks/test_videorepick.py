"""VideoRepick new-value tier truth table (xhard1, xhard2): demonstration picks and places the target → several swaps → in the execution segment pick and place the same cube (identity, not position) N times, then press the button.

The demonstration and swaps are walked through via the task class's real ``step`` (swap animation is in step); N comes from this episode's ``num_repeats``.
Errors and boundaries: in the execution segment picking the cube now at "the target's pre-swap position" fails; pressing the button one pick-and-place short fails (within the time window);
the one pick-and-place in the demonstration does not count toward the execution segment count.
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from .swap_driver import bin_at

TASK = "VideoRepick"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier, k=0: World.build(TASK, tier, k)


def _demo_and_swaps(w):
    """Demonstration segment: pick up target → put down → stay still → wait for all swaps to end → reset. Returns the target's pre-swap position."""
    env = w.env
    cube = env.target_cube_1
    w.grasp(cube)
    assert w.step() == OK
    w.release_onto(cube, w.xyz(cube)[:2])
    assert w.step() == OK
    origin = w.xyz(cube)[:2].copy()
    w.still()
    last_end = env.swap_schedule[-1][3]
    while int(env.elapsed_steps) < last_end + 30:
        out = w.step()
        assert out == OK, int(env.elapsed_steps)
    first_exec = next(i for i, t in enumerate(env.task_list) if not t["demonstration"])
    assert w.stage == first_exec, "all demonstration and swap items complete"
    return origin


def _cycles(w, n):
    cube = w.env.target_cube_1
    for _ in range(n):
        w.grasp(cube)
        assert w.step() == OK
        w.release_onto(cube, w.xyz(cube)[:2])
        assert w.step() == OK


@pytest.mark.parametrize("tier", TIERS)
def test_repick_same_identity_n_times_succeeds(world, tier):
    w = world(tier)
    _demo_and_swaps(w)
    _cycles(w, w.env.num_repeats)
    w.press(w.env.button_left)
    assert w.step() == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_one_short_then_button_fails(world, tier):
    w = world(tier)
    _demo_and_swaps(w)
    _cycles(w, w.env.num_repeats - 1)
    w.press(w.env.button_left)
    assert w.step()["fail"] is True


#: Cell choice (T13 offline probe, first 8 formal episodes each of xhard1/xhard2): "another cube is now at the target's pre-swap position"
#: holds in xhard1 episodes 0, 2, 5 and xhard2 episodes 0, 1, 2, 5, 6; episode 0 holds in both tiers (the original skip never actually triggered),
#: but cells are still pinned explicitly per tier with the trigger condition as a precondition assertion; if the packaged specs change so the condition no longer holds, the test fails loudly instead of silently skipping.
OLD_POS_K = {"xhard1": 0, "xhard2": 0}


@pytest.mark.parametrize("tier", TIERS)
def test_original_position_cube_is_a_trap(world, tier):
    w = world(tier, OLD_POS_K[tier])
    origin = _demo_and_swaps(w)
    impostor = bin_at(w, origin, w.env.spawned_cubes)
    assert impostor is not None and impostor is not w.env.target_cube_1, \
        "cell choice invalid: no other cube is now at the target's pre-swap position; reselect OLD_POS_K"
    w.grasp(impostor)
    assert w.step() == {"success": False, "fail": True}
