"""VideoPlaceButton new-value tier truth table (xhard1, xhard2): in the execution segment place the asked cube on the target stand it was "last placed on before the button / first placed on after the button".

The answer is derived by hand-written rules from the placement events that actually happened in the demonstration segment (before → the last target stand the cube landed on before the button; after → the first after the button);
target stands swap positions at the end of the demonstration, and the answer follows the stand's identity. Errors and boundaries: swapping before/after, placing on the stand now at the answer stand's pre-swap position,
and picking another cube in the execution segment are all failures.
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import demo_driver as DD
from .swap_driver import bin_at

TASK = "VideoPlaceButton"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier, k=0: World.build(TASK, tier, k)


def _answer(w, log):
    seq = DD.target_placements(log, w.env.target_cube, w.env.targets)
    if w.env.target_target_language == "before":
        return [t for t, after in seq if not after][-1], [t for t, after in seq if after][0]
    return [t for t, after in seq if after][0], [t for t, after in seq if not after][-1]


def _place(w, target):
    cube = w.env.target_cube
    w.grasp(cube)
    assert w.step() == OK
    w.release_onto(cube, w.xyz(target)[:2])
    return w.step()


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("k", range(2))
def test_place_onto_answer_target_succeeds(world, tier, k):
    w = world(tier, k)
    log = DD.run_demo(w)
    answer, _ = _answer(w, log)
    assert answer is w.env.target_target
    assert _place(w, answer) == {"success": True, "fail": False}


#: Cell choice (measured in T12, first 8 formal episodes each of xhard1/xhard2): the before/after swap can be built in episode 0 (possible in 8/8 episodes);
#: "the stand now at the answer stand's pre-swap position" exists only in some episodes (xhard1 episodes 1, 4, 6; xhard2 episodes 2, 4, 5);
#: when episode 0 was used for everything, this branch held in neither tier and the assertion was vacuous. Below, episodes that can trigger it are pinned per tier, and the trigger condition is written as a precondition assertion:
#: if the packaged specs change so the condition no longer holds, the test fails loudly instead of silently doing nothing.
SWAP_K = {"xhard1": 0, "xhard2": 0}
OLD_POS_K = {"xhard1": 1, "xhard2": 2}


@pytest.mark.parametrize("tier", TIERS)
def test_before_after_swapped_fails(world, tier):
    w = world(tier, SWAP_K[tier])
    log = DD.run_demo(w)
    answer, mirror = _answer(w, log)
    assert mirror is not answer, "cell choice invalid: the same stand was used before and after the button in this episode, so before/after swap cannot be distinguished; reselect SWAP_K"
    assert _place(w, mirror) == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS)
def test_answer_old_position_and_wrong_cube_fail(world, tier):
    w = world(tier, OLD_POS_K[tier])
    pre = {t.name: w.xyz(t)[:2].copy() for t in w.env.targets}
    log = DD.run_demo(w)
    answer, _ = _answer(w, log)
    impostor = bin_at(w, pre[answer.name], w.env.targets)
    assert impostor is not None and impostor is not answer, \
        "cell choice invalid: no other stand is now at the answer stand's pre-swap position; reselect OLD_POS_K"
    assert _place(w, impostor) == {"success": False, "fail": True}
    w = World.build(TASK, tier)
    DD.run_demo(w)
    w.grasp(w.env.non_target_cubes[0])
    assert w.step() == {"success": False, "fail": True}
