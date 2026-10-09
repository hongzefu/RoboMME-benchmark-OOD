"""VideoPlaceOrder new-value tier truth table (xhard1, xhard2): in the execution segment place the asked cube on the target stand it was placed on "the k-th time" in the demonstration (temporal order).

The answer is derived from the placement events that actually happened in the demonstration segment: the asked cube's ``which_in_subset``-th target-stand placement in time order (k is the ordinal in the prompt).
Target stands swap positions at the end of the demonstration, and the answer follows the stand's identity. Errors and boundaries: impersonation by the first/last placement (if not the k-th),
the stand now at the answer stand's pre-swap position, and picking another cube in the execution segment are all failures; a button press in the middle of the placement sequence does not affect the ordinal.
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import demo_driver as DD
from .swap_driver import bin_at

TASK = "VideoPlaceOrder"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier, k=0: World.build(TASK, tier, k)


def _visits(w, log):
    return [t for t, _ in DD.target_placements(log, w.env.target_cube, w.env.targets)]


def _place(w, target):
    cube = w.env.target_cube
    w.grasp(cube)
    assert w.step() == OK
    w.release_onto(cube, w.xyz(target)[:2])
    return w.step()


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("k", range(2))
def test_place_onto_kth_visit_succeeds(world, tier, k):
    w = world(tier, k)
    log = DD.run_demo(w)
    assert log.button_at is not None, "the button was pressed in the demonstration"
    visits = _visits(w, log)
    answer = visits[w.env.which_in_subset - 1]
    assert answer is w.env.target_target
    assert _place(w, answer) == {"success": True, "fail": False}


#: Cell choice (measured in T12, first 8 formal episodes each of xhard1/xhard2): first/last impersonation can be built in episode 0 (possible in 8/8 episodes);
#: "the stand now at the answer stand's pre-swap position" exists only in some episodes (xhard1 episodes 1, 2, 4, 7; xhard2 episodes 0-6);
#: when xhard1 episode 0 was used for everything, this branch did not hold and the assertion was vacuous. Below, episodes that can trigger it are pinned per tier, and the trigger condition is written as a precondition assertion:
#: if the packaged specs change so the condition no longer holds, the test fails loudly instead of silently doing nothing.
ORDINAL_K = {"xhard1": 0, "xhard2": 0}
OLD_POS_K = {"xhard1": 1, "xhard2": 0}


@pytest.mark.parametrize("tier", TIERS)
def test_other_ordinal_fails(world, tier):
    w = world(tier, ORDINAL_K[tier])
    log = DD.run_demo(w)
    visits = _visits(w, log)
    answer = visits[w.env.which_in_subset - 1]
    decoy = next((t for t in (visits[0], visits[-1]) if t is not answer), None)
    assert decoy is not None, "cell choice invalid: first and last placements both landed on the answer stand in this episode, so ordinal impersonation cannot be built; reselect ORDINAL_K"
    assert _place(w, decoy) == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS)
def test_answer_old_position_and_wrong_cube_fail(world, tier):
    w = world(tier, OLD_POS_K[tier])
    pre = {t.name: w.xyz(t)[:2].copy() for t in w.env.targets}
    log = DD.run_demo(w)
    answer = _visits(w, log)[w.env.which_in_subset - 1]
    impostor = bin_at(w, pre[answer.name], w.env.targets)
    assert impostor is not None and impostor is not answer, \
        "cell choice invalid: no other stand is now at the answer stand's pre-swap position; reselect OLD_POS_K"
    assert _place(w, impostor) == {"success": False, "fail": True}
    w = World.build(TASK, tier)
    DD.run_demo(w)
    w.grasp(w.env.non_target_cubes[0])
    assert w.step() == {"success": False, "fail": True}
