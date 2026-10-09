"""VideoUnmaskSwap new-value tier truth table (xhard1, xhard2): after the containers are swapped several times, pick up the container still hiding the target-color cube (following identity).

Walk through the reveal and all swap windows via the real ``step`` (cubes moving with containers is done by production's parking/drop-back logic), then pick and place by the post-swap geometric hiding relation:
positive case succeeds; picking the container now at "the target cube's pre-swap position" (if it has been replaced by another container) fails; lifting a distractor container fails;
picking the right container before the swaps finish (stillness item not complete) does not advance.
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import swap_driver as S
from . import unmask_driver as D

TASK = "VideoUnmaskSwap"
TIERS = O.tiers_of(TASK)


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier, k=0: World.build(TASK, tier, k)


@pytest.mark.parametrize("tier", TIERS)
def test_follow_identity_through_swaps_succeeds(world, tier):
    w = world(tier)
    S.run_through_swaps(w)
    assert w.stage == 1, "stillness item complete after all swaps finish"
    bins = S.bins_in_order(w)
    assert [b.name for b in bins] == [w.env.selected_bins[i].name for i in range(w.env.pick_times)]
    assert S.pick_sequence(w, bins) == {"success": True, "fail": False}


#: Cell choice (T13 offline probe, first 8 formal episodes each of xhard1/xhard2): "another container is now at the target cube's pre-swap position"
#: holds in xhard1 episodes 0, 1, 2, 4, 7 and xhard2 episodes 1, 3, 4, 5, 7; originally only xhard1 episode 0 was parametrized,
#: and xhard2 was never covered (its episode 0 does not hold). Episodes that can trigger it are pinned per tier with the trigger condition as a precondition assertion: if the packaged specs change so the condition
#: no longer holds, the test fails loudly instead of silently skipping.
OLD_POS_K = {"xhard1": 0, "xhard2": 1}


@pytest.mark.parametrize("tier", TIERS)
def test_original_position_is_a_trap(world, tier):
    w = world(tier, OLD_POS_K[tier])
    origin = S.run_through_swaps(w)
    target = D.colour_cubes(w.env)[0]
    right = S.bins_in_order(w)[0]
    impostor = S.bin_at(w, origin[target.name], w.env.spawned_bins)
    assert impostor is not None and impostor is not right, \
        "cell choice invalid: no other container is now at the target cube's pre-swap position; reselect OLD_POS_K"
    D.lift(w, impostor)
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_distractor_container_after_swaps_fails(world, tier):
    w = world(tier)
    S.run_through_swaps(w)
    D.lift(w, w.env.distractor_bins[0])
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_picking_before_swaps_finish_does_not_advance(world, tier):
    w = world(tier)
    right = w.env.selected_bins[0]
    D.lift(w, right)
    for _ in range(5):
        assert w.tick() == S.OK
    assert w.stage == 0
