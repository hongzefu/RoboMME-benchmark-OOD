"""ButtonUnmaskSwap new-value tier truth table (xhard1, xhard2): press two different buttons in turn → wait for the swaps to end → pick up the container still hiding the target-color cube.

Errors and boundaries: pressing the same button again does not advance; pressing two buttons in the same frame advances only one item and the second can never be completed;
missing the second button does not advance; picking the right container before the swaps end does not advance; picking a distractor container after the swaps fails; after rebuilding the env the button list is back to two.
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import swap_driver as S
from . import unmask_driver as D

TASK = "ButtonUnmaskSwap"
TIERS = O.tiers_of(TASK)
OK = S.OK


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _tap(w, button):
    w.press(button)
    out = w.step()
    w.unpress(button)
    return out


@pytest.mark.parametrize("tier", TIERS)
def test_two_buttons_then_follow_identity_succeeds(world, tier):
    w = world(tier)
    assert len(w.env.button_list) == 2, "a newly built env restores the button list to two (no cross-episode residue)"
    assert _tap(w, w.env.button_left) == OK and w.stage == 1
    assert _tap(w, w.env.button_right) == OK and w.stage == 2
    S.run_through_swaps(w)
    assert w.stage == 3, "the wait-for-swaps item is complete"
    bins = S.bins_in_order(w)
    assert [b.name for b in bins] == [w.env.selected_bins[i].name for i in range(w.env.pick_times)]
    assert S.pick_sequence(w, bins) == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_both_buttons_in_one_frame_strands_second_item(world, tier):
    w = world(tier)
    w.press(w.env.button_left)
    w.press(w.env.button_right)
    w.step()
    w.unpress(w.env.button_left)
    w.unpress(w.env.button_right)
    assert w.stage == 1 and w.env.button_list == []
    for _ in range(3):
        _tap(w, w.env.button_right)
    assert w.stage == 1, "both buttons were removed from the list in the same frame, so the second item can no longer be completed"


@pytest.mark.parametrize("tier", TIERS[:1])
def test_repeat_button_missing_button_early_pick_then_distractor(world, tier):
    """Checks in sequence within one episode: pressing the same button again does not advance; with the second button missing, picking the right container does not advance; with both buttons pressed
    and the swaps finished, lifting a distractor container fails (Swap scene building is heavy, so several negative cases share one episode)."""
    w = world(tier)
    _tap(w, w.env.button_left)
    assert _tap(w, w.env.button_left) == OK and w.stage == 1
    right = w.env.selected_bins[0]
    D.lift(w, right)
    for _ in range(2):
        assert w.step() == OK
    assert w.stage == 1
    D.put_down(w, right)
    assert _tap(w, w.env.button_right) == OK and w.stage == 2
    S.run_through_swaps(w)
    D.lift(w, w.env.distractor_bins[0])
    assert w.tick() == {"success": False, "fail": True}
