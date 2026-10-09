"""ButtonUnmask new-value tier truth table (xhard1-xhard4): press the button to reveal first → then pick up "the container hiding the cube of the k-th color" in order and put it down.

Errors and boundaries: skipping the button and picking the right container directly does not advance (the button item does not judge failure); after the button, picking the wrong container or lifting a distractor container fails;
a button press depth of zero does not count as pressed, only pressing to the bottom of travel does. Pick count comes from this tier.
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import unmask_driver as D

TASK = "ButtonUnmask"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _bins_in_order(w):
    cubes = D.colour_cubes(w.env)[: w.env.xhard_pick_count]
    return [D.bin_hiding(w, c, w.env.spawned_bins) for c in cubes]


def _press(w):
    w.press(w.env.button)
    out = w.tick()
    w.unpress(w.env.button)
    return out


@pytest.mark.parametrize("tier", TIERS)
def test_button_then_each_hiding_container_succeeds(world, tier):
    w = world(tier)
    assert _press(w) == OK and w.stage == 1
    bins = _bins_in_order(w)
    out = None
    for k, b in enumerate(bins):
        assert w.env.color_names[k] in w.env.task_list[w.stage]["name"]
        D.lift(w, b)
        out = w.tick()
        if k < len(bins) - 1:
            assert out == OK
            D.put_down(w, b)
            assert w.tick() == OK
    assert out == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_skipping_button_does_not_advance(world, tier):
    w = world(tier)
    D.lift(w, _bins_in_order(w)[0])
    for _ in range(3):
        assert w.tick() == OK
    assert w.stage == 0


@pytest.mark.parametrize("tier", TIERS)
def test_wrong_or_distractor_container_after_button_fails(world, tier):
    w = world(tier)
    _press(w)
    right = _bins_in_order(w)[0]
    D.lift(w, next(b for b in w.env.spawned_bins if b is not right))
    assert w.tick() == {"success": False, "fail": True}
    w = World.build(TASK, tier)
    _press(w)
    D.lift(w, w.env.distractor_bins[-1])
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_button_depth_zero_is_not_a_press(world, tier):
    w = world(tier)
    w.press(w.env.button, depth=0.0)
    assert w.tick() == OK and w.stage == 0
    w.press(w.env.button)
    assert w.tick() == OK and w.stage == 1
