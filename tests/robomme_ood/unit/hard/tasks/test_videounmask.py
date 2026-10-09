"""VideoUnmask new-value tier truth table (xhard1-xhard4): stay still after the demonstration → pick up "the container hiding the cube of the k-th color" in order, put it down, then pick the next.

Errors and boundaries: picking the wrong container (another in-region container) fails; lifting the next one directly without putting down between two picks fails;
lifting any distractor container fails; staying still for fewer than 64 steps does not advance. Pick count (single/double/triple) comes from this tier (packaged spec replay).
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import unmask_driver as D

TASK = "VideoUnmask"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _settle(w):
    """First item, stillness check: advances after the arm stays still for some steps (the step count is judged by production static_check; the test only advances until the stage changes)."""
    w.still()
    for _ in range(200):
        out = w.tick()
        if w.stage >= 1:
            return out
    raise AssertionError("still not advanced after staying still 200 steps")


def _bins_in_order(w):
    cubes = D.colour_cubes(w.env)[: w.env.xhard_pick_count]
    return [D.bin_hiding(w, c, w.env.spawned_bins) for c in cubes]


@pytest.mark.parametrize("tier", TIERS)
def test_pick_each_hiding_container_in_order_succeeds(world, tier):
    w = world(tier)
    assert _settle(w) == OK
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


@pytest.mark.parametrize("tier", TIERS)
def test_wrong_container_fails(world, tier):
    w = world(tier)
    _settle(w)
    right = _bins_in_order(w)[0]
    wrong = next(b for b in w.env.spawned_bins if b is not right)
    D.lift(w, wrong)
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS)
def test_distractor_container_fails(world, tier):
    w = world(tier)
    _settle(w)
    D.lift(w, w.env.distractor_bins[0])
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_second_pick_without_putting_down_fails(world, tier):
    w = world(tier)
    _settle(w)
    first, second = _bins_in_order(w)[:2]
    D.lift(w, first)
    assert w.tick() == OK
    D.lift(w, second)  # the first one is still lifted
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_moving_robot_delays_static_stage(world, tier):
    w = world(tier)
    w.moving()
    for _ in range(100):
        assert w.tick() == OK
    assert w.stage == 0
