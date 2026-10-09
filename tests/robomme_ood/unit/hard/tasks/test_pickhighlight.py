"""PickHighlight new-value tier truth table (xhard1, xhard2): press button → pick and place each highlighted cube in turn → press the button again at the end for success.

Errors and boundaries: missing a target or not pressing the final button is not success; picking a non-highlighted cube fails; picking any cube before the first button fails;
picking the same target again neither advances nor fails. Highlighted targets and order come from this episode (packaged spec replay).
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world

TASK = "PickHighlight"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _press_release(w):
    w.press(w.env.button)
    out = w.tick()
    w.unpress(w.env.button)
    return out


def _pick_place(w, cube):
    w.grasp(cube)
    a = w.tick()
    w.release_onto(cube, w.xyz(cube)[:2])
    b = w.tick()
    return a, b


@pytest.mark.parametrize("tier", TIERS)
def test_all_highlighted_then_final_button_succeeds(world, tier):
    w = world(tier)
    assert _press_release(w) == OK
    for cube in w.env.target_cubes:
        assert _pick_place(w, cube) == (OK, OK)
    assert w.tick() == OK, "not pressing the final button is not success"
    w.press(w.env.button)
    assert w.tick() == {"success": True, "fail": False}
    assert all(v == 1 for v in w.env.target_cube_pickup_counts.values())


@pytest.mark.parametrize("tier", TIERS)
def test_missing_last_target_never_succeeds(world, tier):
    w = world(tier)
    _press_release(w)
    for cube in w.env.target_cubes[:-1]:
        _pick_place(w, cube)
    w.press(w.env.button)
    for _ in range(3):
        assert w.tick() == OK


@pytest.mark.parametrize("tier", TIERS)
def test_unhighlighted_cube_fails(world, tier):
    w = world(tier)
    _press_release(w)
    other = next(c for c in w.env.all_cubes if c not in w.env.target_cubes)
    w.grasp(other)
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_pick_before_first_button_fails(world, tier):
    w = world(tier)
    w.grasp(w.env.target_cubes[0])
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_repeating_same_target_does_not_advance(world, tier):
    w = world(tier)
    _press_release(w)
    first = w.env.target_cubes[0]
    _pick_place(w, first)
    stage = w.stage
    assert _pick_place(w, first) == (OK, OK)
    assert w.stage == stage, "picking the first target again should not advance to the next item"
    assert w.env.target_cube_pickup_counts[first.name] == 2
