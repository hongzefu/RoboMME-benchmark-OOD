"""BinFill new-value tier truth table (xhard1, xhard2): drop the correct number of cubes color by color following the language sequence, then press the button for success.

Errors and boundaries: pressing the button one cube short, dropping one extra cube in the button phase, dropping the wrong color (right count, wrong color), and pressing the button early are all failures;
after being dropped a cube is removed from the scene and counted only once. Colors and quotas come from this episode (packaged spec replay).
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world

TASK = "BinFill"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _cubes(w, color):
    return list(getattr(w.env, f"{color}_cubes"))


def _drop_in(w, cube):
    """Pick up → put into the hole board (land at the board center, release, lift the gripper away)."""
    w.grasp(cube)
    a = w.tick()
    w.release_onto(cube, w.xyz(w.env.board_with_hole)[:2])
    b = w.tick()
    return a, b


def _fill(w, plan):
    for color, n in plan:
        for cube in _cubes(w, color)[:n]:
            assert _drop_in(w, cube) == (OK, OK)


@pytest.mark.parametrize("tier", TIERS)
def test_quota_then_button_succeeds(world, tier):
    w = world(tier)
    plan = list(w.env.binfill_language_sequence)
    _fill(w, plan)
    for color, n in plan:
        assert getattr(w.env, f"{color}_cubes_in_bin") == n
    w.press(w.env.button)
    assert w.tick() == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS)
def test_one_short_then_button_fails(world, tier):
    w = world(tier)
    plan = list(w.env.binfill_language_sequence)
    color, n = plan[-1]
    _fill(w, plan[:-1] + [(color, n - 1)])
    w.press(w.env.button)
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS)
def test_extra_cube_in_button_stage_fails(world, tier):
    w = world(tier)
    plan = list(w.env.binfill_language_sequence)
    _fill(w, plan)
    color, n = plan[0]
    spare = _cubes(w, color)[n]  # one extra cube of the same color
    w.grasp(spare)
    w.tick()
    w.release_onto(spare, w.xyz(w.env.board_with_hole)[:2])
    assert w.tick()["fail"] is True


@pytest.mark.parametrize("tier", TIERS[:1])
def test_wrong_colour_right_total_fails_at_button(world, tier):
    """Right total count, wrong color: replace one cube of the last color with a cube of a surplus color → count mismatch when pressing the button means failure."""
    w = world(tier)
    plan = list(w.env.binfill_language_sequence)
    color, n = plan[-1]
    other = next(c for c, _ in plan if c != color)
    used = dict(plan)
    _fill(w, plan[:-1] + [(color, n - 1)])
    substitute = _cubes(w, other)[used[other]]
    assert _drop_in(w, substitute) == (OK, OK)
    w.press(w.env.button)
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_early_button_fails_and_binned_cube_counts_once(world, tier):
    w = world(tier)
    color, _ = w.env.binfill_language_sequence[0]
    cube = _cubes(w, color)[0]
    _drop_in(w, cube)
    assert getattr(w.env, f"{color}_cubes_in_bin") == 1
    # dropped cubes are moved out of the hole board, so another evaluate does not count them twice
    w.tick()
    w.tick()
    assert getattr(w.env, f"{color}_cubes_in_bin") == 1
    w.press(w.env.button)
    assert w.tick() == {"success": False, "fail": True}
