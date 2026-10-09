"""SwingXtimes new-value tier truth table (xhard1-xhard5, N from this episode's ``num_repeats``): pick up → swing right→left N rounds → put down → button is success.

Errors and boundaries: reversed left/right order does not advance; staying several steps on the same side counts as one swing (step's enter/exit hysteresis); total swings above the cap fails;
picking a non-target cube or pressing the button early fails; the distance threshold includes equality, the height threshold does not (checked after replacing thresholds with values exactly representable in float32).
Everything goes through the task class's real ``step`` (swing counting is in step) and ``evaluate``.
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world

TASK = "SwingXtimes"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}
SWING_Z = 0.08  # height when swinging above a target: below the enter threshold
AWAY_Z = 0.25


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _over(w, target, z=SWING_Z):
    x, y = w.xyz(target)[:2]
    w.move(w.env.target_cube, (x, y, z))
    w.tcp_to((x, y, z))


def _between(w):
    """Leave both targets: move high above the midpoint of the line between the two targets (horizontal distance to both far beyond the exit threshold)."""
    a, b = w.xyz(w.env.target_right)[:2], w.xyz(w.env.target_left)[:2]
    mid = (a + b) / 2
    w.move(w.env.target_cube, (mid[0], mid[1], AWAY_Z))
    w.tcp_to((mid[0], mid[1], AWAY_Z))


def _swing_rounds(w, n):
    for _ in range(n):
        for side in (w.env.target_right, w.env.target_left):
            _over(w, side)
            assert w.step() == OK
            _between(w)
            assert w.step() == OK


def _pick(w):
    w.grasp(w.env.target_cube)
    assert w.step() == OK
    assert w.stage == 1


@pytest.mark.parametrize("tier", TIERS)
def test_n_rounds_then_putdown_then_button_succeeds(world, tier):
    w = world(tier)
    n = w.env.num_repeats
    _pick(w)
    _swing_rounds(w, n)
    assert w.env.swing_count == 2 * n
    w.release_onto(w.env.target_cube, w.xyz(w.env.target_cube)[:2])
    assert w.step() == OK
    w.press(w.env.button)
    assert w.step() == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS)
def test_one_round_short_then_button_fails(world, tier):
    w = world(tier)
    _pick(w)
    _swing_rounds(w, w.env.num_repeats - 1)
    w.press(w.env.button)
    assert w.step()["fail"] is True


@pytest.mark.parametrize("tier", TIERS[:1])
def test_left_before_right_does_not_advance(world, tier):
    w = world(tier)
    _pick(w)
    _over(w, w.env.target_left)
    assert w.step() == OK
    assert w.stage == 1, "swinging left first does not advance (right→left required)"
    _between(w)
    _over(w, w.env.target_right)
    assert w.step() == OK and w.stage == 2


@pytest.mark.parametrize("tier", TIERS[:1])
def test_staying_on_one_side_counts_once(world, tier):
    w = world(tier)
    _pick(w)
    _over(w, w.env.target_right)
    for _ in range(4):
        w.step()
    assert w.env.swing_count == 1


@pytest.mark.parametrize("tier", TIERS)
def test_exceeding_max_swings_fails(world, tier):
    """Swinging more than the production cap ``max_swings`` fails (after the extra round)."""
    w = world(tier)
    _pick(w)
    limit = w.env.max_swings
    out = OK
    sides = [w.env.target_right, w.env.target_left]
    for i in range(limit + 2):
        _over(w, sides[i % 2])
        out = w.step()
        _between(w)
        out = w.step()
        if out["fail"]:
            break
    assert w.env.swing_count > limit
    assert out == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_non_target_pickup_and_early_button_fail(world, tier):
    w = world(tier)
    _pick(w)
    w.agent.held = None
    w.grasp(w.env.distractor_cubes[0])
    assert w.step()["fail"] is True
    w = World.build(TASK, tier)
    _pick(w)
    w.press(w.env.button)
    assert w.step()["fail"] is True


@pytest.mark.parametrize("tier", TIERS[:1])
def test_swing_distance_inclusive_and_height_exclusive(world, tier):
    """Threshold equality: the task table's swing verdict reads the sampling parameter ``swing_thresholds``; replace it with float32-exact
    1/32 m and 1/8 m and move the targets to the origin: horizontal distance exactly at the threshold → reached (``<=``); height exactly at the threshold → not reached (``<``)."""
    w = world(tier)
    _pick(w)
    thr = w.env._sampling["parameters"]["swing_thresholds"]
    thr["distance"], thr["z"] = 2.0 ** -5, 2.0 ** -3
    right = w.env.target_right
    w.move(right, (0.0, 0.0, 0.0))
    w.move(w.env.target_cube, (2.0 ** -5 + 1e-4, 0.0, 0.1))
    assert w.tick() == OK and w.stage == 1
    w.move(w.env.target_cube, (0.0, 0.0, 2.0 ** -3))
    assert w.tick() == OK and w.stage == 1
    w.move(w.env.target_cube, (2.0 ** -5, 0.0, 0.1))
    assert w.tick() == OK and w.stage == 2
