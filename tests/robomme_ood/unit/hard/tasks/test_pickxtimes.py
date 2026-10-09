"""PickXtimes new-value tier truth table: N complete pick-and-places then pressing the button is success; N−1/N+1, holding continuously, wrong object, pressing the button while holding are failure or not success.

A CPU world stand-in drives the real task table and ``evaluate``; N comes from this episode's ``num_repeats`` (packaged spec replay), not hard-coded in the test.
(Threshold-equality boundary tests live in SwingXtimes: its thresholds come from sampling parameters and can be replaced with values exactly representable in float32.)
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world

TASK = "PickXtimes"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


def _disc_xy(w):
    return w.xyz(w.env.target)[:2]


def _cycle(w, n):
    """Complete pick-and-place n times: pick up the target cube → place on the plate, evaluate once each."""
    for _ in range(n):
        w.grasp(w.env.target_cube)
        assert w.tick() == OK
        w.release_onto(w.env.target_cube, _disc_xy(w))
        assert w.tick() == OK


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_n_cycles_then_button_succeeds(world, tier):
    w = world(tier)
    n = w.env.num_repeats
    _cycle(w, n)
    w.press(w.env.button)
    assert w.tick() == {"success": True, "fail": False}
    # terminal state stable after success
    assert w.tick()["success"] is True


@pytest.mark.parametrize("tier", TIERS)
def test_one_cycle_short_then_button_fails(world, tier):
    w = world(tier)
    _cycle(w, w.env.num_repeats - 1)
    w.press(w.env.button)
    assert w.tick() == {"success": False, "fail": True}
    w.unpress(w.env.button)
    assert w.tick()["fail"] is True, "failure terminal state must persist"


@pytest.mark.parametrize("tier", TIERS)
def test_extra_cycle_fails(world, tier):
    w = world(tier)
    _cycle(w, w.env.num_repeats)
    w.grasp(w.env.target_cube)  # (N+1)-th pick: any cube picked up in the button phase counts as failure
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS)
def test_pressing_button_while_holding_fails(world, tier):
    w = world(tier)
    _cycle(w, w.env.num_repeats)
    w.grasp(w.env.target_cube)
    w.press(w.env.button)
    assert w.tick() == {"success": False, "fail": True}, "failure verdict takes priority over completion verdict"


@pytest.mark.parametrize("tier", TIERS)
def test_holding_forever_never_completes(world, tier):
    w = world(tier)
    w.grasp(w.env.target_cube)
    for _ in range(5):
        assert w.tick() == OK
    assert w.stage == 1, "holding continuously only completes the first item \"pick up\"; the place item does not advance"


@pytest.mark.parametrize("tier", TIERS)
def test_grasping_distractor_mid_task_fails(world, tier):
    w = world(tier)
    _cycle(w, 1)
    w.grasp(w.env.distractor_cubes[0])
    assert w.tick() == {"success": False, "fail": True}
