"""PatternLock new-value tier truth table (xhard1-xhard3): in the execution segment, starting from the start point, fully retracing the same button sequence as the demonstration is success.

Errors and boundaries: skipping a node (touching the button after next directly) fails; staying several steps on the same button counts once (debounce);
touching a button off the path during the retrace fails; failure is latched. The path comes from this episode (packaged spec replay).
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import stick_driver as SD

TASK = "PatternLock"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier, k=0: World.build(TASK, tier, k)


@pytest.mark.parametrize("tier", TIERS)
def test_replaying_the_whole_path_succeeds(world, tier):
    w = world(tier)
    SD.run_demo(w)
    path = w.env.selected_buttons
    out = None
    for b in path[1:]:
        SD.touch(w, b)
        out = w.step()
    assert out == {"success": True, "fail": False}
    assert w.env.match is True


@pytest.mark.parametrize("tier", TIERS)
def test_skipping_a_node_fails(world, tier):
    w = world(tier)
    SD.run_demo(w)
    SD.touch(w, w.env.selected_buttons[2])
    assert w.step() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_dwelling_on_a_button_counts_once(world, tier):
    w = world(tier)
    SD.run_demo(w)
    before = len(w.env.achieved_list)
    nxt = w.env.selected_buttons[1]
    SD.touch(w, nxt)
    for _ in range(4):
        assert w.step() == OK
    assert len(w.env.achieved_list) == before + 1


@pytest.mark.parametrize("tier", TIERS[:1])
def test_stray_touch_fails_and_stays_failed(world, tier):
    w = world(tier)
    SD.run_demo(w)
    on_path = set(id(b) for b in w.env.selected_buttons)
    stray = next(b for b in w.env.buttons_grid if id(b) not in on_path)
    SD.touch(w, stray)
    assert w.step()["fail"] is True
