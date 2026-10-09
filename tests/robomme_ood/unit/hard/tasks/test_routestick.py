"""RouteStick new-value tier truth table (xhard1-xhard3): in the execution segment follow the demonstrated stop sequence, going around each peg from the side the demonstration specified.

Direction criterion: the mean cross product of this segment's TCP trajectory relative to the "previous stop → this stop" chord, positive is clockwise, negative counterclockwise
(production ``direction_fail``). The test places a waypoint outside the chord midpoint (above all height thresholds), with the side given by a hand-computed normal vector.
Errors and boundaries: right stop but wrong winding fails; waypoint exactly on the chord (zero cross product) fails; failure is latched; jumping to another stop fails.
"""
from __future__ import annotations

import numpy as np
import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import stick_driver as SD

TASK = "RouteStick"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier, k=0: World.build(TASK, tier, k)


def _via(w, prev, cur, side):
    """Waypoint: chord midpoint offset by side×0.1 m along the left normal (−ly, lx); side=+1 gives a positive cross product (clockwise)."""
    p, c = w.xyz(prev)[:2], w.xyz(cur)[:2]
    line = c - p
    n = np.array([-line[1], line[0]]) / np.linalg.norm(line)
    return (p + c) / 2 + side * 0.1 * n


VIA_STEPS = 8  # steps spent at the waypoint: trajectory points are dominated by the waypoint (the TCP point at the moment of reset is also in the first segment's trajectory, see the handover notes)


def _segment(w, prev, cur, side):
    SD.hover(w, _via(w, prev, cur, side))
    for _ in range(VIA_STEPS):
        out = w.step()
        if out["fail"]:
            return out
    SD.touch(w, cur)
    return w.step()


def _sign(direction):
    return {"clockwise": +1, "counterclockwise": -1}[direction]


@pytest.mark.parametrize("tier", TIERS)
def test_correct_nodes_and_sides_succeed(world, tier):
    w = world(tier)
    SD.run_demo(w)
    path, dirs = w.env.selected_buttons, w.env.swing_directions
    out = None
    for i, cur in enumerate(path[1:]):
        out = _segment(w, path[i], cur, _sign(dirs[i]))
        if i < len(dirs) - 1:
            assert out == OK, i
    assert out == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS)
def test_reversed_side_fails_and_latches(world, tier):
    w = world(tier)
    SD.run_demo(w)
    path, dirs = w.env.selected_buttons, w.env.swing_directions
    assert _segment(w, path[0], path[1], -_sign(dirs[0]))["fail"] is True
    SD.hover(w, w.xyz(path[1])[:2])
    assert w.step()["fail"] is True, "failure latched"


@pytest.mark.parametrize("tier", TIERS[:1])
def test_path_on_the_line_fails(world, tier):
    """The whole second segment moves along the chord (zero cross product) → failure. The second segment is used because the first segment's trajectory cache still holds the TCP point from the moment of reset
    (production ``_gripper_xy_trace`` is not cleared at the start of the execution segment, see the production issue in the handover notes); the cache is only cleared after the first segment."""
    w = world(tier)
    SD.run_demo(w)
    path, dirs = w.env.selected_buttons, w.env.swing_directions
    assert _segment(w, path[0], path[1], _sign(dirs[0])) == OK
    assert _segment(w, path[1], path[2], 0)["fail"] is True


@pytest.mark.parametrize("tier", TIERS[:1])
def test_jumping_to_a_wrong_node_fails(world, tier):
    w = world(tier)
    SD.run_demo(w)
    path = w.env.selected_buttons
    wrong = next(b for b in w.env.buttons_grid if b is not path[0] and b is not path[1])
    SD.touch(w, wrong)
    assert w.step()["fail"] is True
