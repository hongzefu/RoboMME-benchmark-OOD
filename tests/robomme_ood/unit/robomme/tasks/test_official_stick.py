"""PatternLock and RouteStick native three-tier truth tables (C05, C07 paths and cross products).

Both tasks use the stick: the tcp landing above a target (within the horizontal distance threshold and below the height threshold) "touches" that target.
The demo segment is really driven: touch the targets on the demo path in turn -> reset (solve_strong_reset sets after_demo) -> return to the start pose (swing_qpos).
- PatternLock: retrace the same order in the online segment; the start must be touched in the online segment (when reset back to swing_qpos the tcp is right above the start),
  and the last len(path) touch records must equal the path item by item; touching a button off the path (not expected, not the previous one) -> failure;
  lingering on the same button records it only once.
- RouteStick: besides landing correctly, each segment must detour in the demonstrated direction: the mean cross product of online trajectory points relative to the directed segment "previous target -> this target"
  > 0 is clockwise, < 0 counterclockwise, = 0 failure; wrong direction or touching the wrong target -> failure, and failure latches.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from _official_world import OfficialWorld, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

DIFFS = ("easy", "medium", "hard")
TOUCH_Z = T.STICK_TOUCH_Z
HIGH_Z = T.STICK_HIGH_Z


def _touch(ep, target):
    x, y, _ = target.xyz
    ep.tcp_to(x, y, TOUCH_Z)
    ep.step()


def _drive_demo(ep, home_on_start=True):
    env = ep.env
    path = env.selected_buttons
    _touch(ep, path[0])
    env.swing_qpos = env.agent.robot.qpos.clone()  # solve_swingonto(record_swing_qpos=True) records the start pose
    for t in path[1:]:
        _touch(ep, t)
    ep.tcp_to(0.0, 0.0, HIGH_Z)
    env.after_demo = True  # set by solve_strong_reset
    guard = 0
    while ep.task_index < ep.first_online_index() - 1:
        ep.step()
        guard += 1
        assert guard < 50
    if home_on_start:  # reset back to swing_qpos: tcp returns above the start
        x, y, _ = path[0].xyz
        ep.tcp_to(x, y, TOUCH_Z)
    ep.step()
    assert ep.task_index == ep.first_online_index()


# --------------------------------------------------------------------------- PatternLock


@pytest.fixture
def pl():
    with OfficialWorld("PatternLock") as w:
        yield w


@pytest.mark.parametrize("diff", DIFFS)
def test_pl_retrace_succeeds(pl, diff):
    ep = pl.make(diff, seed=9)
    env = ep.env
    grid = len(env.buttons_grid)
    assert int(round(grid ** 0.5)) ** 2 == grid
    _drive_demo(ep)
    for t in env.selected_buttons[1:]:
        _touch(ep, t)
        ep.step()  # linger one step: the same button is not recorded twice
    assert ep.success and not ep.fail
    assert [a.name for a in env.achieved_list] == [s.name for s in env.selected_buttons]


@pytest.mark.parametrize("diff", DIFFS)
def test_pl_wrong_button_fails(pl, diff):
    ep = pl.make(diff, seed=9)
    env = ep.env
    _drive_demo(ep)
    path = env.selected_buttons
    stray = next(b for b in env.buttons_grid if all(b is not p for p in path))
    _touch(ep, stray)
    assert ep.fail and not ep.success


def test_pl_skipping_a_node_fails(pl):
    for seed in range(40):
        ep = pl.make("hard", seed=seed)
        if len(ep.env.selected_buttons) >= 3:
            break
    env = ep.env
    _drive_demo(ep)
    _touch(ep, env.selected_buttons[2])  # skip selected[1]
    assert ep.fail and not ep.success


def test_pl_suffix_only_is_not_a_match(pl):
    """The start is not touched in the online segment (only the suffix is retraced): all subtasks complete, but the recent records differ from the path -> failure."""
    ep = pl.make("easy", seed=9)
    env = ep.env
    _drive_demo(ep, home_on_start=False)
    for t in env.selected_buttons[1:]:
        _touch(ep, t)
    assert ep.fail and not ep.success


def test_pl_demo_touches_are_not_recorded(pl):
    ep = pl.make("medium", seed=9)
    env = ep.env
    path = env.selected_buttons
    for t in path:
        _touch(ep, t)
    assert env.achieved_list == []  # nothing recorded before after_demo


# --------------------------------------------------------------------------- RouteStick


@pytest.fixture
def rs():
    with OfficialWorld("RouteStick") as w:
        yield w


def _detour(ep, prev, curr, sign, steps=3):
    """Walk along the directed segment prev->curr, offset to the sign side of the left normal (high, touching no target), and finally land on curr."""
    p, c = prev.xyz[:2], curr.xyz[:2]
    line = c - p
    normal = np.array([-line[1], line[0]]) / np.linalg.norm(line)
    for i in range(1, steps + 1):
        q = p + line * i / (steps + 1) + sign * T.DETOUR_OFFSET * normal
        ep.tcp_to(q[0], q[1], HIGH_Z)
        ep.step()
    _touch(ep, curr)


def _sign(direction):
    return 1.0 if direction == "clockwise" else -1.0  # positive cross product on the left-normal side -> clockwise


@pytest.mark.parametrize("diff", DIFFS)
def test_rs_follow_route_with_directions_succeeds(rs, diff):
    ep = rs.make(diff, seed=10)
    env = ep.env
    path = env.selected_buttons
    assert all(env.buttons_grid.index(b) in (0, 2, 4, 6, 8) for b in path)  # only the 5 raised targets are walked
    assert len(env.swing_directions) == len(path) - 1
    for name, d in zip([t["name"] for t in env.task_list if not t["demonstration"]], env.swing_directions):
        assert name.endswith(d)
    _drive_demo(ep)
    for prev, curr, d in zip(path[:-1], path[1:], env.swing_directions):
        _detour(ep, prev, curr, _sign(d))
        assert not ep.fail
    assert ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_rs_wrong_direction_fails_and_latches(rs, diff):
    ep = rs.make(diff, seed=10)
    env = ep.env
    path = env.selected_buttons
    _drive_demo(ep)
    _detour(ep, path[0], path[1], -_sign(env.swing_directions[0]))
    ep.step()
    assert ep.fail and not ep.success
    ep.step(3)
    assert ep.fail  # failure latches


def test_rs_straight_line_is_on_the_line_and_fails(rs):
    ep = rs.make("easy", seed=10)
    env = ep.env
    path = env.selected_buttons
    _drive_demo(ep)
    _detour(ep, path[0], path[1], 0.0)
    ep.step()
    assert ep.fail and not ep.success


def test_rs_touching_unexpected_raised_target_fails(rs):
    ep = rs.make("easy", seed=10)
    env = ep.env
    path = env.selected_buttons
    _drive_demo(ep)
    stray = next(env.buttons_grid[i] for i in (0, 2, 4, 6, 8)
                 if env.buttons_grid[i] is not path[0] and env.buttons_grid[i] is not path[1])
    _touch(ep, stray)
    assert ep.fail and not ep.success


def test_rs_direction_fail_degenerate_inputs(rs):
    """Degenerate inputs of direction_fail: no trajectory, zero-length segment -> failure (returns False and sets failureflag)."""
    ep = rs.make("easy", seed=10)
    env = ep.env
    a, b = env.buttons_grid[0], env.buttons_grid[2]
    env._gripper_xy_trace = []
    env.failureflag = torch.tensor([False])
    assert env.direction_fail([b, a, "clockwise"]) is False and bool(env.failureflag.item())
    env._gripper_xy_trace = [(1, torch.tensor([0.0, 0.0]))]
    env.failureflag = torch.tensor([False])
    assert env.direction_fail([a, a, "clockwise"]) is False and bool(env.failureflag.item())
    assert env.direction_fail(None) is True
