"""PickHighlight native three-tier truth table (C05): press the button first, then pick up each highlighted cube once (putting it down in between).

Expectations: button -> pick up each highlighted cube in turn -> success; missing one highlighted cube -> no success; picking a non-highlighted cube (during a pick subtask) -> failure;
picking the same highlighted cube twice does not count as a second one.

Current official behavior (registered as conditional in the contract delta, plan Q13 "PickHighlight failure callback conflicts with language"):
- the button subtask's failure_func is not a lambda but a constant False evaluated once at _load_scene, so picking any cube before pressing the button never fails;
- success only checks the count "every highlighted cube picked >= 1 time" and does not require the button press -- picking all highlighted cubes without pressing the button also succeeds;
- the second goal sentence says "finally press the button again to stop", but the task list has no final button subtask.
"""
from __future__ import annotations

import pytest

from _official_world import OfficialWorld, goal_text

TASK = "PickHighlight"
DIFFS = ("easy", "medium", "hard")


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _press(ep):
    ep.press(ep.env.button)
    ep.step()
    ep.unpress(ep.env.button)


def _pick_targets(ep, targets):
    for i, cube in enumerate(targets):
        ep.grasp(cube)
        ep.step()
        if i != len(targets) - 1:
            ep.release(cube)
            ep.step()


@pytest.mark.parametrize("diff", DIFFS)
def test_button_then_all_highlighted_succeeds(world, diff):
    ep = world.make(diff, seed=7)
    env = ep.env
    assert len(env.target_cubes) == env.configs[diff]["pickup"]
    _press(ep)
    _pick_targets(ep, env.target_cubes[:-1])
    if len(env.target_cubes) > 1:
        ep.release(env.target_cubes[-2])
        ep.step()
        assert not ep.success  # missing the last one: no success
    _pick_targets(ep, env.target_cubes[-1:])
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", ("medium", "hard"))
def test_repick_same_highlight_is_not_second(world, diff):
    ep = world.make(diff, seed=7)
    env = ep.env
    _press(ep)
    first = env.target_cubes[0]
    for _ in range(3):
        ep.grasp(first)
        ep.step()
        ep.release(first)
        ep.step()
    assert not ep.success and not ep.fail
    assert env.target_cube_pickup_counts[env.target_cube_names[0]] == 3
    assert all(v == 0 for k, v in env.target_cube_pickup_counts.items() if k != env.target_cube_names[0])


@pytest.mark.parametrize("diff", DIFFS)
def test_non_highlighted_pick_fails_in_pick_phase(world, diff):
    ep = world.make(diff, seed=7)
    env = ep.env
    others = [c for c in env.all_cubes if all(c is not t for t in env.target_cubes)]
    assert others
    _press(ep)
    ep.grasp(others[0])
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_status_quo_button_failure_is_constant(world, diff):
    """The button subtask's failure condition is a constant computed at load time: picking a non-highlighted cube before pressing the button does not fail."""
    ep = world.make(diff, seed=7)
    env = ep.env
    assert env.task_list[0]["failure_func"] is False
    others = [c for c in env.all_cubes if all(c is not t for t in env.target_cubes)]
    ep.grasp(others[0])
    ep.step()
    assert not ep.fail


@pytest.mark.parametrize("diff", DIFFS)
def test_status_quo_success_without_button(world, diff):
    """Picking each highlighted cube once without pressing the button -> success (conflicts with the "first press the button" language)."""
    ep = world.make(diff, seed=7)
    env = ep.env
    assert all("first press the button" in g for g in goal_text(env))
    assert any("press the button again" in g for g in goal_text(env))
    assert all("button" not in (t.get("name") or "") for t in env.task_list[1:])  # no final button subtask
    _pick_targets(ep, env.target_cubes)
    assert ep.success and ep.task_index == 0
