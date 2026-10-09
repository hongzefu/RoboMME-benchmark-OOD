"""StopCube native three-tier truth table (C05, C06 timing and scene motion).

The cube shuttles between two endpoints (real move_straight_line). Arrival steps, stop window and deadline are not derived by formula in the test; they are hand-computed for a concrete
(move_interval, stop_time) and pinned in ``official_thresholds.STOPCUBE_CASES``; the test picks seeds by these two values. Expectations
(goal language: "stop the cube ... for the k-th time"):
- first hover the tcp above the button (preparation subtask), press at the stop_time-th arrival -> success, and success persists on further steps;
- pressing at the (stop_time-1)-th arrival -> failure (stop step outside the window); pressing while the cube is not on the target -> immediate failure;
- never pressing -> failure once past the deadline. All three tiers share the same logic in the official implementation (difficulty does not affect values).
"""
from __future__ import annotations

import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

TASK = "StopCube"
DIFFS = ("easy", "medium", "hard")
ORDINALS = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}
CASES = T.STOPCUBE_CASES


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _case_episode(world, diff, case):
    """Pick a seed whose (move_interval, stop_time) matches the pinned case (offline _initialize_episode, not a simulation reset)."""
    for seed in range(64):
        ep = world.make(diff, seed=seed)
        if (ep.env.move_interval, ep.env.stop_time) == (case["move_interval"], case["stop_time"]):
            return ep
    pytest.fail(f"no seed within 64 matches {case}")


def _hover(ep):
    bx, by, _ = ep.env.button.xyz
    ep.tcp_to(bx, by, T.LIFT_Z)


def _run_to(ep, elapsed):
    while int(ep.env.elapsed_steps) < elapsed:
        ep.step()


def _press_at(ep, step):
    """Advance until elapsed == step, then press the button (the cube position seen by the next evaluate is exactly move_straight_line(cur_step=step))."""
    _run_to(ep, step)
    ep.press(ep.env.button)
    ep.step()
    ep.unpress(ep.env.button)


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"mi{c['move_interval']}-st{c['stop_time']}")
def test_stop_on_the_nth_visit_succeeds_and_persists(world, diff, case):
    ep = _case_episode(world, diff, case)
    env = ep.env
    st = case["stop_time"]
    goals = goal_text(env)
    assert f"for the {ORDINALS[st]} time" in goals[0] and f"on its {ORDINALS[st]} visit" in goals[1]
    _hover(ep)
    ep.step()
    assert ep.task_index == 1  # preparation subtask complete
    _press_at(ep, case["visits"][st - 1])
    ep.step(3)
    assert ep.success and not ep.fail
    locked = env.stop_timestep
    lo, hi = case["window"]
    assert lo <= locked <= hi
    _run_to(ep, case["deadline"] + case["move_interval"])  # continue after success: still success past the deadline, stop step not rewritten
    assert ep.success and not ep.fail
    assert env.stop_timestep == locked


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"mi{c['move_interval']}-st{c['stop_time']}")
def test_stop_on_previous_visit_fails(world, diff, case):
    ep = _case_episode(world, diff, case)
    _hover(ep)
    ep.step()
    _press_at(ep, case["visits"][case["stop_time"] - 2])
    _run_to(ep, case["deadline"])
    assert any(f for _, f, _ in ep.history)
    assert not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_press_off_target_fails_immediately(world, diff):
    case = CASES[0]
    ep = _case_episode(world, diff, case)
    _hover(ep)
    ep.step()
    _press_at(ep, case["off_target_step"])
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"mi{c['move_interval']}-st{c['stop_time']}")
def test_never_pressing_times_out(world, diff, case):
    ep = _case_episode(world, diff, case)
    _hover(ep)
    _run_to(ep, case["deadline"])
    assert not ep.fail
    ep.step()
    assert ep.fail and not ep.success


def test_without_hover_the_final_subgoal_is_never_reached(world):
    case = CASES[0]
    ep = _case_episode(world, "easy", case)
    _press_at(ep, case["visits"][case["stop_time"] - 1])  # did not hover over the button first: pointer stays at the preparation subtask, pressing at the right moment does not count
    _run_to(ep, case["deadline"] + 1)
    assert ep.fail and not ep.success


def test_stop_freezes_cube(world):
    case = CASES[0]
    ep = _case_episode(world, "easy", case)
    env = ep.env
    _hover(ep)
    ep.step()
    _press_at(ep, case["visits"][case["stop_time"] - 1])
    frozen = env.cube.xyz.copy()
    ep.step(10)
    assert (env.cube.xyz == frozen).all()
