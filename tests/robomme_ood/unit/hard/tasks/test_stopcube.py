"""StopCube new-value tier truth table (xhard1-xhard5): the cube moves back and forth along a line; pressing the button to stop it on the N-th pass over the target is success.

The clock advances through the task class's real ``step`` (cube motion ``move_straight_line`` and "stop on press" are both in step):
pressing at the midpoint of the N-th pass window → success; pressing on the (N−1)-th pass → failure; never pressing, past segment N → failure;
terminal state unchanged when advancing after success. N and cadence come from this episode (packaged spec replay).
"""
from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from robomme_ood.robomme_env.utils.subgoal_evaluate_func import is_obj_stopped_onto

from .. import offline_scene as O
from ..world import World, cpu_world

TASK = "StopCube"
TIERS = O.tiers_of(TASK)


@pytest.mark.parametrize("difficulty", ["missing", None, "easy", "medium", "hard", "xhard1", "xhard2", "xhard3", "xhard4", "xhard5"])
@pytest.mark.parametrize("distance", [0.049, 0.050, 0.051, 0.055, 0.059, 0.060, 0.061])
def test_stop_threshold_difficulty_matrix(difficulty, distance):
    env = SimpleNamespace(cube_half_size=0.02, elapsed_steps=17)
    if difficulty != "missing":
        env.difficulty = difficulty
    obj = SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[distance, 0, 0]], dtype=torch.float64)))
    target = SimpleNamespace(pose=SimpleNamespace(p=torch.zeros((1, 3), dtype=torch.float64)))
    allowed = ({0.049, 0.050} if difficulty in {"xhard1", "xhard2", "xhard3", "xhard4", "xhard5"}
               else {0.049, 0.050, 0.051, 0.055, 0.059, 0.060})
    assert bool(is_obj_stopped_onto(env, obj, target, True)) == (distance in allowed)
    assert getattr(env, "stop_timestep", None) == (17 if distance in allowed else None)


def test_stop_requires_stopped_and_preserves_first_timestep():
    env = SimpleNamespace(cube_half_size=0.02, difficulty="xhard5", elapsed_steps=17)
    obj = SimpleNamespace(pose=SimpleNamespace(p=torch.zeros((1, 3), dtype=torch.float64)))
    assert is_obj_stopped_onto(env, obj, obj, False) is False
    assert not hasattr(env, "stop_timestep")
    assert is_obj_stopped_onto(env, obj, obj, True) is True
    assert env.stop_timestep == 17
    env.elapsed_steps = 99
    assert is_obj_stopped_onto(env, obj, obj, True) is True
    assert env.stop_timestep == 17


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _hover(w):
    """First item "move above the button": TCP stops low directly above the button."""
    x, y = w.xyz(w.env.button)[:2]
    w.tcp_to((x, y, 0.1))


def _run_until(w, step_index):
    out = None
    while int(w.env.elapsed_steps) < step_index:
        out = w.step()
        if out["fail"]:
            return out
    return out


def _press_at(w, step_index):
    _hover(w)
    out = _run_until(w, step_index)
    if out is not None and out["fail"]:
        return out
    w.press(w.env.button)
    return w.step()


@pytest.mark.parametrize("tier", TIERS)
def test_press_on_nth_pass_succeeds(world, tier):
    w = world(tier)
    t, n = w.env.move_interval, w.env.stop_time
    out = _press_at(w, (n - 1) * t + t // 2)  # midpoint of segment n: the cube is passing the target
    assert out == {"success": True, "fail": False}
    lo, hi = w.env.stop_time_range
    assert lo <= w.env.stop_timestep <= hi
    for _ in range(3):
        assert w.step()["success"] is True, "terminal state unchanged when advancing after success"


@pytest.mark.parametrize("tier", TIERS)
def test_press_on_previous_pass_fails(world, tier):
    w = world(tier)
    t, n = w.env.move_interval, w.env.stop_time
    out = _press_at(w, (n - 2) * t + t // 2)  # (n−1)-th pass
    w.unpress(w.env.button)
    out = w.step() if not out["fail"] else out
    # stopped on the target but not on the n-th pass: either judged wrong immediately (not on target when pressed), or the timing mismatches after the task table completes
    for _ in range(t * 2):
        if out["fail"]:
            break
        out = w.step()
    assert out == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:2])
def test_never_pressing_fails_after_window(world, tier):
    w = world(tier)
    t, n = w.env.move_interval, w.env.stop_time
    _hover(w)
    out = _run_until(w, n * t + 2)
    assert out == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_press_off_target_fails_immediately(world, tier):
    """Pressing when the cube is not on the target (segment start, cube at the route endpoint) → immediate failure."""
    w = world(tier)
    t, n = w.env.move_interval, w.env.stop_time
    out = _press_at(w, (n - 1) * t + 1)
    assert out == {"success": False, "fail": True}
