"""Official sequential_task_check and several predicates (C06 failure priority/advance one item at a time/switch permission/cache clearing; C07 threshold equality).

Calls the real functions directly with hand-written task tables (func/failure_func are small functions returning fixed values); expectations written item by item.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robomme.robomme_env.utils import subgoal_evaluate_func as sef
from tests.robomme_ood.unit.robomme import official_thresholds as T


def _env(**kw):
    return SimpleNamespace(**kw)


def _task(name, done=False, fail=None, demo=False, **extra):
    return {"func": lambda: done, "name": name, "failure_func": fail, "demonstration": demo, **extra}


def test_empty_list_is_complete():
    env = _env()
    assert sef.sequential_task_check(env, [], True) == (True, "No tasks", False, None)
    assert env.current_task_index == -1


def test_failure_checked_before_completion():
    env = _env()
    tasks = [_task("t0", done=True, fail=lambda: True)]
    assert sef.sequential_task_check(env, tasks, True) == (False, "t0", True, None)
    assert env.timestep == 0 and env.current_task_failure is True


def test_one_task_per_call_and_last_task():
    env = _env()
    tasks = [_task("t0", done=True), _task("t1", done=True, specialflag="swap"), _task("t2", done=True)]
    assert sef.sequential_task_check(env, tasks, True) == (False, "t1", False, None)
    assert env.timestep == 1
    assert sef.sequential_task_check(env, tasks, True) == (False, "t2", False, None)
    assert sef.sequential_task_check(env, tasks, True) == (True, "All tasks completed", False, None)
    assert env.timestep == 3
    assert sef.sequential_task_check(env, tasks, True)[:2] == (True, "All tasks completed")


def test_specialflag_returned_while_pending():
    env = _env()
    tasks = [_task("t0", specialflag="swap")]
    assert sef.sequential_task_check(env, tasks, True) == (False, "t0", False, "swap")
    assert env.current_task_specialflag == "swap"


def test_subgoal_switch_permission():
    env = _env()
    tasks = [_task("first", demo=True, choice_label="c0"), _task("second")]
    sef.sequential_task_check(env, tasks, True)
    assert env.current_task_name == "first" and env.current_task_demonstration is True
    tasks[0]["func"] = lambda: True
    sef.sequential_task_check(env, tasks, False)   # complete and advance, but switching the "external" subgoal is not allowed
    sef.sequential_task_check(env, tasks, False)
    assert env.timestep == 1 and env.current_task_name == "first"
    assert env.current_task_name_online == "second"  # the real-time subgoal is updated as usual
    sef.sequential_task_check(env, tasks, True)
    assert env.current_task_name == "second" and env.current_task_demonstration is False


def test_static_cache_cleared_on_task_switch():
    env = _env(first_timestep=5)
    tasks = [_task("t0", done=True), _task("t1")]
    sef.sequential_task_check(env, tasks, True)  # first call: _last_task_index None -> 0, clears the old cache
    assert not hasattr(env, "first_timestep")
    env.first_timestep = 7
    sef.sequential_task_check(env, tasks, True)  # switch to t1: clears again
    assert not hasattr(env, "first_timestep")


def test_alternate_key_names():
    env = _env()
    tasks = [{"task_func": lambda: False, "task_name": "alt", "demo": True, "failure": lambda: [0, None]}]
    assert sef.sequential_task_check(env, tasks, True) == (False, "alt", False, None)
    assert env.current_task_demonstration is True
    with pytest.raises(KeyError):
        sef.sequential_task_check(_env(), [{"name": "nofunc"}], True)


@pytest.mark.parametrize("value, failed", [
    (None, False), (False, False), (True, True), ([False, None], False), ([False, True], True),
    ({"a": 0, "b": torch.tensor([True])}, True), (torch.tensor([]), False), (torch.tensor([False, True]), True),
    (np.array([]), False), (np.array([0, 0]), False), ((0, (0, 1)), True),
])
def test_failure_value_coercion(value, failed):
    env = _env()
    tasks = [_task("t", fail=lambda v=value: v)]
    assert sef.sequential_task_check(env, tasks, True)[2] is failed


def test_failure_func_exception_counts_as_failure():
    def boom():
        raise RuntimeError("x")

    assert sef.sequential_task_check(_env(), [_task("t", fail=boom)], True)[2] is True


def test_non_callable_failure_value_is_constant():
    assert sef.sequential_task_check(_env(), [_task("t", fail=True)], True)[2] is True
    assert sef.sequential_task_check(_env(), [_task("t", fail=False)], True)[2] is False


# --------------------------------------------------------------------------- Predicates


@pytest.mark.parametrize("stop, inside", [(60, True), (90, True), (120, True), (59, False), (121, False)])
def test_correct_timestep_inclusive_window(stop, inside):
    assert sef.correct_timestep(_env(elapsed_steps=0), time_range=(60, 120), stop_timestep=stop) is inside


def test_timewindow_starts_on_first_call():
    env = _env(elapsed_steps=10)
    f = lambda: True  # noqa: E731
    assert sef.timewindow(env, f, timewindow_timer=1, min_steps=5, max_steps=8) is False   # elapsed 0
    for step, ok in ((14, False), (15, True), (18, True), (19, False)):
        env.elapsed_steps = step
        assert sef.timewindow(env, f, timewindow_timer=1, min_steps=5, max_steps=8) is ok, step


def _actor(x, y, z=0.0):
    return SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[x, y, z]], dtype=torch.float32)))


def test_stopped_onto_latches_first_stop_step():
    env = _env(cube_half_size=T.CUBE_HALF, elapsed_steps=30)
    target = _actor(0, 0)
    assert sef.is_obj_stopped_onto(env, _actor(T.STOP_ONTO_XY, 0), target, stop=True) is True   # inclusive
    assert env.stop_timestep == 30
    env.elapsed_steps = 40
    assert sef.is_obj_stopped_onto(env, _actor(0, 0), target, stop=True) is True
    assert env.stop_timestep == 30  # not rewritten
    assert sef.is_obj_stopped_onto(_env(cube_half_size=T.CUBE_HALF), _actor(T.STOP_ONTO_XY + T.EPS, 0), target, stop=True) is False
    assert sef.is_obj_stopped_onto(_env(cube_half_size=T.CUBE_HALF), _actor(0, 0), target, stop=False) is False


@pytest.mark.parametrize("dx, dy, label8, label4", [
    (1, 0, "forward", "forward"), (-1, 0, "backward", "backward"), (0, 1, "left", "left"), (0, -1, "right", "right"),
    (1, 1, "forward-left", "forward"), (-1, -1, "backward-right", "backward"), (0, 0, "same", "same"),
])
def test_direction_compass(dx, dy, label8, label4):
    a, b = _actor(dx, dy), _actor(0, 0)
    assert sef.direction(a, b) == label8
    # with 4 directions the dot products on both sides of the diagonal are equal; take forward/backward, listed first
    assert sef.direction(a, b, direction=4) == label4


def test_check_in_bin_number():
    assert sef.check_in_bin_number(None, [1, 0, 2], [1, 0, 2]) is True
    assert sef.check_in_bin_number(None, [1, 0, 1], [1, 0, 2]) is False
    assert sef.check_in_bin_number(None, [1, 0], [1, 0, 0]) is False


class _Btn:
    def __init__(self, depth):
        self.depth = depth

    def get_qpos(self):
        return torch.tensor([[-self.depth]])


@pytest.mark.parametrize("depth, pressed", [(T.BUTTON_DEPTH + T.EPS, True), (T.BUTTON_DEPTH, False), (0.0, False)])
def test_button_depth_strict(depth, pressed):
    env = _env(button=object())
    assert bool(sef.is_button_pressed(env, _Btn(depth))) is pressed


def test_pressed_buttons_removed_from_list():
    env = _env(button=object())
    a, b, c = _Btn(2 * T.BUTTON_DEPTH), _Btn(0.0), _Btn(2 * T.BUTTON_DEPTH)
    lst = [a, b, c]
    assert sef.is_any_button_pressed_removelist(env, lst) is True and lst == [b]
    assert sef.is_any_button_pressed_removelist(env, lst) is False
    assert sef.is_any_button_pressed_removelist(env, []) is False
