"""Terminal status, truncation and action normalization of the official DemonstrationWrapper (C06 failure priority, C09 timing).

Hand-written event sequences (the inner env double returns the scripted success/fail each step); expectations:
- status priority success > fail > timeout > ongoing (success and fail on the same step -> success; fail and step cap reached -> fail);
- consecutive non-demo subtask steps reaching max_steps_without_demonstration -> truncated -> timeout; demo subtasks are not counted;
- on terminated, one extra low-level step (same action) is taken to record the last frame; the extra step does not change the outer ee continuity cache;
- action normalization: stick tasks take the first 7 dims, others the first 8; fewer raises ValueError;
- reset: demo batch + one initial action step concatenated, then NO RECORD frames filtered; initial action = home pose (stick uses swing_qpos); cross-episode caches cleared.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
from robomme.robomme_env.utils import planner_denseStep

from _official_fakes import FakeTaskEnv, as_made
from tests.robomme_ood.unit.robomme import official_thresholds as T

# the package __init__ shadows the submodule attribute with a same-named class; fetch the module itself by module path
dw_module = importlib.import_module("robomme.env_record_wrapper.DemonstrationWrapper")



def make(env_id="PickXtimes", outcomes=(), max_steps=100, demo_batch=None, demonstration=False, **flags):
    inner = FakeTaskEnv(env_id=env_id, outcomes=[(False, False)] + list(outcomes), demonstration=demonstration)
    w = DemonstrationWrapper(as_made(inner), max_steps_without_demonstration=max_steps, gui_render=False, **flags)
    batch = demo_batch if demo_batch is not None else planner_denseStep.empty_step_batch()
    # demo trajectories come from motion planning (cannot run on CPU); a hand-written demo batch is given here
    w.get_demonstration_trajectory = lambda: batch
    obs, info = w.reset()
    return w, inner, obs, info


@pytest.mark.parametrize("success, fail, status, terminated", [
    (False, False, "ongoing", False),
    (True, False, "success", True),
    (False, True, "fail", True),
    (True, True, "success", True),
])
def test_status_priority(success, fail, status, terminated):
    w, inner, *_ = make(outcomes=[(success, fail)])
    obs, r, term, trunc, info = w.step(np.zeros(8))
    assert info["status"] == status
    assert bool(term) is terminated and bool(trunc) is False
    assert w.episode_success is success


def test_timeout_after_no_demo_steps():
    # the reset initial action step counts as 1 step; cap 3 -> the 2nd online step reaches 3
    w, inner, *_ = make(max_steps=3)
    assert w.step(np.zeros(8))[4]["status"] == "ongoing"
    obs, r, term, trunc, info = w.step(np.zeros(8))
    assert info["status"] == "timeout" and bool(trunc) and not bool(term)


@pytest.mark.parametrize("outcome, status", [((False, True), "fail"), ((True, False), "success")])
def test_timeout_loses_to_terminal_outcome(outcome, status):
    w, inner, *_ = make(max_steps=2, outcomes=[outcome])
    assert w.step(np.zeros(8))[4]["status"] == status


def test_demonstration_steps_do_not_count_towards_limit():
    w, inner, *_ = make(max_steps=2, demonstration=True)
    for _ in range(5):
        assert w.step(np.zeros(8))[4]["status"] == "ongoing"
    assert w.steps_without_demonstration == 0


def test_terminal_step_takes_one_extra_low_level_step():
    w, inner, *_ = make(outcomes=[(True, False), (True, False)])
    before = inner.step_calls
    action = np.arange(8, dtype=np.float64)
    obs, *_rest = w.step(action)
    assert inner.step_calls - before == 2
    np.testing.assert_array_equal(inner.actions[-1], inner.actions[-2])
    assert all(len(v) == 1 for v in obs.values())  # the extra step only records the last frame and is not part of this step's return


def test_extra_step_does_not_disturb_pose_continuity(monkeypatch):
    calls = []
    real = dw_module.build_endeffector_pose_dict
    counter = iter(range(1000))

    def spy(p, q, prev_q, prev_rpy):
        pose, _, _ = real(p, q, None, None)
        k = next(counter)
        calls.append((prev_q, prev_rpy))
        return pose, torch.tensor([float(k)]), torch.tensor([float(k)])

    monkeypatch.setattr(dw_module, "build_endeffector_pose_dict", spy)
    w, inner, *_ = make(outcomes=[(False, False), (True, False), (True, False)])
    w.step(np.zeros(8))           # normal step: gets the cache from the reset step
    calls.clear()
    w.step(np.zeros(8))           # terminal step: runs the extra step first, then the outer augment
    extra_prev, outer_prev = calls
    assert torch.equal(extra_prev[0], outer_prev[0])  # the outer layer still gets the cache from before the extra step


@pytest.mark.parametrize("env_id, n_in, n_out", [("PickXtimes", 10, 8), ("PickXtimes", 8, 8),
                                                 ("PatternLock", 9, 7), ("RouteStick", 7, 7)])
def test_action_normalization(env_id, n_in, n_out):
    w, inner, *_ = make(env_id=env_id)
    w.step(np.arange(n_in, dtype=np.float64))
    np.testing.assert_array_equal(inner.actions[-1], np.arange(n_out))


@pytest.mark.parametrize("env_id, n_in", [("PickXtimes", 7), ("RouteStick", 6)])
def test_action_too_short_raises(env_id, n_in):
    w, inner, *_ = make(env_id=env_id)
    with pytest.raises(ValueError, match="at least"):
        w.step(np.zeros(n_in))


def test_reset_initial_action_home_or_swing():
    _, inner, *_ = make("PickXtimes")
    np.testing.assert_allclose(inner.actions[0], T.HOME_ACTION)
    _, inner, *_ = make("RouteStick")
    np.testing.assert_allclose(inner.actions[0], np.full(7, 0.5))  # first 7 dims of swing_qpos


def _demo_batch(subgoals):
    steps = [({"front_rgb_list": np.full((2, 2, 3), i, np.uint8)}, torch.tensor([0.0]), torch.tensor([False]),
              torch.tensor([False]), {"simple_subgoal_online": s, "status": "ongoing"})
             for i, s in enumerate(subgoals)]
    return planner_denseStep.to_step_batch(steps)


def test_reset_filters_no_record_and_appends_init_step():
    w, inner, obs, info = make(demo_batch=_demo_batch(["watch", "NO RECORD", "  NO RECORD ", "watch again"]))
    subgoals = w.demonstration_data[4]["simple_subgoal_online"]
    assert subgoals == ["watch", "watch again", inner.current_task_name]
    assert len(obs["front_rgb_list"]) == 3
    assert info["simple_subgoal_online"] == inner.current_task_name  # flattened info takes the last frame
    assert info["status"] == "ongoing"


def test_filter_keeps_batch_when_everything_is_no_record():
    w, *_ = make()
    batch = _demo_batch(["NO RECORD", "NO RECORD"])
    assert w._filter_no_record_from_step_batch(batch) is batch


def test_reset_clears_cross_episode_state():
    w, inner, *_ = make(outcomes=[(True, False)])
    w.step(np.zeros(8))
    assert w.episode_success and w._prev_ee_quat_wxyz is not None
    w.steps_without_demonstration = 41
    w.latched_replacements = ["<1, 2>"]
    inner.outcomes = [(False, False)]
    w.reset()
    assert w.episode_success is False and w.steps_without_demonstration == 1  # only the reset initial step remains
    assert w.latched_replacements is None  # placeholder coordinates latched in the previous episode are cleared


def test_step_returns_last_scalars_and_flat_info():
    w, inner, *_ = make(outcomes=[(False, False)])
    obs, r, term, trunc, info = w.step(np.zeros(8))
    assert isinstance(r, torch.Tensor) and r.ndim == 0 and r.dtype == torch.float32
    assert term.dtype == torch.bool and trunc.dtype == torch.bool
    assert all(isinstance(v, list) and len(v) == 1 for v in obs.values())
    assert isinstance(info["status"], str) and isinstance(info["task_goal"], list)
