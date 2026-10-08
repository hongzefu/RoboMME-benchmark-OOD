"""官方 DemonstrationWrapper 的终局状态、截断与动作规整（C06 失败优先、C09 时序）。

手写事件序列（内层替身 env 每步返回脚本里的 success/fail），期望：
- status 优先级 success > fail > timeout > ongoing（同一步既成功又失败 → success；既失败又到步数上限 → fail）；
- 非演示子任务连续步数达到 max_steps_without_demonstration → truncated → timeout；演示子任务不计数；
- terminated 时额外多走一个底层步（同一动作）录下末帧，额外步不改动外层的 ee 连续性缓存；
- 动作规整：stick 任务取前 7 维、其余取前 8 维，不足即 ValueError；
- reset：演示批 + 一个初始动作步拼接后过滤 NO RECORD 帧；初始动作 = home 位（stick 用 swing_qpos）；清零跨局缓存。
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
from robomme.robomme_env.utils import planner_denseStep

from _official_fakes import FakeTaskEnv, as_made
from tests.robomme_hard.unit.robomme import official_thresholds as T

# 包的 __init__ 用同名类覆盖了子模块属性，按模块路径取模块本身
dw_module = importlib.import_module("robomme.env_record_wrapper.DemonstrationWrapper")



def make(env_id="PickXtimes", outcomes=(), max_steps=100, demo_batch=None, demonstration=False, **flags):
    inner = FakeTaskEnv(env_id=env_id, outcomes=[(False, False)] + list(outcomes), demonstration=demonstration)
    w = DemonstrationWrapper(as_made(inner), max_steps_without_demonstration=max_steps, gui_render=False, **flags)
    batch = demo_batch if demo_batch is not None else planner_denseStep.empty_step_batch()
    # 演示轨迹由运动规划生成（CPU 跑不了）；这里给定一个手写的演示批
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
    # reset 的初始动作步计 1 步；上限 3 → 第 2 次在线 step 达到 3
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
    assert all(len(v) == 1 for v in obs.values())  # 额外步只为录末帧，不进本步返回


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
    w.step(np.zeros(8))           # 普通步：拿到 reset 那一步的缓存
    calls.clear()
    w.step(np.zeros(8))           # 终局步：先跑额外步，再跑外层 augment
    extra_prev, outer_prev = calls
    assert torch.equal(extra_prev[0], outer_prev[0])  # 外层拿到的仍是额外步之前的缓存


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
    np.testing.assert_allclose(inner.actions[0], np.full(7, 0.5))  # swing_qpos 的前 7 维


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
    assert info["simple_subgoal_online"] == inner.current_task_name  # 平铺 info 取最后一帧
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
    assert w.episode_success is False and w.steps_without_demonstration == 1  # 只剩 reset 的初始步
    assert w.latched_replacements is None  # 上一局锁存的占位符坐标被清掉


def test_step_returns_last_scalars_and_flat_info():
    w, inner, *_ = make(outcomes=[(False, False)])
    obs, r, term, trunc, info = w.step(np.zeros(8))
    assert isinstance(r, torch.Tensor) and r.ndim == 0 and r.dtype == torch.float32
    assert term.dtype == torch.bool and trunc.dtype == torch.bool
    assert all(isinstance(v, list) and len(v) == 1 for v in obs.values())
    assert isinstance(info["status"], str) and isinstance(info["task_goal"], list)
