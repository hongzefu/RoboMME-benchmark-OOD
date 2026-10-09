"""官方包的失败路径（C09 异常恢复钩子、C13 入口依赖的 error 状态）。

- FailAwareWrapper：内层 step 抛任何 Exception → (None, 0.0, True, False, {status:error, error_message, exception_type})；
  BaseException（KeyboardInterrupt）不吞；正常步原样透传。
- EndeffectorDemonstrationWrapper：IK 失败显式返回 ({}, 0.0, True, False, {status:"error", error_message})，
  不调用内层 step（计划细则 4.7 保留的提醒）；IK 成功时把 7 关节解 + 夹爪交给内层；stick 任务不带夹爪。
- MultiStepDemonstrationWrapper：screw 3 次、RRT* 3 次都失败 → RRTPlanFailure；经 FailAwareWrapper 变成 error 状态。
不注入 sys.modules；规划器用实例属性替身。
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robomme.env_record_wrapper.EndeffectorDemonstrationWrapper import EndeffectorDemonstrationWrapper
from robomme.env_record_wrapper.FailAwareWrapper import FailAwareWrapper
from robomme.env_record_wrapper.MultiStepDemonstrationWrapper import (
    DATASET_RRT_MAX_ATTEMPTS,
    DATASET_SCREW_MAX_ATTEMPTS,
    MultiStepDemonstrationWrapper,
    RRTPlanFailure,
)
from robomme.robomme_env.utils.planner_fail_safe import ScrewPlanFailure

from _official_fakes import FakeTaskEnv


class _Boom(FakeTaskEnv):
    def __init__(self, exc, **kw):
        super().__init__(**kw)
        self.exc = exc

    def step(self, action):
        raise self.exc


# --------------------------------------------------------------------------- FailAwareWrapper


@pytest.mark.parametrize("exc", [ValueError("boom"), RuntimeError("ik"), KeyError("k"), RRTPlanFailure("rrt")])
def test_failaware_converts_exceptions(exc):
    env = FailAwareWrapper(_Boom(exc))
    out = env.step(np.zeros(8))
    assert out[:4] == (None, 0.0, True, False)
    info = out[4]
    assert set(info) == {"status", "error_message", "exception_type"}
    assert info["status"] == "error" and info["exception_type"] == type(exc).__name__
    assert str(exc) in info["error_message"]


def test_failaware_does_not_swallow_base_exceptions():
    env = FailAwareWrapper(_Boom(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        env.step(np.zeros(8))


def test_failaware_passthrough_and_last_obs():
    inner = FakeTaskEnv(outcomes=[(False, False)])
    env = FailAwareWrapper(inner)
    obs0, _ = env.reset()
    assert env._last_obs is obs0
    obs, r, term, trunc, info = env.step(np.arange(8))
    assert env._last_obs is obs and not bool(term) and "status" not in info
    np.testing.assert_array_equal(inner.actions[-1], np.arange(8))


# --------------------------------------------------------------------------- EndeffectorDemonstrationWrapper


class _IKPlanner:
    def __init__(self, status="Success", solutions=None):
        self.goals, self.status = [], status
        self.solutions = solutions if solutions is not None else [np.arange(9) / 10.0]
        self.planner = SimpleNamespace(transform_goal_to_wrt_base=self._to_base, IK=self._ik)
        self.robot = SimpleNamespace(get_qpos=lambda: torch.zeros(1, 9))

    def _to_base(self, goal):
        self.goals.append(np.asarray(goal).copy())
        return goal

    def _ik(self, goal_base, qpos):
        return self.status, self.solutions


def _ee(env_id="PickXtimes", planner=None, repr_="rpy"):
    inner = FakeTaskEnv(env_id=env_id, outcomes=[(False, False)])
    w = EndeffectorDemonstrationWrapper(inner, action_repr=repr_)
    w._ee_pose_planner = planner or _IKPlanner()
    return w, inner


@pytest.mark.parametrize("status, sols", [("Fail", []), ("Success", []), ("Fail", [np.zeros(9)])])
def test_ee_ik_failure_returns_error_without_stepping(status, sols):
    w, inner = _ee(planner=_IKPlanner(status, sols))
    out = w.step([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    assert out[0] == {} and out[1:4] == (0.0, True, False)
    assert out[4]["status"] == "error" and "IK failed" in out[4]["error_message"]
    assert inner.step_calls == 0


def test_ee_rpy_to_quat_and_joint_action():
    planner = _IKPlanner()
    w, inner = _ee(planner=planner)
    w.step([0.1, 0.2, 0.3, 0.0, 0.0, math.pi / 2, -1.0])
    goal = planner.goals[-1]
    # 手算：绕 z 转 90° 的 wxyz 四元数 = (cos45°, 0, 0, sin45°)
    np.testing.assert_allclose(goal, [0.1, 0.2, 0.3, math.sqrt(0.5), 0, 0, math.sqrt(0.5)], atol=1e-6)
    np.testing.assert_allclose(inner.actions[-1], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0])


def test_ee_quat_mode_passes_quat_and_gripper():
    planner = _IKPlanner()
    w, inner = _ee(planner=planner, repr_="quat")
    w.step([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(planner.goals[-1], [0.1, 0.2, 0.3, 1, 0, 0, 0])
    assert inner.actions[-1][-1] == 1.0 and inner.actions[-1].shape == (8,)


def test_ee_stick_env_drops_gripper():
    planner = _IKPlanner()
    w, inner = _ee(env_id="PatternLock", planner=planner)
    w.step([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])  # stick：6 维即可
    assert inner.actions[-1].shape == (7,)


@pytest.mark.parametrize("env_id, repr_, n, ok", [
    ("PickXtimes", "rpy", 6, False), ("PickXtimes", "rpy", 7, True),
    ("PickXtimes", "quat", 7, False), ("PickXtimes", "quat", 8, True),
    ("RouteStick", "rpy", 5, False), ("RouteStick", "rpy", 6, True),
    ("RouteStick", "quat", 6, False), ("RouteStick", "quat", 7, True),
])
def test_ee_action_length_validation(env_id, repr_, n, ok):
    w, _ = _ee(env_id=env_id, repr_=repr_)
    action = [0.1] * n
    if ok:
        w.step(action)
    else:
        with pytest.raises(ValueError, match="at least"):
            w.step(action)


def test_ee_rejects_unknown_repr():
    with pytest.raises(ValueError, match="Unsupported action_repr"):
        EndeffectorDemonstrationWrapper(FakeTaskEnv(), action_repr="euler")


def test_failaware_turns_short_ee_action_into_error():
    w, _ = _ee()
    out = FailAwareWrapper(w).step([0.1] * 3)
    assert out[4]["status"] == "error" and out[4]["exception_type"] == "ValueError"


# --------------------------------------------------------------------------- MultiStepDemonstrationWrapper


class _WaypointPlanner:
    def __init__(self, env, screw=(), rrt=()):
        self.env = env
        self.screw, self.rrt = list(screw), list(rrt)
        self.calls = {"screw": 0, "rrt": 0, "close": 0, "open": 0}

    def _run(self, kind, script, n_steps=2):
        self.calls[kind] += 1
        res = script.pop(0) if script else 0
        if isinstance(res, Exception):
            raise res
        if res != -1:
            for _ in range(n_steps):
                self.env.step(np.zeros(8))
        return res

    def move_to_pose_with_screw(self, pose):
        return self._run("screw", self.screw)

    def move_to_pose_with_RRTStar(self, pose):
        return self._run("rrt", self.rrt)

    def close_gripper(self):
        return self._run("close", [], n_steps=1)

    def open_gripper(self):
        return self._run("open", [], n_steps=1)


def _ms(env_id="PickXtimes", **script):
    inner = FakeTaskEnv(env_id=env_id)
    w = MultiStepDemonstrationWrapper(inner, gui_render=False, vis=False)
    w._planner = _WaypointPlanner(inner, **script)
    return w, inner


def test_waypoint_all_planning_fails_raises():
    w, _ = _ms(screw=[-1] * 3, rrt=[RuntimeError("x"), -1, -1])
    with pytest.raises(RRTPlanFailure):
        w.step([0.1, 0.2, 0.3, 0, 0, 0, 1])
    assert w._planner.calls["screw"] == DATASET_SCREW_MAX_ATTEMPTS
    assert w._planner.calls["rrt"] == DATASET_RRT_MAX_ATTEMPTS


def test_waypoint_failure_becomes_error_status():
    w, _ = _ms(screw=[-1] * 3, rrt=[-1] * 3)
    out = FailAwareWrapper(w).step([0.1, 0.2, 0.3, 0, 0, 0, 1])
    assert out[4]["status"] == "error" and out[4]["exception_type"] == "RRTPlanFailure"


def test_waypoint_screw_exception_retries_then_rrt_succeeds():
    w, inner = _ms(screw=[ScrewPlanFailure("a"), ScrewPlanFailure("b"), -1], rrt=[-1, 0])
    obs, r, term, trunc, info = w.step([0.1, 0.2, 0.3, 0, 0, 0, 0])
    assert w._planner.calls == {"screw": 3, "rrt": 2, "close": 0, "open": 0}
    assert len(obs["sensor_data"]) == 2  # 只收成功那次 RRT* 的 2 个底层步


@pytest.mark.parametrize("env_id, grip, closes, opens, n_steps", [
    ("PickXtimes", -1, 1, 0, 3), ("PickXtimes", 1, 0, 1, 3), ("PickXtimes", 0, 0, 0, 2),
    ("PatternLock", -1, 0, 0, 2), ("RouteStick", 1, 0, 0, 2),
])
def test_waypoint_gripper_actions(env_id, grip, closes, opens, n_steps):
    w, inner = _ms(env_id=env_id)
    obs, r, term, trunc, info = w.step([0.1, 0.2, 0.3, 0, 0, 0, grip])
    assert (w._planner.calls["close"], w._planner.calls["open"]) == (closes, opens)
    assert inner.step_calls == n_steps
    assert all(len(v) == n_steps for v in obs.values())  # dict-of-lists，每键一帧一项
    assert isinstance(term, torch.Tensor) and term.ndim == 0  # 只回最后一步的标量


def test_waypoint_short_action_rejected():
    w, _ = _ms()
    with pytest.raises(ValueError, match="at least 7"):
        w.step([0.1] * 6)


def test_waypoint_reset_and_close_drop_planner():
    w, inner = _ms()
    w.reset()
    assert w._planner is None
    w._planner = object()
    w.close()
    assert w._planner is None and inner.closed
