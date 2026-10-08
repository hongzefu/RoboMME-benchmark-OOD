"""DemonstrationWrapper._augment_obs_and_info 的 8 个 include_* 开关（C08）。

日常门禁跑 18 组（全关、全开、8 个单开、8 个单关）；四种动作空间 × 256 全组合（经真实 BenchmarkEnvBuilder，
multi_choice 强制带前视内外参）标 slow。期望由替身观测手算：
- 恒有的 obs 五键：front/wrist RGB (H,W,3) uint8、joint (7,) = qpos 前 7 维、eef (6,) float64 = tcp 位置 + rpy、
  gripper (2,) = qpos 第 8、9 维（stick 任务恒为 0）；
- 开关键：深度 (H,W,1) int16、外参 (3,4) float32（去 batch 维）、内参 (3,3) float32 进 info、
  available_multi_choices 是 {label, action, need_parameter} 列表；CPU Tensor 一律转成 NumPy。
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from robomme.env_record_wrapper import episode_config_resolver as ecr
from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder
from robomme.robomme_env.utils import planner_denseStep

from _official_fakes import H, W, FakeTaskEnv, GymMakeSpy, as_made, find_wrapper

FLAGS = (
    "include_maniskill_obs", "include_front_depth", "include_wrist_depth", "include_front_camera_extrinsic",
    "include_wrist_camera_extrinsic", "include_available_multi_choices", "include_front_camera_intrinsic",
    "include_wrist_camera_intrinsic",
)
OBS_KEY = {
    "include_maniskill_obs": "maniskill_obs", "include_front_depth": "front_depth_list",
    "include_wrist_depth": "wrist_depth_list", "include_front_camera_extrinsic": "front_camera_extrinsic_list",
    "include_wrist_camera_extrinsic": "wrist_camera_extrinsic_list",
}
INFO_KEY = {
    "include_available_multi_choices": "available_multi_choices",
    "include_front_camera_intrinsic": "front_camera_intrinsic",
    "include_wrist_camera_intrinsic": "wrist_camera_intrinsic",
}
BASE_OBS = {"front_rgb_list", "wrist_rgb_list", "joint_state_list", "eef_state_list", "gripper_state_list"}
BASE_INFO = {"success", "fail", "simple_subgoal_online", "grounded_subgoal_online", "task_goal", "status"}


def combos_18():
    off = dict.fromkeys(FLAGS, False)
    on = dict.fromkeys(FLAGS, True)
    return [off, on] + [{**off, f: True} for f in FLAGS] + [{**on, f: False} for f in FLAGS]


def _reset_with_empty_demo(w):
    w.get_demonstration_trajectory = lambda: planner_denseStep.empty_step_batch()
    return w.reset()


def _check(obs, info, flags, step_idx, stick=False):
    assert set(obs) == BASE_OBS | {OBS_KEY[f] for f in OBS_KEY if flags[f]}
    assert set(info) == BASE_INFO | {INFO_KEY[f] for f in INFO_KEY if flags[f]}
    rgb = obs["front_rgb_list"][-1]
    assert isinstance(rgb, np.ndarray) and rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
    assert (rgb == step_idx).all()
    np.testing.assert_allclose(obs["joint_state_list"][-1], np.arange(7) / 10, rtol=1e-6)
    eef = obs["eef_state_list"][-1]
    assert eef.dtype == np.float64 and eef.shape == (6,)
    np.testing.assert_allclose(eef, [0.1, 0.2, 0.3, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(obs["gripper_state_list"][-1], [0, 0] if stick else [0.7, 0.8], rtol=1e-6)
    if flags["include_front_depth"]:
        d = obs["front_depth_list"][-1]
        assert d.dtype == np.int16 and d.shape == (H, W, 1) and (d == 100 + step_idx).all()
    if flags["include_front_camera_extrinsic"]:
        e = obs["front_camera_extrinsic_list"][-1]
        assert e.dtype == np.float32 and e.shape == (3, 4)
        np.testing.assert_array_equal(e, np.arange(12).reshape(3, 4))
    if flags["include_wrist_camera_extrinsic"]:
        np.testing.assert_array_equal(obs["wrist_camera_extrinsic_list"][-1], np.arange(12).reshape(3, 4) + 100)
    if flags["include_front_camera_intrinsic"]:
        i = info["front_camera_intrinsic"]
        assert i.dtype == np.float32 and i.shape == (3, 3)
        np.testing.assert_array_equal(i, np.eye(3))
    if flags["include_wrist_camera_intrinsic"]:
        np.testing.assert_array_equal(info["wrist_camera_intrinsic"], np.eye(3) * 2)
    if flags["include_available_multi_choices"]:
        opts = info["available_multi_choices"]
        assert all(set(o) == {"label", "action", "need_parameter"} for o in opts)
        assert [o["label"] for o in opts] == ["a", "b", "c"]  # PickXtimes：拾起／放到目标／按钮
        assert [o["need_parameter"] for o in opts] == [True, False, False]
    if flags["include_maniskill_obs"]:
        assert set(obs["maniskill_obs"][-1]) == {"sensor_data", "sensor_param"}


@pytest.mark.parametrize("combo", range(18))
def test_switches_18(combo):
    flags = combos_18()[combo]
    inner = FakeTaskEnv()
    w = DemonstrationWrapper(as_made(inner), max_steps_without_demonstration=99, gui_render=False, **flags)
    _reset_with_empty_demo(w)
    obs, _r, _t, _tr, info = w.step(np.zeros(8))
    _check(obs, info, flags, step_idx=inner.step_calls)


def test_stick_task_gripper_is_zero():
    flags = dict.fromkeys(FLAGS, False)
    inner = FakeTaskEnv(env_id="RouteStick")
    w = DemonstrationWrapper(as_made(inner), max_steps_without_demonstration=99, gui_render=False, **flags)
    _reset_with_empty_demo(w)
    obs, *_rest, info = w.step(np.zeros(7))
    assert obs["joint_state_list"][-1].shape == (7,)
    np.testing.assert_array_equal(obs["gripper_state_list"][-1], [0.0, 0.0])


def test_frames_are_snapshots_not_aliases():
    """reset 返回的批里每帧是独立快照：后续步写新帧不改前面已返回的帧。"""
    inner = FakeTaskEnv()
    w = DemonstrationWrapper(as_made(inner), max_steps_without_demonstration=99, gui_render=False,
                             include_front_depth=True)
    obs0, _ = _reset_with_empty_demo(w)
    first = obs0["front_rgb_list"][-1].copy()
    w.step(np.zeros(8))
    w.step(np.zeros(8))
    np.testing.assert_array_equal(obs0["front_rgb_list"][-1], first)


@pytest.mark.slow
@pytest.mark.parametrize("space", ["joint_angle", "ee_pose", "waypoint", "multi_choice"])
def test_switches_all_256_through_builder(monkeypatch, tmp_path, space):
    spy = GymMakeSpy()
    monkeypatch.setattr(ecr, "gym", spy.namespace())
    forced = space == "multi_choice"
    for bits in itertools.product((False, True), repeat=len(FLAGS)):
        flags = dict(zip(FLAGS, bits))
        env = BenchmarkEnvBuilder("PickXtimes", action_space=space,
                                  override_metadata_path=tmp_path).make_env_for_episode(0, **flags)
        demo = find_wrapper(env, "DemonstrationWrapper")
        _reset_with_empty_demo(demo)
        obs, _r, _t, _tr, info = demo.step(np.zeros(8))
        eff = dict(flags)
        eff["include_front_camera_extrinsic"] |= forced
        eff["include_front_camera_intrinsic"] |= forced
        _check(obs, info, eff, step_idx=demo.unwrapped.step_calls)
