"""Two places in C08/C09 adjacent to recording read-back: the 8 ``include_*`` switches of ``DemonstrationWrapper._augment_obs_and_info``,
and ``_filter_no_record_from_step_batch`` before reset returns.

Parametrized over the two DemonstrationWrappers of the official package and the hard package (byte-for-byte copy); CPU stand-in env, no simulation.
The daily gate runs 18 switch combinations (all off, all on, 8 single-on, 8 single-off); all 256 combinations are marked slow.
"""
from __future__ import annotations

import importlib
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recording_fakes import (
    EXTRINSIC,
    FINGER_OPEN,
    IMG,
    INTRINSIC,
    WRIST_EXTRINSIC,
    WRIST_INTRINSIC,
    Event,
    FakeTaskEnv,
    action_of,
    front_depth,
    front_rgb,
    tcp_xyz,
    wrist_depth,
    wrist_rgb,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*to get variables from other wrappers")

SWITCHES = (
    "include_maniskill_obs",
    "include_front_depth",
    "include_wrist_depth",
    "include_front_camera_extrinsic",
    "include_wrist_camera_extrinsic",
    "include_available_multi_choices",
    "include_front_camera_intrinsic",
    "include_wrist_camera_intrinsic",
)
# keys added when each switch is on: (lands in obs or info, key name)
ADDS = {
    "include_maniskill_obs": ("obs", "maniskill_obs"),
    "include_front_depth": ("obs", "front_depth_list"),
    "include_wrist_depth": ("obs", "wrist_depth_list"),
    "include_front_camera_extrinsic": ("obs", "front_camera_extrinsic_list"),
    "include_wrist_camera_extrinsic": ("obs", "wrist_camera_extrinsic_list"),
    "include_available_multi_choices": ("info", "available_multi_choices"),
    "include_front_camera_intrinsic": ("info", "front_camera_intrinsic"),
    "include_wrist_camera_intrinsic": ("info", "wrist_camera_intrinsic"),
}
BASE_OBS = {"front_rgb_list", "wrist_rgb_list", "joint_state_list", "eef_state_list", "gripper_state_list"}
BASE_INFO_ADDED = {"simple_subgoal_online", "grounded_subgoal_online", "task_goal"}


def _combos_daily():
    n = len(SWITCHES)
    out = [(False,) * n, (True,) * n]
    for i in range(n):
        out.append(tuple(j == i for j in range(n)))
        out.append(tuple(j != i for j in range(n)))
    return out


def _combos_all():
    return list(itertools.product((False, True), repeat=len(SWITCHES)))


@pytest.fixture(params=["robomme", "robomme_ood"])
def demo_mod(request):
    return importlib.import_module(f"{request.param}.env_record_wrapper.DemonstrationWrapper")


def _make(demo_mod, flags, env_id="FakeTask"):
    env = FakeTaskEnv([Event(name="pick")], env_id=env_id)
    env.spec = SimpleNamespace(id=env_id)
    kw = dict(zip(SWITCHES, flags))
    w = demo_mod.DemonstrationWrapper(env, max_steps_without_demonstration=5, gui_render=False, **kw)
    env.reset()
    obs, _, _, _, info = env.step(torch.from_numpy(action_of(3)))
    return w, env, obs, info


def _check(demo_mod, flags, monkeypatch):
    monkeypatch.setattr(demo_mod, "get_vqa_options", lambda env, planner, target, env_id: [
        {"label": "a", "action": "pick", "available": [1]},
        {"label": "b", "action": "press", "available": []},
    ])
    w, env, obs, info = _make(demo_mod, flags)
    obs_keys_before = {k: set(v) for k, v in obs.items()}
    info_before = dict(info)
    new_obs, new_info = w._augment_obs_and_info(obs, info, torch.from_numpy(action_of(3)))
    on = {s for s, f in zip(SWITCHES, flags) if f}
    assert set(new_obs) == BASE_OBS | {ADDS[s][1] for s in on if ADDS[s][0] == "obs"}
    assert set(new_info) == set(info) | BASE_INFO_ADDED | {ADDS[s][1] for s in on if ADDS[s][0] == "info"}
    # input is not modified in place
    assert {k: set(v) for k, v in obs.items()} == obs_keys_before and info == info_before
    a = action_of(3)
    expect = {
        "front_rgb_list": (front_rgb(1), np.uint8),
        "wrist_rgb_list": (wrist_rgb(1), np.uint8),
        "joint_state_list": (a[:7].astype(np.float32), np.float32),
        "gripper_state_list": (np.array([FINGER_OPEN, FINGER_OPEN], dtype=np.float32), np.float32),  # action_of(3) gripper +1 → stand-in open level
        "front_depth_list": (front_depth(1), np.int16),
        "wrist_depth_list": (wrist_depth(1), np.int16),
        "front_camera_extrinsic_list": (EXTRINSIC, np.float32),
        "wrist_camera_extrinsic_list": (WRIST_EXTRINSIC, np.float32),
    }
    for k, v in new_obs.items():
        if k == "maniskill_obs":
            assert v is obs
            continue
        assert isinstance(v, np.ndarray), k  # all CPU Tensors converted to NumPy
        if k == "eef_state_list":
            assert v.dtype == np.float64 and v.shape == (6,)
            np.testing.assert_allclose(v, tcp_xyz(1) + [0, 0, 0], atol=1e-6)
            continue
        want, dt = expect[k]
        assert v.dtype == dt and v.shape == want.shape, k
        np.testing.assert_allclose(v, want, rtol=0, atol=1e-6)
    assert new_obs["front_rgb_list"].shape == (IMG, IMG, 3)
    if "include_front_camera_intrinsic" in on:
        np.testing.assert_array_equal(new_info["front_camera_intrinsic"], INTRINSIC)
        assert new_info["front_camera_intrinsic"].shape == (3, 3)
    if "include_wrist_camera_intrinsic" in on:
        np.testing.assert_array_equal(new_info["wrist_camera_intrinsic"], WRIST_INTRINSIC)
    if "include_available_multi_choices" in on:
        assert new_info["available_multi_choices"] == [
            {"label": "a", "action": "pick", "need_parameter": True},
            {"label": "b", "action": "press", "need_parameter": False},
        ]
    assert new_info["simple_subgoal_online"] == "pick"
    assert new_info["task_goal"] == []  # FakeTask has no language goal


@pytest.mark.parametrize("flags", _combos_daily(), ids=lambda f: "".join("1" if x else "0" for x in f))
def test_include_switches_daily(demo_mod, flags, monkeypatch):
    _check(demo_mod, flags, monkeypatch)


@pytest.mark.slow
@pytest.mark.parametrize("flags", _combos_all(), ids=lambda f: "".join("1" if x else "0" for x in f))
def test_include_switches_all(demo_mod, flags, monkeypatch):
    _check(demo_mod, flags, monkeypatch)


@pytest.mark.parametrize("env_id", ["PatternLock", "RouteStick"])
def test_stick_env_gripper_zero(demo_mod, env_id):
    """stick tasks are judged by task id: gripper state is always 0, joint dims 8 and 9 are not read."""
    w, env, obs, info = _make(demo_mod, (False,) * len(SWITCHES), env_id=env_id)
    new_obs, _ = w._augment_obs_and_info(obs, info, None)
    assert new_obs["gripper_state_list"].tolist() == [0.0, 0.0]
    assert new_obs["gripper_state_list"].dtype == np.float64


def test_switch_fixture_self_check():
    """Fixture self-check (not a negative case of the production checker): the daily 18 combinations are distinct and cover single-on and single-off for each switch; all combinations are exactly 2^8."""
    combos = _combos_daily()
    assert len(set(combos)) == len(combos) == 2 + 2 * len(SWITCHES)
    assert len(set(_combos_all())) == 2 ** len(SWITCHES)


# ---------------------------------------------------------------- NO RECORD filtering


def _batch(names, n=None):
    n = len(names) if n is None else n
    obs = {"front_rgb_list": [np.full((2, 2, 3), i, np.uint8) for i in range(n)], "scalar": 7}
    info = {"simple_subgoal_online": list(names), "status": ["ongoing"] * n}
    return (obs, torch.arange(n, dtype=torch.float32), torch.zeros(n, dtype=torch.bool),
            torch.zeros(n, dtype=torch.bool), info)


@pytest.fixture
def demo_wrapper(demo_mod):
    w, *_ = _make(demo_mod, (False,) * len(SWITCHES))
    return w


def test_no_record_filter_keeps_order(demo_wrapper):
    out = demo_wrapper._filter_no_record_from_step_batch(_batch(["NO RECORD", "a", " NO RECORD ", "b"]))
    obs, rew, ter, trn, info = out
    assert [int(x[0, 0, 0]) for x in obs["front_rgb_list"]] == [1, 3]
    assert rew.tolist() == [1.0, 3.0] and ter.numel() == 2 and trn.numel() == 2
    assert info["simple_subgoal_online"] == ["a", "b"] and info["status"] == ["ongoing", "ongoing"]
    assert obs["scalar"] == 7  # fields of unequal length kept as is


@pytest.mark.parametrize(
    "batch",
    [
        _batch(["a", "b"]),  # no NO RECORD
        _batch(["NO RECORD", "NO RECORD"]),  # filtering everything would leave it empty → defensively returned as is
        _batch([]),  # empty batch
    ],
)
def test_no_record_filter_returns_input_unchanged(demo_wrapper, batch):
    assert demo_wrapper._filter_no_record_from_step_batch(batch) is batch


def test_no_record_filter_rejects_malformed(demo_wrapper):
    obs, rew, ter, trn, info = _batch(["NO RECORD", "a"])
    bad_len = (obs, rew, ter, trn, {"simple_subgoal_online": ["NO RECORD"]})
    assert demo_wrapper._filter_no_record_from_step_batch(bad_len) is bad_len
    not_tensor = (obs, [0.0, 1.0], ter, trn, info)
    assert demo_wrapper._filter_no_record_from_step_batch(not_tensor) is not_tensor
    assert demo_wrapper._filter_no_record_from_step_batch("x") == "x"
