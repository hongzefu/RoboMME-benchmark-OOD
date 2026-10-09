"""step of the four action-space wrappers (C08.02): joint_angle passthrough, ee_pose, waypoint, multi_choice.

Each action space is assembled as the real chain (outer action wrapper -> real DemonstrationWrapper -> OrderEnforcing -> CPU double task env),
with planners/IK always replaced by CPU spies. Run once each for the official and hard packages: the hard package's DemonstrationWrapper and OraclePlanner are copies,
EndeffectorDemonstrationWrapper and MultiStepDemonstrationWrapper are shims of the official modules (see src/robomme_ood/UPSTREAM.json).

All expectations are hand-written:
- return contract: obs is dict-of-lists, each key's length = the low-level steps actually taken and collected for this action; each frame's front RGB pixel value encodes
  which low-level step it came from (order checkable); joints (7,), end effector (6,) float64, gripper (2,); reward a 0-d float32 tensor,
  terminated/truncated 0-d bool tensors; info a flat dict;
- input: NumPy, Python list and CPU Tensor inputs are all converted to float64 NumPy for the inner layer; the caller's action object is not modified in place;
- ee_pose: IK input = hand-computed target pose and the robot's current qpos, taking the first 7 dims of the first solution + gripper; the planner is built once, with different arm/stick constructor arguments
  (the IK failure path is already in tests/robomme_ood/unit/robomme/test_fail_paths.py, not repeated);
- waypoint: the pose given to screw = waypoint + hand-computed quaternion; movement frames first, gripper frames after; the gripper only accepts exact +/-1;
- multi_choice: reset builds the planner and camera cache; point=[y,x] selects the nearest candidate via real pinhole projection and hands it to solve;
  order and caps of the screw->RRT* retry hook; solve returning -1 -> RuntimeError.
"""
from __future__ import annotations

import copy
import importlib
import math

import numpy as np
import pytest
import torch

from robomme.robomme_env.utils import planner_fail_safe as pfs
from robomme.robomme_env.utils.planner_fail_safe import ScrewPlanFailure

from tests.robomme_ood.unit.wrappers import wrappers_pins as P
from tests.robomme_ood.unit.wrappers.wrappers_fakes import (
    E_ID,
    H,
    K,
    QPOS9,
    W,
    ScriptedEnv,
    as_made,
    frame_value,
    ik_spy_factory,
    planner_spy_classes,
)

PKGS = ("robomme", "robomme_ood")
# the package __init__ shadows the submodule attribute with a same-named class; fetch the module itself by module path (the hard package's same-named module is its shim)
ee_mod = importlib.import_module("robomme.env_record_wrapper.EndeffectorDemonstrationWrapper")
SQ = math.sqrt(0.5)
BASE_OBS = {"front_rgb_list", "wrist_rgb_list", "joint_state_list", "eef_state_list", "gripper_state_list"}


def _mods(pkg):
    return {
        name: importlib.import_module(f"{pkg}.env_record_wrapper.{name}")
        for name in ("DemonstrationWrapper", "EndeffectorDemonstrationWrapper", "MultiStepDemonstrationWrapper",
                     "OraclePlannerDemonstrationWrapper")
    }


def _vqa(pkg):
    return importlib.import_module(f"{pkg}.robomme_env.utils.vqa_options")


@pytest.fixture(params=PKGS)
def pkg(request):
    return request.param


@pytest.fixture
def log():
    return []


@pytest.fixture
def planners(monkeypatch, log):
    """Replace FailAware planners with spies (hard's planner_fail_safe is a shim, the same module object). Returns the script dict, which tests may fill before reset."""
    def install(new_scripts=None):
        a, s = planner_spy_classes(log, new_scripts or {})
        monkeypatch.setattr(pfs, "FailAwarePandaArmMotionPlanningSolver", a)
        monkeypatch.setattr(pfs, "FailAwarePandaStickMotionPlanningSolver", s)

    install()
    return install


def _demo(pkg, log, env_id="PickXtimes", **flags):
    inner = ScriptedEnv(env_id=env_id, log=log)
    demo = _mods(pkg)["DemonstrationWrapper"].DemonstrationWrapper(
        as_made(inner), max_steps_without_demonstration=100, gui_render=False, **flags)
    return demo, inner


def _rgb_values(obs):
    return [int(np.asarray(f).reshape(-1)[0]) for f in obs["front_rgb_list"]]


def _check_contract(out, n_frames, first_step):
    """Hand-written return contract: n_frames frames, coming from low-level steps first_step, first_step+1, ... in turn."""
    obs, reward, terminated, truncated, info = out
    assert BASE_OBS <= set(obs)
    assert all(isinstance(v, list) and len(v) == n_frames for v in obs.values())
    assert _rgb_values(obs) == [frame_value(first_step + i) for i in range(n_frames)]
    for i in range(n_frames):
        rgb = obs["front_rgb_list"][i]
        assert isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (H, W, 3)
        assert obs["joint_state_list"][i].shape == (7,)
        np.testing.assert_allclose(obs["joint_state_list"][i], QPOS9[:7], rtol=0, atol=1e-7)
        eef = obs["eef_state_list"][i]
        assert eef.dtype == np.float64 and eef.shape == (6,)
        np.testing.assert_allclose(eef, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0], atol=1e-6)
        assert obs["gripper_state_list"][i].shape == (2,)
    assert isinstance(reward, torch.Tensor) and reward.ndim == 0 and reward.dtype == torch.float32
    for flag in (terminated, truncated):
        assert isinstance(flag, torch.Tensor) and flag.ndim == 0 and flag.dtype == torch.bool
    assert isinstance(info, dict) and info["status"] == "ongoing"
    assert isinstance(info["simple_subgoal_online"], str)


def _as_kind(values, kind):
    if kind == "numpy":
        return np.asarray(values, dtype=np.float32)
    if kind == "list":
        return list(values)
    return torch.tensor(values, dtype=torch.float32)  # CPU Tensor


def _snapshot(x):
    return x.clone() if isinstance(x, torch.Tensor) else copy.deepcopy(x)


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    return np.array_equal(np.asarray(a), np.asarray(b))


# --------------------------------------------------------------------------- joint_angle (passthrough)

KINDS = ("numpy", "list", "tensor")


@pytest.mark.parametrize("kind", KINDS)
def test_joint_angle_step_contract(pkg, log, planners, kind):
    demo, inner = _demo(pkg, log)
    demo.reset()                                         # initial action step = low-level step 1
    values = [0.25, -0.5, 0.75, -1.0, 0.125, 0.5, -0.25, 1.0, 9.0]  # the 9th dim should be truncated
    action = _as_kind(values, kind)
    before = _snapshot(action)
    out = demo.step(action)
    _check_contract(out, n_frames=1, first_step=2)
    sent = inner.actions[-1]
    assert sent.dtype == np.float64 and sent.shape == (8,)
    np.testing.assert_array_equal(sent, values[:8])
    assert _same(action, before)                          # caller's action not modified
    if kind == "numpy":
        sent[0] = 123.0
        assert action[0] == np.float32(0.25)              # the inner layer gets a copy, not an alias


# --------------------------------------------------------------------------- ee_pose


def _ee(pkg, log, monkeypatch, env_id="PickXtimes", repr_="rpy", solutions=None):
    arm, stick = ik_spy_factory(log, solutions if solutions is not None else [np.arange(9) / 10.0])
    monkeypatch.setattr(ee_mod, "PandaArmMotionPlanningSolver", arm)
    monkeypatch.setattr(ee_mod, "PandaStickMotionPlanningSolver", stick)
    demo, inner = _demo(pkg, log, env_id=env_id)
    cls = _mods(pkg)["EndeffectorDemonstrationWrapper"].EndeffectorDemonstrationWrapper
    w = cls(demo, action_repr=repr_)
    w.reset()
    return w, demo, inner


@pytest.mark.parametrize("kind", KINDS)
def test_ee_pose_success_contract(pkg, log, planners, monkeypatch, kind):
    first = np.array([0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.04, 0.04])
    second = np.full(9, 7.0)
    w, demo, inner = _ee(pkg, log, monkeypatch, solutions=[first, second])
    values = [0.1, 0.2, 0.3, 0.0, 0.0, math.pi / 2, -1.0, 99.0]  # the 8th dim is extra and should be ignored
    action = _as_kind(values, kind)
    before = _snapshot(action)
    out = w.step(action)
    _check_contract(out, n_frames=1, first_step=2)
    # IK input: world-frame goal = position + wxyz quaternion for a 90 deg rotation about z (hand-computed cos45, 0, 0, sin45); current qpos takes row 0 of the robot
    to_base = [e for e in log if e[0] == "ik.to_base"]
    solve = [e for e in log if e[0] == "ik.solve"]
    np.testing.assert_allclose(to_base[-1][1], [0.1, 0.2, 0.3, SQ, 0, 0, SQ], atol=1e-6)
    np.testing.assert_allclose(solve[-1][1], to_base[-1][1])
    np.testing.assert_allclose(solve[-1][2], QPOS9, atol=1e-7)
    # take the first 7 dims of the first solution + gripper
    sent = inner.actions[-1]
    assert sent.dtype == np.float64 and sent.shape == (8,)
    np.testing.assert_allclose(sent, list(first[:7]) + [-1.0])
    assert _same(action, before)


def test_ee_pose_planner_built_once_with_arm_kwargs(pkg, log, planners, monkeypatch):
    w, demo, inner = _ee(pkg, log, monkeypatch)
    w.step([0.1, 0.2, 0.3, 0, 0, 0, 1.0])
    w.step([0.1, 0.2, 0.3, 0, 0, 0, 1.0])
    new = [e for e in log if e[0] == "ik.new"]
    assert new == [("ik.new", "arm", dict(debug=False, vis=False, base_pose=inner.agent.robot.pose,
                                          visualize_target_grasp_pose=False, print_env_info=False))]
    assert w._ee_pose_planner.env is demo               # the IK planner is bound to the inner DemonstrationWrapper


def test_ee_pose_stick_quat_drops_gripper(pkg, log, planners, monkeypatch):
    w, demo, inner = _ee(pkg, log, monkeypatch, env_id="RouteStick", repr_="quat")
    out = w.step([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])    # stick + quat: 7 dims suffice
    _check_contract(out, n_frames=1, first_step=2)
    new = [e for e in log if e[0] == "ik.new"]
    assert new[0][1] == "stick" and new[0][2]["joint_vel_limits"] == P.STICK_JOINT_VEL_LIMITS
    np.testing.assert_allclose([e for e in log if e[0] == "ik.to_base"][-1][1], [0.1, 0.2, 0.3, 1, 0, 0, 0])
    assert inner.actions[-1].shape == (7,)
    np.testing.assert_allclose(inner.actions[-1], np.arange(7) / 10.0)


def test_ee_pose_reset_and_close_pass_through(pkg, log, planners, monkeypatch):
    w, demo, inner = _ee(pkg, log, monkeypatch)
    resets_before = log.count(("env.reset",))
    w.reset()
    assert log.count(("env.reset",)) == resets_before + 1
    w.close()
    assert inner.closed


# --------------------------------------------------------------------------- waypoint


def _waypoint(pkg, log, env_id="PickXtimes", vis=False):
    demo, inner = _demo(pkg, log, env_id=env_id)
    cls = _mods(pkg)["MultiStepDemonstrationWrapper"].MultiStepDemonstrationWrapper
    w = cls(demo, gui_render=False, vis=vis)
    w.reset()
    return w, demo, inner


@pytest.mark.parametrize("kind", KINDS)
def test_waypoint_success_contract_and_order(pkg, log, planners, kind):
    planners({"screw": [2], "close": [1]})
    w, demo, inner = _waypoint(pkg, log)
    values = [0.3, 0.1, 0.2, 0.0, 0.0, math.pi / 2, -1.0]
    action = _as_kind(values, kind)
    before = _snapshot(action)
    out = w.step(action)
    # 2 movement frames (low-level steps 2, 3) first, 1 gripper frame (step 4) after
    _check_contract(out, n_frames=3, first_step=2)
    calls = [e for e in log if e[0].startswith("planner.") and e[0] != "planner.new"]
    assert [c[0] for c in calls] == ["planner.screw", "planner.close"]
    pose = calls[0][1]
    np.testing.assert_allclose(pose.p, [0.3, 0.1, 0.2], atol=1e-6)
    np.testing.assert_allclose(pose.q, [SQ, 0, 0, SQ], atol=1e-6)
    assert _same(action, before)


@pytest.mark.parametrize("rpy, quat", [((math.pi, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
                                       ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))])
def test_waypoint_pose_quaternion_handcomputed(pkg, log, planners, rpy, quat):
    w, demo, inner = _waypoint(pkg, log)
    w.step([0.0, 0.0, 0.5, *rpy, 0.0])
    pose = [e for e in log if e[0] == "planner.screw"][-1][1]
    q = np.asarray(pose.q, dtype=np.float64)
    # quaternions q and -q are the same rotation: compare after sign alignment
    q = q if np.dot(q, quat) >= 0 else -q
    np.testing.assert_allclose(q, quat, atol=1e-6)


@pytest.mark.parametrize("env_id, kind, extra", [
    ("PickXtimes", "arm", dict(vis=True, visualize_target_grasp_pose=True)),
    ("PatternLock", "stick", dict(vis=True, visualize_target_grasp_pose=False, joint_vel_limits=P.STICK_JOINT_VEL_LIMITS)),
])
def test_waypoint_planner_built_once(pkg, log, planners, env_id, kind, extra):
    w, demo, inner = _waypoint(pkg, log, env_id=env_id, vis=True)
    w.step([0.0, 0.0, 0.5, 0, 0, 0, 0])
    w.step([0.0, 0.0, 0.5, 0, 0, 0, 0])
    new = [e for e in log if e[0] == "planner.new"]
    expected = dict(debug=False, base_pose=inner.agent.robot.pose, print_env_info=False, **extra)
    # the first is the planner used by the inner DemonstrationWrapper.reset demo; waypoint's own planner is built only once over two steps
    assert len(new) == 2 and new[1] == ("planner.new", kind, expected)
    assert w._planner.env is demo


@pytest.mark.parametrize("grip", [-0.5, 0.999, -1.0001])
def test_waypoint_gripper_requires_exact_unit(pkg, log, planners, grip):
    w, demo, inner = _waypoint(pkg, log)
    out = w.step([0.0, 0.0, 0.5, 0, 0, 0, grip])
    names = [e[0] for e in log]
    assert "planner.close" not in names and "planner.open" not in names
    assert len(out[0]["front_rgb_list"]) == 1            # only screw's 1 low-level step


def test_waypoint_open_gripper_after_move(pkg, log, planners):
    planners({"screw": [1], "open": [2]})
    w, demo, inner = _waypoint(pkg, log)
    out = w.step([0.0, 0.0, 0.5, 0, 0, 0, 1.0])
    _check_contract(out, n_frames=3, first_step=2)
    assert [e[0] for e in log if e[0] in ("planner.screw", "planner.open", "planner.close")] == \
        ["planner.screw", "planner.open"]


def test_waypoint_failed_attempt_frames_are_dropped(pkg, log, planners):
    """Lock in current behavior: screw takes 1 step and then raises ScrewPlanFailure; the env really advanced, but this step does not appear in the returned frames."""
    planners({"screw": [("steps_then_raise", 1, ScrewPlanFailure("mid")), 2]})
    w, demo, inner = _waypoint(pkg, log)
    out = w.step([0.0, 0.0, 0.5, 0, 0, 0, 0])
    assert inner.step_calls == 1 + 3                       # initial action step + 1 failed step + 2 successful steps
    assert _rgb_values(out[0]) == [frame_value(3), frame_value(4)]


def test_waypoint_last_step_signals(pkg, log, planners):
    planners({"screw": [2]})
    w, demo, inner = _waypoint(pkg, log)
    inner.outcomes = [(False, False), (True, False), (True, False)]  # the 2nd movement step succeeds -> extra low-level step
    obs, reward, terminated, truncated, info = w.step([0.0, 0.0, 0.5, 0, 0, 0, 0])
    assert bool(terminated) is True and info["status"] == "success"
    assert len(obs["front_rgb_list"]) == 2                 # the extra step is not in the returned frames
    assert inner.step_calls == 1 + 3


# --------------------------------------------------------------------------- multi_choice


def _oracle(pkg, log, monkeypatch, solve_steps=2, solve_result=None):
    seen = []
    vqa = _vqa(pkg)

    def fake_pickup(env, planner, obj=None, **kw):
        seen.append(obj)
        for _ in range(solve_steps):
            planner.env.step(np.zeros(8))
        return solve_result

    monkeypatch.setattr(vqa, "solve_pickup", fake_pickup)
    demo, inner = _demo(pkg, log, include_front_camera_extrinsic=True, include_front_camera_intrinsic=True)
    cls = _mods(pkg)["OraclePlannerDemonstrationWrapper"].OraclePlannerDemonstrationWrapper
    w = cls(demo, env_id="PickXtimes", gui_render=False)
    obs, info = w.reset()
    return w, demo, inner, seen, (obs, info)


def test_multi_choice_reset_builds_planner_and_camera_cache(pkg, log, planners, monkeypatch):
    w, demo, inner, _, (obs, info) = _oracle(pkg, log, monkeypatch)
    new = [e for e in log if e[0] == "planner.new"]
    # OraclePlanner.reset first builds its own planner (vis = gui_render) then resets the inner layer; the second is the planner used by the inner demo
    assert len(new) == 2
    oracle_new = new[0]
    assert oracle_new == ("planner.new", "arm", dict(debug=False, vis=False, base_pose=inner.agent.robot.pose,
                                                     visualize_target_grasp_pose=False, print_env_info=False))
    assert w.planner.env is demo
    assert [o["label"] for o in info["available_multi_choices"]] == ["a", "b", "c"]
    np.testing.assert_array_equal(w._front_camera_intrinsic_cv, np.asarray(K))
    np.testing.assert_array_equal(w._front_camera_extrinsic_cv, np.asarray(E_ID))
    assert w._front_camera_intrinsic_cv.dtype == np.float64
    assert w._front_rgb_shape == (H, W)


def test_multi_choice_point_selects_nearest_by_projection(pkg, log, planners, monkeypatch):
    w, demo, inner, seen, _ = _oracle(pkg, log, monkeypatch)
    # cube_near (0.1, 0.05, 1.0) projects to: x = 100*0.1 + 32 = 42, y = 100*0.05 + 24 = 29; cube_far -> (22, 14)
    command = {"choice": "A", "point": [30, 41]}            # [y, x], nearest to cube_near
    before = copy.deepcopy(command)
    evals_before = log.count(("env.evaluate", False)), log.count(("env.evaluate", True))
    out = w.step(command)
    assert seen == [inner.all_cubes[0]]
    _check_contract(out, n_frames=2, first_step=2)
    assert command == before                                 # command dict not modified
    assert (log.count(("env.evaluate", False)) - evals_before[0], log.count(("env.evaluate", True)) - evals_before[1]) == (1, 1)
    assert [o["label"] for o in out[4]["available_multi_choices"]] == ["a", "b", "c"]


def test_multi_choice_point_near_far_cube(pkg, log, planners, monkeypatch):
    w, demo, inner, seen, _ = _oracle(pkg, log, monkeypatch)
    w.step({"choice": "a", "point": [13, 21]})               # near the cube_far projection (22, 14)
    assert seen == [inner.all_cubes[1]]


def test_multi_choice_solve_minus_one_raises(pkg, log, planners, monkeypatch):
    w, demo, inner, seen, _ = _oracle(pkg, log, monkeypatch, solve_steps=1, solve_result=-1)
    with pytest.raises(RuntimeError, match="Oracle solve failed"):
        w.step({"choice": "a", "point": [29, 42]})


def _retry_harness(pkg, log, scripts):
    """Really construct OraclePlanner (caps taken from its __init__) and attach the retry hook to the planner spy."""
    a, _ = planner_spy_classes(log, scripts)
    planner = a(env=None)
    cls = _mods(pkg)["OraclePlannerDemonstrationWrapper"].OraclePlannerDemonstrationWrapper
    w = cls(as_made(ScriptedEnv(log=[])), env_id="PickXtimes", gui_render=False)
    w._wrap_planner_with_screw_then_rrt_retry(planner, screw_failure_exc=ScrewPlanFailure)
    return planner


def _alternating_failures(n):
    return [ScrewPlanFailure(str(i)) if i % 2 == 0 else -1 for i in range(n)]


def test_multi_choice_retry_hook_order(pkg, log):
    rrt = [ValueError("r") if i % 2 == 0 else -1 for i in range(P.ORACLE_RRT_ATTEMPTS - 1)] + [0]
    planner = _retry_harness(pkg, log, {"screw": _alternating_failures(P.ORACLE_SCREW_ATTEMPTS), "rrt": rrt})
    assert planner.move_to_pose_with_screw("G") == 0
    calls = [e for e in log if e[0] in ("planner.screw", "planner.rrt")]
    assert calls == [("planner.screw", "G")] * P.ORACLE_SCREW_ATTEMPTS + [("planner.rrt", "G")] * P.ORACLE_RRT_ATTEMPTS


def test_multi_choice_retry_hook_screw_success_skips_rrt(pkg, log):
    planner = _retry_harness(pkg, log, {"screw": [-1, 0]})
    assert planner.move_to_pose_with_screw("G") == 0
    assert [e[0] for e in log if e[0] in ("planner.screw", "planner.rrt")] == ["planner.screw", "planner.screw"]


def test_multi_choice_retry_hook_exhausted(pkg, log):
    planner = _retry_harness(pkg, log, {"screw": [-1] * P.ORACLE_SCREW_ATTEMPTS, "rrt": [-1] * P.ORACLE_RRT_ATTEMPTS})
    with pytest.raises(RuntimeError, match="exhausted"):
        planner.move_to_pose_with_screw("G")


def test_multi_choice_retry_hook_does_not_swallow_other_screw_errors(pkg, log):
    """Lock in current behavior: when screw raises an exception other than ScrewPlanFailure it propagates directly, without falling back to RRT*."""
    planner = _retry_harness(pkg, log, {"screw": [KeyError("k")]})
    with pytest.raises(KeyError):
        planner.move_to_pose_with_screw("G")
    assert [e[0] for e in log if e[0] in ("planner.screw", "planner.rrt")] == ["planner.screw"]


@pytest.mark.parametrize("value, ok", [
    (np.eye(3), True), (np.eye(3).reshape(1, 3, 3), True), (np.arange(8), False),
    (np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]]), False), (None, False),
])
def test_multi_choice_intrinsic_normalization(pkg, value, ok):
    cls = _mods(pkg)["OraclePlannerDemonstrationWrapper"].OraclePlannerDemonstrationWrapper
    out = cls._normalize_intrinsic_cv(value)
    if ok:
        assert out.shape == (3, 3) and out.dtype == np.float64
        np.testing.assert_array_equal(out, np.eye(3))
    else:
        assert out is None


def test_multi_choice_bad_camera_does_not_overwrite_cache(pkg, log, planners, monkeypatch):
    w, demo, inner, seen, _ = _oracle(pkg, log, monkeypatch)
    w._update_front_camera_cache(obs_like={"front_camera_extrinsic_list": [np.full((3, 4), np.inf)]},
                                 info_like={"front_camera_intrinsic": [np.arange(4)]})
    np.testing.assert_array_equal(w._front_camera_intrinsic_cv, np.asarray(K))
    np.testing.assert_array_equal(w._front_camera_extrinsic_cv, np.asarray(E_ID))
