"""Demo and initial-action timing of DemonstrationWrapper.reset (C09.02), run once each for the official and hard copies.

Calls the real ``reset`` -> real ``get_demonstration_trajectory`` -> real ``planner_denseStep._collect_dense_steps``;
only the motion planners (mplib) are replaced by CPU spies (the two FailAware planner classes in ``planner_fail_safe``); the task's ``solve`` is a hand-written
double. Expectations are always hand-written event sequences:

- event order: env.reset -> build planner -> for each demo task "evaluate(True) -> solve's low-level steps -> evaluate(True)" -> initial action step;
  solve of non-demo tasks is not called;
- frame sequence: demo segments concatenated in task order, initial action step last; ``NO RECORD`` frames filtered out, the same sentence goal at different indices all kept;
- demo -> online: ``demonstration_record_traj`` is True during the demo and False after reset returns; demo steps do not count toward steps without demonstration;
- terminal extra low-level step: when a demo step terminates, one extra low-level step (same action) is taken but not added to demo frames;
- exception recovery hook: screw failure (exception or -1) falls back to RRT*, RRT* failures retry up to the cap (pinned in wrappers_pins) then return -1; solve raising ScrewPlanFailure or returning -1
  does not lose collected frames and later tasks run as usual; other exceptions propagate and the env.step interception is restored; non-callable solve -> ValueError.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from robomme.robomme_env.utils import planner_fail_safe as pfs
from robomme.robomme_env.utils.planner_fail_safe import ScrewPlanFailure

from tests.robomme_ood.unit.robomme import official_thresholds as T
from tests.robomme_ood.unit.wrappers import wrappers_pins as P
from tests.robomme_ood.unit.wrappers.wrappers_fakes import (
    SWING7,
    ScriptedEnv,
    as_made,
    frame_value,
    planner_spy_classes,
)

PKGS = ("robomme", "robomme_ood")


@pytest.fixture(params=PKGS)
def demo_cls(request):
    mod = importlib.import_module(f"{request.param}.env_record_wrapper.DemonstrationWrapper")
    return mod.DemonstrationWrapper


@pytest.fixture
def log():
    return []


def _install_planners(monkeypatch, log, scripts=None):
    arm, stick = planner_spy_classes(log, scripts)
    # the hard package's planner_fail_safe is a shim of the official module (the same module object); replacing it once takes effect for both packages
    hard_pfs = importlib.import_module("robomme_ood.robomme_env.utils.planner_fail_safe")
    assert hard_pfs is pfs
    monkeypatch.setattr(pfs, "FailAwarePandaArmMotionPlanningSolver", arm)
    monkeypatch.setattr(pfs, "FailAwarePandaStickMotionPlanningSolver", stick)


def _make(demo_cls, log, tasks, env_id="PickXtimes", outcomes=(), max_steps=100):
    inner = ScriptedEnv(env_id=env_id, log=log, outcomes=outcomes, task_list=tasks)
    w = demo_cls(as_made(inner), max_steps_without_demonstration=max_steps, gui_render=False)
    return w, inner


def _subgoal(env, text, demo=True):
    """Mimics sequential_task_check: directly change the current subgoal text and demo flag on the task env."""
    env.unwrapped.current_task_name = text
    env.unwrapped.current_task_demonstration = demo


def _rgb_values(obs):
    return [int(np.asarray(f).reshape(-1)[0]) for f in obs["front_rgb_list"]]


# --------------------------------------------------------------------------- Main timing


def test_reset_timeline_and_frames(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log, {"screw": [2], "close": [1]})
    seen_record_flag = []

    def solve_t0(env, planner):
        seen_record_flag.append(env.unwrapped.demonstration_record_traj)
        _subgoal(env, "pick A")
        planner.move_to_pose_with_screw("P0")          # 2 low-level steps
        _subgoal(env, "NO RECORD")
        planner.env.step(np.zeros(8))                   # 1 low-level step (should be filtered out)
        return 0

    def solve_t1(env, planner):  # non-demo task: reset must not call it
        log.append(("solve.t1",))

    def solve_t2(env, planner):
        seen_record_flag.append(env.unwrapped.demonstration_record_traj)
        _subgoal(env, "pick A")                         # same text as t0, different index
        planner.close_gripper()                         # 1 low-level step
        _subgoal(env, "put down", demo=False)           # demo ends, switch to the online subtask
        return None

    tasks = [{"name": "t0", "demonstration": True, "solve": solve_t0},
             {"name": "t1", "demonstration": False, "solve": solve_t1},
             {"name": "t2", "demonstration": True, "solve": solve_t2}]
    w, inner = _make(demo_cls, log, tasks)
    obs, info = w.reset()

    arm_kwargs = dict(debug=False, vis=False, base_pose=inner.agent.robot.pose,
                      visualize_target_grasp_pose=False, print_env_info=False)
    assert log == [
        ("env.reset",),
        ("planner.new", "arm", arm_kwargs),
        ("env.evaluate", True),
        ("planner.screw", "P0"), ("env.step", 1, "pick A"), ("env.step", 2, "pick A"),
        ("env.step", 3, "NO RECORD"),
        ("env.evaluate", True),
        ("env.evaluate", True),
        ("planner.close", None), ("env.step", 4, "pick A"),
        ("env.evaluate", True),
        ("env.step", 5, "put down"),                    # initial action step
    ]
    # frames: NO RECORD of step 3 filtered out; all three frames with the same text "pick A" kept
    assert _rgb_values(obs) == [frame_value(1), frame_value(2), frame_value(4), frame_value(5)]
    subgoals = w.demonstration_data[4]["simple_subgoal_online"]
    assert subgoals == ["pick A", "pick A", "pick A", "put down"]
    assert all(len(v) == 4 for v in obs.values())
    assert w.demonstration_data[1].shape == (4,)
    # initial action = home pose
    np.testing.assert_allclose(inner.actions[-1], T.HOME_ACTION)
    # demo -> online
    assert seen_record_flag == [True, True]
    assert inner.demonstration_record_traj is False
    assert w.steps_without_demonstration == 1          # only the initial action step counts
    assert (w.task_list_length, w.non_demonstration_task_length) == (3, 1)
    assert info["simple_subgoal_online"] == "put down" and info["status"] == "ongoing"


def test_planner_env_is_wrapper_and_step_restored(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log)
    holder = {}

    def solve(env, planner):
        holder["planner"] = planner
        holder["patched_step"] = planner.env.step
        planner.env.step(np.zeros(8))

    w, _ = _make(demo_cls, log, [{"name": "t", "demonstration": True, "solve": solve}])
    w.reset()
    assert holder["planner"].env is w                   # the planner drives the DemonstrationWrapper itself
    assert holder["patched_step"].__name__ == "_step"  # env.step is intercepted by the collector during solve
    assert w.step.__func__ is demo_cls.step            # restored to the original method afterwards


def test_stick_env_uses_stick_planner_and_swing_init(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log)
    w, inner = _make(demo_cls, log, [], env_id="RouteStick")
    w.reset()
    new = [e for e in log if e[0] == "planner.new"]
    assert new == [("planner.new", "stick", dict(debug=False, vis=False, base_pose=inner.agent.robot.pose,
                                                 visualize_target_grasp_pose=False, print_env_info=False,
                                                 joint_vel_limits=P.STICK_JOINT_VEL_LIMITS))]
    np.testing.assert_allclose(inner.actions[-1], SWING7)
    assert inner.actions[-1].shape == (7,)


# --------------------------------------------------------------------------- Terminal extra low-level step


def test_terminal_inside_demo_takes_unrecorded_extra_step(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log, {"screw": [2]})

    def solve(env, planner):
        _subgoal(env, "demo")
        planner.move_to_pose_with_screw("P")

    # the 2nd low-level step is judged success -> DemonstrationWrapper takes one extra step (step 3), which is not added to demo frames
    w, inner = _make(demo_cls, log, [{"name": "t", "demonstration": True, "solve": solve}],
                     outcomes=[(False, False), (True, False)])
    obs, info = w.reset()
    steps = [e[1] for e in log if e[0] == "env.step"]
    assert steps == [1, 2, 3, 4]                         # 2 planning steps + 1 extra step + 1 initial action step
    np.testing.assert_array_equal(inner.actions[2], inner.actions[1])  # the extra step repeats the same action
    assert _rgb_values(obs) == [frame_value(1), frame_value(2), frame_value(4)]
    assert w.demonstration_data[4]["status"] == ["ongoing", "success", "ongoing"]
    # lock in current behavior: a successful terminal during the demo sets episode_success True; the non-terminal initial action step does not change it
    assert w.episode_success is True


# --------------------------------------------------------------------------- screw -> RRT* recovery hook


def test_screw_failure_falls_back_to_rrt_with_same_pose(demo_cls, log, monkeypatch):
    # screw always fails; the first RRT* attempts alternately raise/return -1, the last succeeds with 2 steps
    rrt = [RuntimeError("r") if i % 2 == 0 else -1 for i in range(P.DEMO_RRT_ATTEMPTS - 1)] + [2]
    _install_planners(monkeypatch, log, {"screw": [ScrewPlanFailure("s")] * P.DEMO_SCREW_ATTEMPTS, "rrt": rrt})
    results = []

    def solve(env, planner):
        _subgoal(env, "demo")
        results.append(planner.move_to_pose_with_screw("POSE"))

    w, _ = _make(demo_cls, log, [{"name": "t", "demonstration": True, "solve": solve}])
    obs, _ = w.reset()
    calls = [e for e in log if e[0] in ("planner.screw", "planner.rrt")]
    # in the demo phase screw and RRT* are each tried up to the pinned cap; each receives the same goal
    assert calls == [("planner.screw", "POSE")] * P.DEMO_SCREW_ATTEMPTS + [("planner.rrt", "POSE")] * P.DEMO_RRT_ATTEMPTS
    assert results == [0] and w._current_demo_task_screw_failed is False
    assert len(obs["front_rgb_list"]) == 3              # 2 steps of the successful RRT* + initial action step


def test_all_planning_fails_keeps_frames_and_continues(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log, {"screw": [-1] * P.DEMO_SCREW_ATTEMPTS, "rrt": [-1] * P.DEMO_RRT_ATTEMPTS})
    order, flags = [], []

    def solve_a(env, planner):
        _subgoal(env, "a")
        planner.env.step(np.zeros(8))                   # 1 step taken before planning
        res = planner.move_to_pose_with_screw("P")
        order.append(("a", res))
        return res                                      # hand the -1 back unchanged

    def solve_b(env, planner):
        flags.append(env._current_demo_task_screw_failed)  # the failure flag is cleared when a new task starts
        _subgoal(env, "b")
        planner.env.step(np.zeros(8))
        order.append(("b", 0))

    tasks = [{"name": "a", "demonstration": True, "solve": solve_a},
             {"name": "b", "demonstration": True, "solve": solve_b}]
    w, _ = _make(demo_cls, log, tasks)
    obs, _ = w.reset()
    assert order == [("a", -1), ("b", 0)]
    assert flags == [False]
    assert [e[0] for e in log].count("planner.screw") == P.DEMO_SCREW_ATTEMPTS
    assert [e[0] for e in log].count("planner.rrt") == P.DEMO_RRT_ATTEMPTS
    # solve returning -1 does not lose a's collected frames: a 1 frame + b 1 frame + initial action step
    assert w.demonstration_data[4]["simple_subgoal_online"][:2] == ["a", "b"]
    assert len(obs["front_rgb_list"]) == 3


def test_solve_raising_screw_failure_keeps_frames(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log)
    evals_before = []

    def solve_a(env, planner):
        _subgoal(env, "a")
        planner.env.step(np.zeros(8))
        raise ScrewPlanFailure("boom")

    def solve_b(env, planner):
        evals_before.append([e for e in log if e[0] == "env.evaluate"].copy())
        _subgoal(env, "b")
        planner.env.step(np.zeros(8))

    tasks = [{"name": "a", "demonstration": True, "solve": solve_a},
             {"name": "b", "demonstration": True, "solve": solve_b}]
    w, _ = _make(demo_cls, log, tasks)
    obs, _ = w.reset()
    assert w.demonstration_data[4]["simple_subgoal_online"][:2] == ["a", "b"]
    # after a raises, its closing evaluate still runs: before b starts there are three (before a, after a, before b)
    assert len(evals_before[0]) == 3


def test_other_exception_propagates_and_restores_step(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log)

    def solve(env, planner):
        planner.env.step(np.zeros(8))
        raise RuntimeError("solver crashed")

    w, _ = _make(demo_cls, log, [{"name": "t", "demonstration": True, "solve": solve}])
    with pytest.raises(RuntimeError, match="solver crashed"):
        w.reset()
    assert w.step.__func__ is demo_cls.step


def test_non_callable_solve_rejected(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log)
    w, _ = _make(demo_cls, log, [{"name": "broken", "demonstration": True, "solve": None}])
    with pytest.raises(ValueError, match="broken"):
        w.reset()


# --------------------------------------------------------------------------- Two consecutive episodes (A -> B)


def test_second_reset_rebuilds_planner_and_drops_old_frames(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log)
    planners = []

    def solve(env, planner):
        planners.append(planner)
        _subgoal(env, f"ep{len(planners)}")
        planner.env.step(np.zeros(8))
        _subgoal(env, "online", demo=False)

    w, inner = _make(demo_cls, log, [{"name": "t", "demonstration": True, "solve": solve}])
    w.reset()
    w.step(np.zeros(8))
    obs, _ = w.reset()
    assert len(planners) == 2 and planners[0] is not planners[1]
    assert w.demonstration_data[4]["simple_subgoal_online"] == ["ep2", "online"]
    # frames of the second episode come from low-level steps 4 and 5 (after the first episode's 2 frames + 1 online step)
    assert _rgb_values(obs) == [frame_value(4), frame_value(5)]
    assert w.steps_without_demonstration == 1
