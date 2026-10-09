"""CPU doubles shared by the action-space wrapper and demo timing tests (C08.02/C09.02), for tests/robomme_ood/unit/wrappers/ only.

- ``ScriptedEnv``: poses as the task env (gymnasium.Env) without building any SAPIEN scene. Observation pixel values encode "which low-level step this is"
  (reset is step 0; the front RGB after step k is all ``frame_value(k)``), so any frame can be traced back to its low-level step;
  camera parameters are fixed to hand-written pinhole intrinsics ``K`` and identity extrinsics for easy hand-computed projection. All step/evaluate/reset calls are recorded in order into
  the shared event log ``log``, and tests compare it item by item against hand-written expected event sequences.
- ``planner_spy_classes``: CPU spies standing in for the real motion planners (mplib etc.). Constructor arguments, every planning call and its arguments go into the same
  event log; a script decides for each call "take n steps / return -1 / raise".
- ``IKPlannerSpy``: stands in for EndeffectorDemonstrationWrapper's IK planner, records IK inputs and gives solutions per script.
"""
from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import EnvSpec
from mani_skill.utils.structs.pose import Pose

# small image: height 48, width 64; intrinsics principal point at the image center, focal length 100 pixels (for hand-computed projection)
H, W = 48, 64
FOCAL = 100.0
K = ((FOCAL, 0.0, W / 2), (0.0, FOCAL, H / 2), (0.0, 0.0, 1.0))
E_ID = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
STICK_IDS = ("PatternLock", "RouteStick")
# double robot joint readings (9 dims; 7 dims for stick tasks), values hand-written
QPOS9 = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.04, 0.04)
SWING7 = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)


def frame_value(k: int) -> int:
    """Pixel value of the front RGB after the k-th low-level step (0 for reset)."""
    return (10 * k) % 256


def _obs(k: int):
    rgb = torch.full((1, H, W, 3), frame_value(k), dtype=torch.uint8)
    depth = torch.full((1, H, W, 1), 100 + k, dtype=torch.int16)
    seg = torch.zeros((1, H, W, 1), dtype=torch.int16)
    ext = torch.tensor([E_ID], dtype=torch.float32)
    intr = torch.tensor([K], dtype=torch.float32)

    def cam():
        return {"rgb": rgb.clone(), "depth": depth.clone(), "segmentation": seg.clone()}

    return {
        "sensor_data": {"base_camera": cam(), "hand_camera": cam()},
        "sensor_param": {
            "base_camera": {"extrinsic_cv": ext.clone(), "intrinsic_cv": intr.clone()},
            "hand_camera": {"extrinsic_cv": ext.clone(), "intrinsic_cv": intr.clone()},
        },
    }


class Actor:
    """Hashable actor double: only a name and world coordinates (real actors hash by object identity)."""

    def __init__(self, name, xyz):
        self.name = name
        self.pose = SimpleNamespace(p=torch.tensor([xyz], dtype=torch.float32))

    def __repr__(self):
        return f"<Actor {self.name}>"


class ScriptedEnv(gym.Env):
    """Scripted task env. outcomes: (success, fail) for each step in turn; (False, False) forever once exhausted."""

    metadata = {"render_modes": []}

    def __init__(self, env_id="PickXtimes", log=None, outcomes=(), task_list=None):
        self.spec = EnvSpec(id=env_id, entry_point="tests:wrappers_fake")
        self.log = log if log is not None else []
        self.outcomes = list(outcomes)
        self.actions = []
        self.step_calls = 0
        self.closed = False
        stick = env_id in STICK_IDS
        n = 7 if stick else 9
        qpos = torch.tensor([QPOS9[:n]], dtype=torch.float32)
        self.agent = SimpleNamespace(
            robot=SimpleNamespace(
                qpos=qpos,
                pose=Pose.create_from_pq(torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]])),
                get_qpos=lambda: qpos.clone(),
            ),
            tcp=SimpleNamespace(pose=Pose.create_from_pq(torch.tensor([[0.1, 0.2, 0.3]]),
                                                         torch.tensor([[1.0, 0, 0, 0]]))),
        )
        # subgoal state (written by sequential_task_check in real tasks; changed directly here by the test's solve doubles)
        self.current_task_demonstration = False
        self.current_task_name = "online"
        self.current_subgoal_segment = None
        self.current_segment = None
        self.segmentation_id_map = {}
        self.task_list = list(task_list or [])
        # fields read by task_goal (PickXtimes) and vqa_options (PickXtimes)
        self.num_repeats = 2
        self.target_color_name = "red"
        self.all_cubes = [Actor("cube_near", (0.1, 0.05, 1.0)), Actor("cube_far", (-0.1, -0.1, 1.0))]
        self.target = Actor("target", (0.0, 0.0, 1.0))
        self.button = Actor("button", (0.0, 0.1, 1.0))
        self.swing_qpos = torch.tensor([SWING7], dtype=torch.float32)

    def reset(self, *, seed=None, options=None):
        self.log.append(("env.reset",))
        return _obs(self.step_calls), {}

    def step(self, action):
        arr = np.asarray(action)
        self.actions.append(arr.copy())
        self.step_calls += 1
        self.log.append(("env.step", self.step_calls, self.current_task_name))
        success, fail = self.outcomes.pop(0) if self.outcomes else (False, False)
        info = {"success": torch.tensor([bool(success)]), "fail": torch.tensor([bool(fail)])}
        terminated = torch.tensor([bool(success) or bool(fail)])
        return _obs(self.step_calls), torch.tensor([0.0]), terminated, torch.tensor([False]), info

    def evaluate(self, solve_complete_eval=False):
        self.log.append(("env.evaluate", bool(solve_complete_eval)))
        return {"success": torch.tensor([False]), "fail": torch.tensor([False])}

    def close(self):
        self.closed = True


def as_made(inner):
    """Wrap the task in an OrderEnforcing layer like gym.make does (task_goal takes the task via ``.env.unwrapped``)."""
    return gym.wrappers.OrderEnforcing(inner)


# --------------------------------------------------------------------------- Planner spies


class _Script:
    """Script for one kind of planning call. Each item: int n (take n low-level steps then return 0), -1 (return -1 without stepping),
    a BaseException instance (raise directly), ("steps_then_raise", n, exc) (take n steps then raise). Once exhausted always "take 1 step"."""

    def __init__(self, items):
        self.items = list(items)

    def next(self):
        return self.items.pop(0) if self.items else 1


def planner_spy_classes(log, scripts=None):
    """Return the two classes (ArmSpy, StickSpy) standing in for FailAwarePandaArm/StickMotionPlanningSolver.

    Construction records ("planner.new", kind, kwargs); planning calls record ("planner.<method>", args).
    Each low-level step sends one action to ``self.env.step``: 8 dims for arm, 7 for stick, with value equal to this spy's cumulative call index (easy to identify)."""
    scripts = {k: _Script(v) for k, v in (scripts or {}).items()}

    class _Base:
        kind = "?"
        dim = 8

        def __init__(self, env, **kwargs):
            self.env = env
            self.kwargs = kwargs
            self.n_calls = 0
            log.append(("planner.new", self.kind, dict(kwargs)))

        def _run(self, name, arg):
            self.n_calls += 1
            log.append((f"planner.{name}", arg))
            item = scripts.get(name, _Script([])).next()
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, tuple) and item[0] == "steps_then_raise":
                for _ in range(item[1]):
                    self.env.step(np.full(self.dim, float(self.n_calls)))
                raise item[2]
            if item == -1:
                return -1
            for _ in range(int(item)):
                self.env.step(np.full(self.dim, float(self.n_calls)))
            return 0

        def move_to_pose_with_screw(self, pose):
            return self._run("screw", pose)

        def move_to_pose_with_RRTStar(self, pose):
            return self._run("rrt", pose)

        def close_gripper(self):
            return self._run("close", None)

        def open_gripper(self):
            return self._run("open", None)

    class ArmSpy(_Base):
        kind = "arm"
        dim = 8

    class StickSpy(_Base):
        kind = "stick"
        dim = 7

    return ArmSpy, StickSpy


class IKPlannerSpy:
    """Stands in for the PandaArm/Stick planners inside EndeffectorDemonstrationWrapper: only transform_goal_to_wrt_base, IK and robot are used."""

    def __init__(self, log, kind, env, kwargs, solutions, status="Success"):
        self.log, self.kind, self.env, self.kwargs = log, kind, env, kwargs
        self.solutions, self.status = solutions, status
        self.planner = SimpleNamespace(transform_goal_to_wrt_base=self._to_base, IK=self._ik)
        self.robot = SimpleNamespace(get_qpos=lambda: torch.tensor([QPOS9], dtype=torch.float32))

    def _to_base(self, goal):
        self.log.append(("ik.to_base", np.asarray(goal, dtype=np.float64).copy()))
        # base at the world origin without rotation: base-frame goal = world-frame goal; a new array is returned on purpose to distinguish input from output
        return np.asarray(goal, dtype=np.float64) + 0.0

    def _ik(self, goal_base, qpos):
        self.log.append(("ik.solve", np.asarray(goal_base).copy(), np.asarray(qpos).copy()))
        return self.status, self.solutions


def ik_spy_factory(log, solutions, status="Success"):
    """Return (ArmFactory, StickFactory): factories usable like class constructors; construction records ("ik.new", kind, kwargs)."""

    def make(kind):
        def factory(env, **kwargs):
            log.append(("ik.new", kind, dict(kwargs)))
            return IKPlannerSpy(log, kind, env, kwargs, solutions, status)

        return factory

    return make("arm"), make("stick")
