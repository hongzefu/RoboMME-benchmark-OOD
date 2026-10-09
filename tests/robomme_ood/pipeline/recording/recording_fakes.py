"""CPU stand-ins and independent event tables for the recording and read-back chain (C10/C11).

The stand-ins only provide the attributes and observation keys actually read by ``RobommeRecordWrapper.step/reset/close``,
build no SAPIEN scene and never touch the GPU. Expected values always come from event tables and hand computation, never by replicating the logic under test:

- after ``action_t`` is "executed" by the stand-in env, the joint reading ``qpos`` is just the first 7 dims of ``action_t``,
  and the two finger positions are decided by the sign of ``action_t[7]`` (>0 open 0.04, otherwise closed 0.0);
- observation image pixel values encode "which env step this is" (reset counts as 0, the t-th step as t),
  so the image of the k-th recorded entry directly tells which step it came from;
- the segmentation map writes the object id in a fixed square region, center computed by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import sapien
import torch

# Synthetic image side length (task requires ≥64×64).
IMG = 64
# Id of the target object and its square region in the segmentation map (rows 10..19, cols 20..29); center by hand is (14, 24).
SEG_ID = 5
SEG_ROWS = (10, 20)
SEG_COLS = (20, 30)
SEG_CENTER_TEXT = "<14, 24>"


def front_rgb(counter: int) -> np.ndarray:
    """Front RGB after the counter-th step (0 for reset); gradient along columns to avoid a constant image."""
    img = np.zeros((IMG, IMG, 3), dtype=np.uint8)
    img[..., 0] = (counter * 7) % 256
    img[..., 1] = np.arange(IMG, dtype=np.uint8)[None, :]
    img[..., 2] = 200
    return img


def wrist_rgb(counter: int) -> np.ndarray:
    img = np.zeros((IMG, IMG, 3), dtype=np.uint8)
    img[..., 0] = 50
    img[..., 1] = (counter * 11) % 256
    img[..., 2] = np.arange(IMG, dtype=np.uint8)[:, None]
    return img


def front_depth(counter: int) -> np.ndarray:
    return np.full((IMG, IMG, 1), 100 + counter, dtype=np.int16)


def wrist_depth(counter: int) -> np.ndarray:
    return np.full((IMG, IMG, 1), 300 + counter, dtype=np.int16)


def segmentation(with_object: bool) -> np.ndarray:
    seg = np.zeros((IMG, IMG, 1), dtype=np.int16)
    if with_object:
        seg[SEG_ROWS[0]:SEG_ROWS[1], SEG_COLS[0]:SEG_COLS[1], 0] = SEG_ID
    return seg


# Camera parameters: extrinsics [I|0] (world is the camera frame), intrinsics fx=fy=10, cx=cy=32.
EXTRINSIC = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float32)
INTRINSIC = np.array([[10.0, 0.0, 32.0], [0.0, 10.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32)
WRIST_EXTRINSIC = (EXTRINSIC * 2.0).astype(np.float32)
WRIST_INTRINSIC = (INTRINSIC * 3.0).astype(np.float32)
# World coordinate of the selected target (0.1, 0.2, 1.0) → pixel x=10*0.1+32=33, y=10*0.2+32=34 → stored as [y, x].
CHOICE_TARGET_XYZ = (0.1, 0.2, 1.0)
CHOICE_POINT_YX = [34, 33]


# The stand-in env's two-finger reading has only two levels (stand-in input, not a constant under test): open and closed.
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.0


def finger_of(gripper_cmd: float) -> float:
    return FINGER_OPEN if gripper_cmd > 0 else FINGER_CLOSED


@dataclass
class Event:
    """One row of the event table: the env's subgoal state and return values at the t-th step."""

    name: str = "pick the cube"  # current_task_name; the recorder skips "NO RECORD"
    demo: bool = False  # current_task_demonstration
    task_index: int = 0
    task_count: int = 3  # len(task_list)
    online_name: str = "online pick"
    subgoal: Optional[str] = "pick the cube at <obj>"
    seg_visible: bool = True
    choice_text: str = ""  # current_choice_label (original option text)
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    # waypoint pending before this step: dict(p, q, type, phase_is_demo) or None
    waypoint: Optional[dict] = None
    elapsed: Optional[int] = None  # overrides elapsed_steps (defaults to the step count)
    # demonstration flag rewritten by an outer layer (e.g. a demonstration wrapper) before this step; None means unchanged
    pre_demo: Optional[bool] = None


class _Pose:
    def __init__(self, p, q):
        self.p = p
        self.q = q


class _Obj:
    """Stand-in object whose pose can be read by extract_actor_position_xyz."""

    def __init__(self, name: str, xyz):
        self.name = name
        self.pose = _Pose(torch.tensor([list(xyz)], dtype=torch.float32), torch.tensor([[1.0, 0, 0, 0]]))


class _Robot:
    def __init__(self):
        self.pose = sapien.Pose()
        self.qpos = torch.zeros((1, 9), dtype=torch.float32)


class _Link:
    def __init__(self, pose):
        self.pose = pose


class _Agent:
    def __init__(self):
        self.robot = _Robot()
        # tcp pose: position varies with step, quaternion is identity (rpy by hand is 0)
        self.tcp = _Link(_Pose(torch.zeros((1, 3), dtype=torch.float64), torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)))


def tcp_xyz(counter: int) -> list[float]:
    return [0.01 * counter, -0.02 * counter, 0.5]


class FakeTaskEnv(gym.Env):
    """CPU stand-in env: advances the subgoal state per the event table; ``step`` does no physics."""

    metadata: dict = {}

    def __init__(self, events: list[Event], *, env_id: str = "FakeTask", difficulty: str = "easy"):
        super().__init__()
        self.events = list(events)
        self.env_id = env_id
        self.difficulty = difficulty
        self.agent = _Agent()
        self.target_obj = _Obj("target", CHOICE_TARGET_XYZ)
        self.segmentation_id_map = {SEG_ID: self.target_obj, 9: _Obj("table-workspace", (0, 0, 0))}
        self.current_segment = self.target_obj
        self.current_segment_online = self.target_obj
        self.counter = 0
        self.reset_calls = 0
        self.close_calls = 0
        self.received_actions: list[Any] = []
        self._pending_waypoint = None
        self.use_fail_planner = False
        self._apply_state(Event(name="NO RECORD", task_index=0))

    # RecordWrapper reads these attributes via both self.unwrapped.X and wrapper.__getattr__
    def _apply_state(self, ev: Event) -> None:
        self.current_task_name = ev.name
        self.current_task_demonstration = ev.demo
        self.current_task_index = ev.task_index
        self.task_list = [f"t{i}" for i in range(ev.task_count)]
        self.current_task_name_online = ev.online_name
        self.current_subgoal_segment = ev.subgoal
        self.current_subgoal_segment_online = ev.subgoal
        self.current_choice_label = ev.choice_text
        self._seg_visible = ev.seg_visible

    def _obs(self) -> dict:
        c = self.counter
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None]  # noqa: E731
        return {
            "sensor_data": {
                "base_camera": {
                    "rgb": t(front_rgb(c)),
                    "depth": t(front_depth(c)),
                    "segmentation": t(segmentation(self._seg_visible)),
                },
                "hand_camera": {"rgb": t(wrist_rgb(c)), "depth": t(wrist_depth(c))},
            },
            "sensor_param": {
                "base_camera": {"extrinsic_cv": t(EXTRINSIC), "intrinsic_cv": t(INTRINSIC)},
                "hand_camera": {"extrinsic_cv": t(WRIST_EXTRINSIC), "intrinsic_cv": t(WRIST_INTRINSIC)},
            },
        }

    def reset(self, *, seed=None, options=None):
        self.counter = 0
        self.reset_calls += 1
        self.elapsed_steps = 0
        self._pending_waypoint = None  # the env of a new episode does not carry the previous episode's pending waypoint
        self.agent.robot.qpos = torch.zeros((1, 9), dtype=torch.float32)
        self.agent.tcp.pose.p = torch.zeros((1, 3), dtype=torch.float64)
        self._apply_state(Event(name="NO RECORD", task_index=0, demo=bool(self.events and self.events[0].demo)))
        return self._obs(), {"reset": True}

    def step(self, action):
        ev = self.events[self.counter]
        self.counter += 1
        self.received_actions.append(action)
        self.elapsed_steps = ev.elapsed if ev.elapsed is not None else self.counter
        a = np.asarray(action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action, dtype=np.float64).reshape(-1)
        g = finger_of(float(a[7])) if a.size >= 8 else 0.0
        self.agent.robot.qpos = torch.tensor([list(a[:7]) + [g, g]], dtype=torch.float32)
        self.agent.tcp.pose.p = torch.tensor([tcp_xyz(self.counter)], dtype=torch.float64)
        self._apply_state(ev)
        info = {"success": torch.tensor([ev.success]), "fail": torch.tensor([not ev.success and ev.terminated])}
        return (
            self._obs(),
            torch.tensor([0.0]),
            torch.tensor([ev.terminated]),
            torch.tensor([ev.truncated]),
            info,
        )

    def close(self):
        self.close_calls += 1


def action_of(t: int) -> np.ndarray:
    """8-dim joint action sent at the t-th step (counted from 1); gripper sign alternates."""
    base = np.array([0.1 * t + 0.01 * j for j in range(7)], dtype=np.float64)
    return np.concatenate([base, [1.0 if t % 2 else -1.0]])


def drive(w, env, events: list[Event], *, actions=None, first_t: int = 1) -> list:
    """Feed the event table step by step into the real RecordWrapper.step; return each step's return value."""
    env.events = list(events)
    env.counter = 0
    returns = []
    for i, ev in enumerate(events):
        t = first_t + i
        if ev.pre_demo is not None:
            env.current_task_demonstration = ev.pre_demo
        if ev.waypoint is not None:
            env._pending_waypoint = dict(ev.waypoint)
        act = actions[i] if actions is not None else torch.from_numpy(action_of(t))
        returns.append(w.step(act))
    return returns


def make_wrapper(record_cls, tmp_path: Path, events: list[Event], *, episode: int = 3, seed: int = 77,
                 save_video: bool = True, env_id: str = "FakeTask"):
    env = FakeTaskEnv(events, env_id=env_id)
    w = record_cls(env, dataset=str(tmp_path / "out"), env_id=env_id, episode=episode, seed=seed, save_video=save_video)
    return w, env


def run_episode(record_cls, tmp_path: Path, events: list[Event], *, episode: int = 3, seed: int = 77,
                save_video: bool = True, env_id: str = "FakeTask", actions=None, close: bool = True):
    """Real RecordWrapper: reset → step × N → (optional) close. Returns (wrapper, env, h5 path, per-step return values)."""
    w, env = make_wrapper(record_cls, tmp_path, events, episode=episode, seed=seed, save_video=save_video, env_id=env_id)
    w.reset()
    returns = drive(w, env, events, actions=actions)
    if close:
        w.close()
    return w, env, w.dataset_path, returns


# ---------------------------------------------------------------- h5 schema

# h5 structure written by the recorder (current official h5_data_format); added or dropped fields must be caught by the tests.
TIMESTEP_GROUPS = {"obs", "action", "info"}
OBS_KEYS = {
    "front_rgb", "wrist_rgb", "front_depth", "wrist_depth", "joint_state", "gripper_state",
    "is_gripper_close", "front_camera_extrinsic", "wrist_camera_extrinsic", "eef_state",
}
ACTION_KEYS = {"joint_action", "eef_action", "waypoint_action", "choice_action"}
INFO_KEYS = {
    "simple_subgoal", "simple_subgoal_online", "grounded_subgoal", "grounded_subgoal_online",
    "is_completed", "is_video_demo", "is_subgoal_boundary",
}
SETUP_KEYS_BASE = {"seed", "available_multi_choices", "difficulty", "front_camera_intrinsic", "wrist_camera_intrinsic"}


# ---------------------------------------------------------------- helpers


def read_tree(path: Path) -> dict:
    """Read an h5 into {path: (dtype, shape, raw value)}, plus {path@attrs: dict}; used for "two copies identical" checks."""
    import h5py

    out: dict = {}

    def visit(name, obj):
        out[f"{name}@attrs"] = dict(obj.attrs)
        if isinstance(obj, h5py.Dataset):
            out[name] = (str(obj.dtype), obj.shape, obj[()])

    with h5py.File(path, "r") as f:
        out["/@attrs"] = dict(f.attrs)
        f.visititems(visit)
    return out


def trees_diff(a: dict, b: dict) -> list[str]:
    """Return the list of differences between two h5 trees (empty means identical key by key, dtype by dtype, value by value; NaN treated as equal)."""
    diffs = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            diffs.append(f"missing key {k}")
            continue
        va, vb = a[k], b[k]
        if k.endswith("@attrs"):
            if va != vb:
                diffs.append(f"attrs differ {k}")
            continue
        if va[0] != vb[0] or va[1] != vb[1]:
            diffs.append(f"dtype/shape differs {k}: {va[:2]} vs {vb[:2]}")
            continue
        xa, xb = np.asarray(va[2]), np.asarray(vb[2])
        same = np.array_equal(xa, xb, equal_nan=True) if xa.dtype.kind == "f" else np.array_equal(xa, xb)
        if not same:
            diffs.append(f"value differs {k}")
    return diffs


def record_module(kind: str):
    """kind = official/hard: return the RecordWrapper module object of the corresponding package."""
    import importlib

    name = {"official": "robomme", "hard": "robomme_ood"}[kind]
    return importlib.import_module(f"{name}.env_record_wrapper.RecordWrapper")
