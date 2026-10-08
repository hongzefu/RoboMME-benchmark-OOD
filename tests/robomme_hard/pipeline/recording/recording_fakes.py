"""录制与读回链路（C10／C11）的 CPU 替身与独立事件表。

替身只提供 ``RobommeRecordWrapper.step／reset／close`` 实际读取的属性与观测键，
不构建任何 SAPIEN 场景、不触碰 GPU。期望值一律由事件表与手算得出，不复刻被测逻辑：

- ``action_t`` 经替身环境「执行」后，关节读数 ``qpos`` 直接取 ``action_t`` 的前 7 维，
  夹爪两指位置由 ``action_t[7]`` 的符号决定（>0 张开 0.04，否则闭合 0.0）；
- 观测图像的像素值由「环境第几次 step」编码（reset 计 0，第 t 次 step 计 t），
  因此录制里第 k 条记录的图像能直接反推它来自哪一次 step；
- 分割图在固定方块区域写入物体 id，中心坐标手算。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import sapien
import torch

# 合成图像边长（任务要求 ≥64×64）。
IMG = 64
# 分割图里目标物体的 id 与方块区域（行 10..19、列 20..29），中心手算为 (14, 24)。
SEG_ID = 5
SEG_ROWS = (10, 20)
SEG_COLS = (20, 30)
SEG_CENTER_TEXT = "<14, 24>"


def front_rgb(counter: int) -> np.ndarray:
    """第 counter 次 step 之后的前视 RGB（reset 为 0）；列方向带梯度，避免整幅常数。"""
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


# 相机参数：外参取 [I|0]（世界即相机系），内参 fx=fy=10、cx=cy=32。
EXTRINSIC = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float32)
INTRINSIC = np.array([[10.0, 0.0, 32.0], [0.0, 10.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32)
WRIST_EXTRINSIC = (EXTRINSIC * 2.0).astype(np.float32)
WRIST_INTRINSIC = (INTRINSIC * 3.0).astype(np.float32)
# 选择目标的世界坐标 (0.1, 0.2, 1.0) → 像素 x=10*0.1+32=33，y=10*0.2+32=34 → 存 [y, x]。
CHOICE_TARGET_XYZ = (0.1, 0.2, 1.0)
CHOICE_POINT_YX = [34, 33]


# 替身环境的两指读数只有两档（替身输入，不是被测常量）：张开与闭合。
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.0


def finger_of(gripper_cmd: float) -> float:
    return FINGER_OPEN if gripper_cmd > 0 else FINGER_CLOSED


@dataclass
class Event:
    """事件表一行：第 t 次 step 时环境所处的子目标状态与环境返回值。"""

    name: str = "pick the cube"  # current_task_name；"NO RECORD" 时录像器跳过
    demo: bool = False  # current_task_demonstration
    task_index: int = 0
    task_count: int = 3  # len(task_list)
    online_name: str = "online pick"
    subgoal: Optional[str] = "pick the cube at <obj>"
    seg_visible: bool = True
    choice_text: str = ""  # current_choice_label（选项原文）
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    # 本步执行前挂起的 waypoint：dict(p, q, type, phase_is_demo) 或 None
    waypoint: Optional[dict] = None
    elapsed: Optional[int] = None  # 覆盖 elapsed_steps（默认等于 step 计数）
    # 本步执行前由外层（如演示包装器）改写的演示标志；None 表示不改
    pre_demo: Optional[bool] = None


class _Pose:
    def __init__(self, p, q):
        self.p = p
        self.q = q


class _Obj:
    """可被 extract_actor_position_xyz 读取位姿的替身物体。"""

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
        # tcp 位姿：位置随 step 变化，四元数取单位元（rpy 手算为 0）
        self.tcp = _Link(_Pose(torch.zeros((1, 3), dtype=torch.float64), torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)))


def tcp_xyz(counter: int) -> list[float]:
    return [0.01 * counter, -0.02 * counter, 0.5]


class FakeTaskEnv(gym.Env):
    """CPU 替身环境：按事件表推进子目标状态，``step`` 不做任何物理。"""

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

    # RecordWrapper 通过 self.unwrapped.X 与 wrapper.__getattr__ 两条路径读这些属性
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
        self._pending_waypoint = None  # 新一局的环境不带上一局挂起的 waypoint
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
    """第 t 次 step（从 1 计）发出的 8 维关节动作；夹爪符号交替。"""
    base = np.array([0.1 * t + 0.01 * j for j in range(7)], dtype=np.float64)
    return np.concatenate([base, [1.0 if t % 2 else -1.0]])


def drive(w, env, events: list[Event], *, actions=None, first_t: int = 1) -> list:
    """把事件表逐步喂给真实 RecordWrapper.step；返回每步返回值。"""
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
    """真实 RecordWrapper：reset → step × N →（可选）close。返回 (wrapper, env, h5 路径, 每步返回值)。"""
    w, env = make_wrapper(record_cls, tmp_path, events, episode=episode, seed=seed, save_video=save_video, env_id=env_id)
    w.reset()
    returns = drive(w, env, events, actions=actions)
    if close:
        w.close()
    return w, env, w.dataset_path, returns


# ---------------------------------------------------------------- h5 模式

# 录像器写出的 h5 结构（官方 h5_data_format 现状）；新增或丢字段都应被测试抓到。
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


# ---------------------------------------------------------------- 工具


def read_tree(path: Path) -> dict:
    """把 h5 读成 {路径: (dtype, shape, 原始值)}，另含 {路径@attrs: dict}；用于「两份相同」判定。"""
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
    """返回两棵 h5 树的差异列表（空表示逐键、逐 dtype、逐值相同；NaN 视为相等）。"""
    diffs = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            diffs.append(f"缺键 {k}")
            continue
        va, vb = a[k], b[k]
        if k.endswith("@attrs"):
            if va != vb:
                diffs.append(f"属性不同 {k}")
            continue
        if va[0] != vb[0] or va[1] != vb[1]:
            diffs.append(f"dtype/shape 不同 {k}: {va[:2]} vs {vb[:2]}")
            continue
        xa, xb = np.asarray(va[2]), np.asarray(vb[2])
        same = np.array_equal(xa, xb, equal_nan=True) if xa.dtype.kind == "f" else np.array_equal(xa, xb)
        if not same:
            diffs.append(f"值不同 {k}")
    return diffs


def record_module(kind: str):
    """kind = official／hard：返回对应包的 RecordWrapper 模块对象。"""
    import importlib

    name = {"official": "robomme", "hard": "robomme_hard"}[kind]
    return importlib.import_module(f"{name}.env_record_wrapper.RecordWrapper")
