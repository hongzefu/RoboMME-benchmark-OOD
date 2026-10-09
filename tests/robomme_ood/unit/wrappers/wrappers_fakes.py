"""动作空间 wrapper 与演示时序测试（C08.02／C09.02）共用的 CPU 替身，只供 tests/robomme_ood/unit/wrappers/ 使用。

- ``ScriptedEnv``：冒充任务环境（gymnasium.Env），不构建任何 SAPIEN 场景。观测像素值编码「这是第几次底层 step」
  （reset 为第 0 次，第 k 次 step 的前视 RGB 全是 ``frame_value(k)``），因此任何一帧都能反推来自哪一次底层 step；
  相机参数固定为手写的针孔内参 ``K`` 与单位外参，便于手算投影。所有 step／evaluate／reset 调用按发生顺序记进
  共享事件表 ``log``，测试拿手写的期望事件序列逐项比对。
- ``planner_spy_classes``：顶替真实运动规划器（mplib 等）的 CPU spy。构造参数、每次规划调用与参数都写进同一张
  事件表；按脚本决定每次调用「走几步 / 返回 -1 / 抛异常」。
- ``IKPlannerSpy``：顶替 EndeffectorDemonstrationWrapper 的 IK 规划器，记录 IK 输入、按脚本给解。
"""
from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import EnvSpec
from mani_skill.utils.structs.pose import Pose

# 小图：高 48、宽 64；内参主点取图像中心，焦距 100 像素（手算投影用）
H, W = 48, 64
FOCAL = 100.0
K = ((FOCAL, 0.0, W / 2), (0.0, FOCAL, H / 2), (0.0, 0.0, 1.0))
E_ID = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
STICK_IDS = ("PatternLock", "RouteStick")
# 替身机器人关节读数（9 维；stick 任务 7 维），值手写
QPOS9 = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.04, 0.04)
SWING7 = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)


def frame_value(k: int) -> int:
    """第 k 次底层 step 之后前视 RGB 的像素值（reset 为 0）。"""
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
    """可哈希的替身 actor：只有名字与世界坐标（真实 actor 按对象身份哈希）。"""

    def __init__(self, name, xyz):
        self.name = name
        self.pose = SimpleNamespace(p=torch.tensor([xyz], dtype=torch.float32))

    def __repr__(self):
        return f"<Actor {self.name}>"


class ScriptedEnv(gym.Env):
    """脚本化任务环境。outcomes：逐次 step 的 (success, fail)，用完后恒为 (False, False)。"""

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
        # 子目标状态（真实任务由 sequential_task_check 写入；这里由测试的 solve 替身直接改）
        self.current_task_demonstration = False
        self.current_task_name = "online"
        self.current_subgoal_segment = None
        self.current_segment = None
        self.segmentation_id_map = {}
        self.task_list = list(task_list or [])
        # task_goal（PickXtimes）与 vqa_options（PickXtimes）读取的字段
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
    """像 gym.make 一样在任务外套一层 OrderEnforcing（task_goal 会经 ``.env.unwrapped`` 取任务）。"""
    return gym.wrappers.OrderEnforcing(inner)


# --------------------------------------------------------------------------- 规划器 spy


class _Script:
    """一种规划调用的脚本。每项：int n（走 n 个底层步后返回 0）、-1（不走步直接返回 -1）、
    BaseException 实例（直接抛）、("steps_then_raise", n, exc)（先走 n 步再抛）。用完后恒为「走 1 步」。"""

    def __init__(self, items):
        self.items = list(items)

    def next(self):
        return self.items.pop(0) if self.items else 1


def planner_spy_classes(log, scripts=None):
    """返回 (ArmSpy, StickSpy) 两个类，顶替 FailAwarePandaArm／StickMotionPlanningSolver。

    构造时记 ("planner.new", kind, kwargs)；规划调用记 ("planner.<方法>", 参数)。
    每个底层步对 ``self.env.step`` 发一个动作：arm 8 维、stick 7 维，值为本 spy 的累计调用序号（便于辨认）。"""
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
    """顶替 EndeffectorDemonstrationWrapper 内部的 PandaArm／Stick 规划器：只用 transform_goal_to_wrt_base、IK、robot。"""

    def __init__(self, log, kind, env, kwargs, solutions, status="Success"):
        self.log, self.kind, self.env, self.kwargs = log, kind, env, kwargs
        self.solutions, self.status = solutions, status
        self.planner = SimpleNamespace(transform_goal_to_wrt_base=self._to_base, IK=self._ik)
        self.robot = SimpleNamespace(get_qpos=lambda: torch.tensor([QPOS9], dtype=torch.float32))

    def _to_base(self, goal):
        self.log.append(("ik.to_base", np.asarray(goal, dtype=np.float64).copy()))
        # 基座在世界原点、无旋转：基座系目标 = 世界系目标；这里故意返回新数组以区分输入输出
        return np.asarray(goal, dtype=np.float64) + 0.0

    def _ik(self, goal_base, qpos):
        self.log.append(("ik.solve", np.asarray(goal_base).copy(), np.asarray(qpos).copy()))
        return self.status, self.solutions


def ik_spy_factory(log, solutions, status="Success"):
    """返回 (ArmFactory, StickFactory)：可当作类构造的工厂，构造即记 ("ik.new", kind, kwargs)。"""

    def make(kind):
        def factory(env, **kwargs):
            log.append(("ik.new", kind, dict(kwargs)))
            return IKPlannerSpy(log, kind, env, kwargs, solutions, status)

        return factory

    return make("arm"), make("stick")
