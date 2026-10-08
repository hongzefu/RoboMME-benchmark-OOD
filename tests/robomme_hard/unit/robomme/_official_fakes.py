"""官方包 wrapper 层单元测试共用的 CPU 替身（只供 tests/robomme_hard/unit/robomme/ 使用）。

- ``FakeTaskEnv``：一个 gymnasium.Env，冒充任务环境给 DemonstrationWrapper 等包装层用；``step`` 按脚本返回
  与 ManiSkill 相同形状的观测（sensor_data 带 batch 维的 torch 张量）、torch 布尔的 terminated／truncated 与
  ``info["success"/"fail"]``，并记下收到的每个动作。
- ``fake_gym_make``：顶替 ``episode_config_resolver.gym.make``，记录 kwargs 后返回 ``FakeTaskEnv``。
"""
from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import EnvSpec
from mani_skill.utils.structs.pose import Pose

H, W = 8, 10  # 小图即可；形状断言按这个推出


def make_obs(step_idx: int = 0):
    rgb = torch.full((1, H, W, 3), step_idx % 256, dtype=torch.uint8)
    depth = torch.full((1, H, W, 1), 100 + step_idx, dtype=torch.int16)
    seg = torch.zeros((1, H, W, 1), dtype=torch.int16)
    ext = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    intr = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)
    cam = lambda: {"rgb": rgb.clone(), "depth": depth.clone(), "segmentation": seg.clone()}  # noqa: E731
    return {
        "sensor_data": {"base_camera": cam(), "hand_camera": cam()},
        "sensor_param": {
            "base_camera": {"extrinsic_cv": ext.clone(), "intrinsic_cv": intr.clone()},
            "hand_camera": {"extrinsic_cv": ext.clone() + 100, "intrinsic_cv": intr.clone() * 2},
        },
    }


class Named:
    """只有名字的可哈希替身 actor（无位姿）。"""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<Named {self.name}>"


class FakeTaskEnv(gym.Env):
    """脚本化的任务环境。outcomes: 每次 step 的 (success, fail) ；用完后一直 (False, False)。"""

    metadata = {"render_modes": []}

    def __init__(self, env_id="PickXtimes", outcomes=None, demonstration=False, stick=None, **make_kwargs):
        self.spec = EnvSpec(id=env_id, entry_point="tests:fake")
        self.make_kwargs = make_kwargs
        self.outcomes = list(outcomes or [])
        self.actions = []
        self.step_calls = 0
        self.closed = False
        stick = env_id in ("PatternLock", "RouteStick") if stick is None else stick
        n = 7 if stick else 9
        qpos = torch.arange(n, dtype=torch.float32).reshape(1, n) / 10
        self.agent = SimpleNamespace(
            robot=SimpleNamespace(qpos=qpos, pose=Pose.create_from_pq(torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]]))),
            tcp=SimpleNamespace(pose=Pose.create_from_pq(torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([[1.0, 0, 0, 0]]))),
        )
        # 子目标状态（真实任务由 sequential_task_check 写入）
        self.current_task_demonstration = demonstration
        self.current_task_name = "pick up the cube"
        self.current_subgoal_segment = None
        self.current_segment = None
        self.segmentation_id_map = {}
        # task_goal 需要的字段（PickXtimes）
        self.num_repeats = 2
        self.target_color_name = "red"
        # vqa_options 需要的字段（PickXtimes）
        self.all_cubes = [Named("cube_red_0")]
        self.target = Named("target")
        self.button = Named("button")
        self.swing_qpos = torch.full((1, 7), 0.5)

    def reset(self, *, seed=None, options=None):
        return make_obs(0), {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.step_calls += 1
        success, fail = self.outcomes.pop(0) if self.outcomes else (False, False)
        info = {"success": torch.tensor([bool(success)]), "fail": torch.tensor([bool(fail)])}
        terminated = torch.tensor([bool(success) or bool(fail)])
        return make_obs(self.step_calls), torch.tensor([0.0]), terminated, torch.tensor([False]), info

    def evaluate(self, solve_complete_eval=False):
        return {"success": torch.tensor([False]), "fail": torch.tensor([False])}

    def close(self):
        self.closed = True


def as_made(inner):
    """像 gym.make 一样在任务外面套一层 OrderEnforcing（真实链里任务外还有 TimeLimit，这里不需要截断）。
    DemonstrationWrapper 把 ``self.env``（即这一层）交给 task_goal，后者再取 ``.env.unwrapped``。"""
    return gym.wrappers.OrderEnforcing(inner)


class GymMakeSpy:
    """顶替 gym.make：记录 (env_id, kwargs)，返回套了 OrderEnforcing 的 FakeTaskEnv。"""

    def __init__(self):
        self.calls = []

    def __call__(self, env_id, **kwargs):
        self.calls.append((env_id, dict(kwargs)))
        return as_made(FakeTaskEnv(env_id=env_id, **kwargs))

    def namespace(self):
        return SimpleNamespace(make=self)


def wrapper_chain(env):
    chain, node = [], env
    while isinstance(node, gym.Wrapper):
        chain.append(type(node).__name__)
        node = node.env
    chain.append(type(node).__name__)
    return chain


def find_wrapper(env, name):
    node = env
    while isinstance(node, gym.Wrapper):
        if type(node).__name__ == name:
            return node
        node = node.env
    raise AssertionError(f"链上没有 {name}")
