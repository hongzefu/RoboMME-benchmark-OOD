"""CPU doubles shared by the official package wrapper-layer unit tests (for tests/robomme_ood/unit/robomme/ only).

- ``FakeTaskEnv``: a gymnasium.Env posing as the task env for wrapper layers such as DemonstrationWrapper; ``step`` returns per script
  observations shaped like ManiSkill (sensor_data as torch tensors with a batch dim), torch-bool terminated/truncated and
  ``info["success"/"fail"]``, and records every action received.
- ``fake_gym_make``: stands in for ``episode_config_resolver.gym.make``, records kwargs and returns a ``FakeTaskEnv``.
"""
from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import EnvSpec
from mani_skill.utils.structs.pose import Pose

H, W = 8, 10  # small images suffice; shape assertions are derived from these


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
    """Hashable actor double with only a name (no pose)."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<Named {self.name}>"


class FakeTaskEnv(gym.Env):
    """Scripted task env. outcomes: (success, fail) for each step; (False, False) forever once exhausted."""

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
        # subgoal state (written by sequential_task_check in real tasks)
        self.current_task_demonstration = demonstration
        self.current_task_name = "pick up the cube"
        self.current_subgoal_segment = None
        self.current_segment = None
        self.segmentation_id_map = {}
        # fields needed by task_goal (PickXtimes)
        self.num_repeats = 2
        self.target_color_name = "red"
        # fields needed by vqa_options (PickXtimes)
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
    """Wrap the task in an OrderEnforcing layer like gym.make does (the real chain also has TimeLimit outside the task; truncation is not needed here).
    DemonstrationWrapper hands ``self.env`` (this layer) to task_goal, which then takes ``.env.unwrapped``."""
    return gym.wrappers.OrderEnforcing(inner)


class GymMakeSpy:
    """Stand-in for gym.make: records (env_id, kwargs) and returns a FakeTaskEnv wrapped in OrderEnforcing."""

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
    raise AssertionError(f"not in the chain: {name}")
