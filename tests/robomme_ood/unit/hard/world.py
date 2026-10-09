"""CPU world stand-in: arm (grasp, TCP, joints), button joints, clock; drives the task class's **real** ``evaluate``/``step``.

Real predicates (``is_obj_pickup``, ``is_obj_dropped_onto``, ``is_button_pressed``, ``static_check`` ...) read
``actor.pose``, ``agent.is_grasping(obj)``, ``agent.tcp.pose``, ``agent.robot.get_qvel()``, button ``get_qpos()``
and ``elapsed_steps``; this stand-in only provides these states, the predicates and task tables themselves all run production code.

``BaseEnv.step`` is replaced with a minimal stand-in (clock + 1 → ``evaluate()`` → terminated from success/fail); the pre/post processing of the task class's overridden
``step`` runs as usual.
"""
from __future__ import annotations

import contextlib

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv

from robomme_ood.robomme_env.utils import reset_panda

from . import offline_scene as O
# The arm stand-in is defined in the offline scene: real BaseEnv runs _load_agent before _load_scene, so the agent must exist when the scene is built
from .offline_scene import TCP_UP_Z, FakeAgent, _pose_at

TABLE_Z = 0.02  # order of magnitude of a cube's center height on the table (only used as z when "dropping"; verdict thresholds come from the production predicates themselves)
LIFT_Z = 0.12  # height after lifting: above is_obj_pickup's 0.05, below is_bin_pickup's 0.15 (containers get their own height)


def _fake_base_step(self, action=None):
    """Stand-in for ``BaseEnv.step``: clock + 1, call the real ``evaluate``, terminated from success/fail."""
    self._elapsed_steps = self._elapsed_steps + 1
    info = self.evaluate()
    info = dict(info)
    info["elapsed_steps"] = self._elapsed_steps
    terminated = bool(torch.as_tensor(info["success"]).any() or torch.as_tensor(info["fail"]).any())
    return None, 0.0, terminated, False, info


@contextlib.contextmanager
def cpu_world():
    """Offline scene + ``BaseEnv.step`` stand-in (``elapsed_steps`` is a read-only BaseEnv property reading ``_elapsed_steps``)."""
    with O.offline_scene():
        saved = BaseEnv.__dict__["step"]
        BaseEnv.step = _fake_base_step
        try:
            yield
        finally:
            BaseEnv.step = saved


class World:
    """One CPU-world episode: builds the task (offline ``_load_scene`` + ``_initialize_episode``) and provides primitives for hand-written events."""

    def __init__(self, env):
        self.env = env
        self.agent = env.agent if isinstance(getattr(env, "agent", None), FakeAgent) else FakeAgent()
        env.agent = self.agent
        env._elapsed_steps = torch.tensor([0], dtype=torch.int32)
        self.agent.reset(reset_panda.get_reset_panda_param("qpos"))

    # ── build ────────────────────────────────────────────────────────────
    @classmethod
    def from_env(cls, env):
        """Run the real ``_initialize_episode`` twice on an env that has already run ``_load_scene``: in the evaluation chain ``gym.make``
        does one BaseEnv reset (including reconfigure → ``_load_scene``) and the evaluation resets once more, so initialization happens exactly twice
        (this is where the ``initializations.0``/``.1`` sections of the packaged specs come from)."""
        world = cls(env)
        # Same as the evaluation chain: DemonstrationWrapper sets use_demonstrationwrapper=True at construction, and after the demonstration
        # demonstration_record_traj=False (each evaluate in the execution segment updates the current subgoal); tests switch the demonstration segment with demo_phase()
        env.use_demonstrationwrapper = True
        env.demonstration_record_traj = False
        for _ in range(2):
            env._initialize_episode(torch.arange(1), {})
            # BaseEnv.reset fetches info once after initialization (get_info → evaluate); the task class's step depends on the state it sets
            env.evaluate()
        return world

    @classmethod
    def build(cls, task: str, tier: str, k: int = 0, *, spec=None):
        """Build the scene by replaying the spec of the k-th packaged formal episode (same parameters as the evaluation chain), then run the real ``_initialize_episode`` twice."""
        header, rows = O.delivered_rows(task, tier, k + 1)
        row = rows[k]
        env = O.make_offline(task, seed=row["seed"], difficulty=tier,
                             sampling_config=header["sampling_config"][task],
                             spec=row["spec"] if spec is None else spec)
        return cls.from_env(env)

    @contextlib.contextmanager
    def demo_phase(self):
        """Demonstration segment: same as when DemonstrationWrapper runs a demonstration task, sets demonstration_record_traj=True and restores it afterwards."""
        self.env.demonstration_record_traj = True
        try:
            yield self
        finally:
            self.env.demonstration_record_traj = False

    # ── primitives ────────────────────────────────────────────────────────────
    @staticmethod
    def xyz(actor) -> np.ndarray:
        return actor.pose.p[0].detach().cpu().numpy().astype(np.float64)

    def move(self, actor, xyz, q=None):
        q = q if q is not None else actor.pose.q[0].tolist()
        actor.set_pose(sapien.Pose(p=[float(v) for v in xyz], q=[float(v) for v in q]))

    def tcp_to(self, xyz):
        self.agent.tcp.pose = _pose_at(xyz)

    def grasp(self, actor, z=LIFT_Z):
        """Pick up and lift to z: grasp the object, TCP follows to the object."""
        x, y, _ = self.xyz(actor)
        self.move(actor, (x, y, z))
        self.agent.held = actor
        self.tcp_to((x, y, z))

    def release_onto(self, actor, xy, z=TABLE_Z):
        """Release at xy onto the table, lift the TCP away."""
        self.move(actor, (xy[0], xy[1], z))
        if self.agent.held is actor:
            self.agent.held = None
        self.tcp_to((xy[0], xy[1], TCP_UP_Z))

    def press(self, button, depth=None):
        """Press the button: joint position = -depth (production ``get_button_depth`` negates); defaults to pressing to the bottom of travel."""
        travel = float(getattr(self.env, "button_travel", 0.0) or 0.0)
        d = travel if depth is None else depth
        button.set_qpos([-d])

    def unpress(self, button):
        button.set_qpos([0.0])

    def still(self):
        self.agent.robot.set_qvel(torch.zeros((1, 9)))

    def moving(self):
        self.agent.robot.set_qvel(torch.ones((1, 9)))

    # ── advance ────────────────────────────────────────────────────────────
    def evaluate(self, solve_complete_eval=False) -> dict:
        info = self.env.evaluate(solve_complete_eval=solve_complete_eval)
        return {"success": bool(torch.as_tensor(info["success"]).any()),
                "fail": bool(torch.as_tensor(info["fail"]).any())}

    def tick(self, n: int = 1) -> dict:
        """Advance the clock n steps (evaluate once per step); return the success/fail of the last step."""
        out = None
        for _ in range(n):
            self.env._elapsed_steps = self.env._elapsed_steps + 1
            out = self.evaluate()
        return out

    def step(self, n: int = 1) -> dict:
        """Advance n steps through the task class's real ``step`` (its pre/post processing runs as usual); return the success/fail of the last step."""
        out = None
        for _ in range(n):
            _, _, _, _, info = self.env.step(None)
            out = {"success": bool(torch.as_tensor(info["success"]).any()),
                   "fail": bool(torch.as_tensor(info["fail"]).any())}
        return out

    @property
    def stage(self) -> int:
        """Current task index of sequential_task_check (production attribute ``timestep``)."""
        return int(getattr(self.env, "timestep", 0))
