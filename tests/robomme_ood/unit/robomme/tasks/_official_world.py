"""CPU offline world for the 16 tasks of the official package (for tests/robomme_ood/unit/robomme/tasks/ only).

Approach:
- the task classes' real ``__init__``, ``_load_scene``, ``_initialize_episode``, ``evaluate`` and ``step`` all run as-is;
- only the "build real objects" layer is replaced: builders in the task module namespace such as ``TableSceneBuilder``, ``build_button``,
  ``spawn_random_cube`` are swapped for fakes returning CPU actor doubles; ``highlight_obj``/``highlight_position``
  (build visual highlights only) become no-ops; ``sapien.render.RenderMaterial`` (segfaults on CPU in practice) becomes a placeholder material;
- ``mani_skill``'s ``BaseEnv.__init__`` is replaced by a lazy stub (no scene, no GPU), and ``BaseEnv.step`` by a minimal version in the same order as the real
  ``BaseEnv.step``: elapsed_steps += 1 -> ``evaluate()`` -> terminated = success | fail;
- the robot is replaced by ``FakeAgent``: tcp position, joint qpos/qvel and what is currently grasped are all placed explicitly by the test.

All replacements take effect only inside the ``OfficialWorld`` context and are restored on exit (an exception midway through entering also rolls back what was replaced); no files are changed
(R9: the protected src/robomme only gets in-process, recoverable replacements).

About the resource guard: inside the context ``BaseEnv.__init__`` (the interceptor installed by the guard) is temporarily replaced by a lazy stub. This is safe provided that:
(1) the task instance's ``scene`` is a ``FakeScene`` and every builder in the task module that would touch a real scene has been replaced by a double;
(2) the task module's ``sapien`` name is replaced by a proxy exposing only ``Pose`` and a placeholder ``RenderMaterial``, so the render system is never initialized;
(3) the interceptor is restored as-is after leaving the context (see ``test_official_world_guard``). Do not reuse this context if these three do not hold.
Placement heights and thresholds all come from the pin file ``tests/robomme_ood/unit/robomme/official_thresholds.py``.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import Pose

from robomme.robomme_env.utils import object_generation as og
from robomme.robomme_env.utils import reset_panda
from tests.robomme_ood.unit.robomme import official_thresholds as T

STICK_TASKS = ("PatternLock", "RouteStick")
# robot base position (the real Panda is placed at x=-0.615 in TableSceneBuilder); used only by InsertPeg to judge "near end/far end"
ROBOT_BASE_XYZ = (-0.615, 0.0, 0.0)


# --------------------------------------------------------------------------- Actor doubles


def _to_pose(pose_like) -> Pose:
    """Normalize sapien.Pose/mani_skill Pose/(p, q) into a mani_skill Pose with a batch dim."""
    if isinstance(pose_like, Pose):
        return Pose.create_from_pq(pose_like.p.clone().reshape(1, 3).float(), pose_like.q.clone().reshape(1, 4).float())
    if isinstance(pose_like, sapien.Pose):
        p, q = np.asarray(pose_like.p, dtype=np.float32), np.asarray(pose_like.q, dtype=np.float32)
    else:
        p, q = pose_like
        p = np.asarray(p, dtype=np.float32).reshape(-1)[:3]
        q = np.asarray(q, dtype=np.float32).reshape(-1)[:4]
    return Pose.create_from_pq(torch.tensor([p.tolist()], dtype=torch.float32), torch.tensor([q.tolist()], dtype=torch.float32))


class FakeActor:
    """Actor with only a name and a pose; pose shapes match real actors (p: (1,3) float32, q: (1,4))."""

    dof = 0

    def __init__(self, name, xyz, q=(1.0, 0.0, 0.0, 0.0), half=None):
        self.name = name
        self.pose = _to_pose((xyz, q))
        if half is not None:
            self._cube_half_size = float(half)

    def __repr__(self):  # easier to read on assertion failure
        return f"<FakeActor {self.name} {self.xyz.round(3).tolist()}>"

    @property
    def xyz(self) -> np.ndarray:
        return self.pose.p[0].detach().cpu().numpy().astype(np.float64)

    def set_pose(self, pose_like):
        self.pose = _to_pose(pose_like)

    def get_pose(self):
        return self.pose

    def move_to(self, x=None, y=None, z=None):
        cur = self.xyz
        new = [cur[0] if x is None else x, cur[1] if y is None else y, cur[2] if z is None else z]
        self.pose = _to_pose((new, self.pose.q[0].detach().cpu().numpy()))

    def set_linear_velocity(self, v):
        pass

    def set_angular_velocity(self, v):
        pass

    def set_qpos(self, q):
        pass

    def set_qvel(self, q):
        pass


class FakeButton(FakeActor):
    """Button: ``get_qpos`` returns ``[[-depth]]`` (the real joint moves negative when pressed)."""

    def __init__(self, name, xyz):
        super().__init__(name, xyz)
        self.depth = 0.0

    def get_qpos(self):
        return torch.tensor([[-float(self.depth)]], dtype=torch.float32)


class FakePeg(FakeActor):
    """Peg body + head and tail links; on body set_pose the head and tail follow along x (the real peg is an articulated body with head and tail fixed)."""

    def __init__(self, name, xyz, length):
        super().__init__(name, xyz)
        self.length = float(length)
        self.head = FakeActor(f"{name}_head", (xyz[0] + length / 2, xyz[1], xyz[2]))
        self.tail = FakeActor(f"{name}_tail", (xyz[0] - length / 2, xyz[1], xyz[2]))

    def set_pose(self, pose_like):
        super().set_pose(pose_like)
        x, y, z = self.xyz
        self.head.move_to(x + self.length / 2, y, z)
        self.tail.move_to(x - self.length / 2, y, z)


class FakeRobot:
    def __init__(self, stick: bool):
        self.pose = _to_pose((ROBOT_BASE_XYZ, (1.0, 0.0, 0.0, 0.0)))
        qpos = reset_panda.get_reset_panda_param("qpos", gripper="stick" if stick else None)
        self.qpos = torch.tensor([np.asarray(qpos, dtype=np.float32).tolist()])
        self.qvel = torch.zeros_like(self.qpos)

    def get_qpos(self):
        return self.qpos

    def get_qvel(self):
        return self.qvel


class FakeAgent:
    def __init__(self, stick: bool):
        self.robot = FakeRobot(stick)
        self.tcp = FakeActor("tcp", (0.0, 0.0, T.CARRY_HIGH_Z))
        self.held = None

    def reset(self, qpos):
        self.robot.qpos = torch.tensor([np.asarray(qpos, dtype=np.float32).reshape(-1).tolist()])
        self.robot.qvel = torch.zeros_like(self.robot.qpos)

    def is_grasping(self, obj):
        return torch.tensor([obj is not None and obj is self.held])


class _FakeActorBuilder:
    def __init__(self):
        self._pose = sapien.Pose()

    def set_initial_pose(self, pose):
        self._pose = pose

    def add_box_visual(self, *a, **k):
        pass

    def add_box_collision(self, *a, **k):
        pass

    def build_kinematic(self, name):
        return FakeActor(name, self._pose.p, self._pose.q)

    build_static = build_dynamic = build_kinematic


class FakeScene:
    def create_actor_builder(self):
        return _FakeActorBuilder()


class _FakeMaterial:
    def __init__(self, *a, **k):
        pass

    def set_base_color(self, *a, **k):
        pass


# --------------------------------------------------------------------------- Fake builders


def _free_xy(env, center, min_dist=0.1):
    """Find a position near center at distance > min_dist from already placed doubles (deterministic, consumes no random numbers)."""
    placed = getattr(env, "_fake_placed", [])
    cx, cy = float(center[0]), float(center[1])
    step = 0.11
    offsets = [(0, 0)] + [(dx * step, dy * step) for r in range(1, 6) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                          if max(abs(dx), abs(dy)) == r]
    for dx, dy in offsets:
        x, y = cx + dx, cy + dy
        if all(np.hypot(x - px, y - py) > min_dist for px, py in placed):
            placed.append((x, y))
            env._fake_placed = placed
            return x, y
    raise AssertionError("offline world cannot fit more doubles")


def _center_of(value, default=(0.0, 0.0)):
    if value is None:
        return default
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(arr[0]), float(arr[1])


def fake_build_button(self, center_xy, *, generator=None, name="button", randomize=True, randomize_range=(0.1, 0.4),
                      **_ignored):
    """Button double. Official tasks always pass center_xy explicitly; the randomize_range default matches the official build_button signature."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    if randomize:  # consume the same random numbers as the real build_button, keeping the subsequent sampling stream position unchanged
        off = torch.rand(2, generator=generator) - 0.5
        cx += float(off[0]) * float(randomize_range[0])
        cy += float(off[1]) * float(randomize_range[1])
    button = FakeButton(name, (cx, cy, 0.0))
    self.button = button
    self.button_joint = None
    if not hasattr(self, "cap_links"):
        self.cap_links = {}
    self.cap_links[name] = [FakeActor(f"{name}_cap", (cx, cy, 0.0))]
    self.cap_link = self.cap_links[name]
    placed = getattr(self, "_fake_placed", [])
    placed.append((cx, cy))
    self._fake_placed = placed
    return og.create_button_obb(center_xy=(cx, cy))


def fake_spawn_random_cube(self, *, region_center=(0, 0), half_size, name_prefix="cube_extra", **_ignored):
    x, y = _free_xy(self, _center_of(region_center))
    return FakeActor(name_prefix, (x, y, float(half_size)), half=half_size)


def fake_spawn_random_target(self, *, region_center=(0, 0), name_prefix="target", **_ignored):
    x, y = _free_xy(self, _center_of(region_center))
    return FakeActor(name_prefix, (x, y, 0.0))


def fake_spawn_random_bin(self, *, region_center=(0, 0), name_prefix="bin", **_ignored):
    x, y = _free_xy(self, _center_of(region_center))
    return FakeActor(name_prefix, (x, y, T.TABLE_Z))


def fake_spawn_fixed_cube(self, position, half_size=None, name_prefix="fixed_cube", **_ignored):
    hs = float(half_size if half_size is not None else self.cube_half_size)
    p = list(np.asarray(position, dtype=np.float64).reshape(-1))
    z = p[2] if len(p) > 2 else hs
    return FakeActor(name_prefix, (p[0], p[1], z), half=hs)


def fake_build_board_with_hole(self, *, position, name="board_with_hole", **_ignored):
    p = list(position) + [0.0] * (3 - len(position))
    return FakeActor(name, p[:3])


def fake_build_disk_target(scene, *, name, initial_pose=None, **_ignored):
    pose = initial_pose if initial_pose is not None else sapien.Pose()
    return FakeActor(name, pose.p, pose.q)


def fake_build_peg(self, length, radius, initial_pose=None, name="peg", head_color=None, tail_color=None, **_ignored):
    p = initial_pose.p if initial_pose is not None else (0.0, 0.0, 0.0)
    peg = FakePeg(name, tuple(float(v) for v in p), length)
    return peg, peg.head, peg.tail


def fake_build_box_with_hole(self, center=(0, 0), **_ignored):
    return FakeActor("box_with_hole", (float(center[0]), float(center[1]), 0.0))


class FakeTableSceneBuilder:
    def __init__(self, env, robot_init_qpos_noise=0):
        pass

    def build(self):
        pass

    def initialize(self, env_idx):
        pass


def _noop(*a, **k):
    return None


def _inert_base_init(self, *args, **kwargs):
    """Lazy stub for BaseEnv.__init__: builds no scene, touches no rendering or GPU."""


def _fake_base_step(self, action):
    """Same order as mani_skill BaseEnv.step: elapsed_steps += 1 -> evaluate -> terminated = success | fail."""
    self._elapsed_steps = self._elapsed_steps + 1
    info = {"elapsed_steps": self._elapsed_steps}
    info.update(self.evaluate())
    terminated = torch.logical_or(info["success"], info["fail"])
    return {}, torch.zeros(1), terminated, torch.zeros(1, dtype=torch.bool), info


MODULE_PATCHES = {
    "TableSceneBuilder": FakeTableSceneBuilder,
    "build_button": fake_build_button,
    "spawn_random_cube": fake_spawn_random_cube,
    "spawn_random_target": fake_spawn_random_target,
    "spawn_random_bin": fake_spawn_random_bin,
    "spawn_fixed_cube": fake_spawn_fixed_cube,
    "build_board_with_hole": fake_build_board_with_hole,
    "build_purple_white_target": fake_build_disk_target,
    "build_gray_white_target": fake_build_disk_target,
    "build_peg": fake_build_peg,
    "build_box_with_hole": fake_build_box_with_hole,
    "highlight_obj": _noop,
    "highlight_position": _noop,
}


def task_module(task: str):
    return importlib.import_module(f"robomme.robomme_env.{task}")


class OfficialWorld:
    """Context manager: installs all offline-world replacements and restores them one by one on exit."""

    def __init__(self, task: str):
        self.task = task
        self.module = task_module(task)
        self._saved = []

    def _swap(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def __enter__(self):
        try:
            for name, value in MODULE_PATCHES.items():
                if hasattr(self.module, name):
                    self._swap(self.module, name, value)
            proxy = SimpleNamespace(Pose=sapien.Pose, render=SimpleNamespace(RenderMaterial=_FakeMaterial))
            self._swap(self.module, "sapien", proxy)
            self._swap(BaseEnv, "__init__", _inert_base_init)
            self._swap(BaseEnv, "step", _fake_base_step)
        except BaseException:
            self._restore()  # failed while half installed: roll back replaced items, then re-raise
            raise
        return self

    def _restore(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)
        self._saved.clear()

    def __exit__(self, *exc):
        self._restore()
        return False

    def make(self, difficulty: str, seed: int = 0, **kwargs) -> "Episode":
        cls = getattr(self.module, self.task)
        env = cls(seed=seed, difficulty=difficulty, **kwargs)
        env.device = torch.device("cpu")
        env.num_envs = 1
        env.scene = FakeScene()
        env.agent = FakeAgent(stick=self.task in STICK_TASKS)
        env._elapsed_steps = torch.zeros(1, dtype=torch.int32)
        # DemonstrationWrapper sets use_demonstrationwrapper=True at construction; the demo recording flag is False in the online segment
        # (subgoal switching is allowed on every step in the online segment, same configuration as real evaluation)
        env.use_demonstrationwrapper = True
        env.demonstration_record_traj = False
        env._load_scene({})
        env._initialize_episode(torch.arange(1), {})
        # the real BaseEnv.reset calls evaluate once via get_info when taking the observation at the end (MoveCube's task_list is built inside evaluate)
        env.evaluate()
        return Episode(env)


class Episode:
    """One offline episode + action primitives for placing the world. All verdicts come from the real ``env.step`` -> ``evaluate``."""

    LIFT_Z = T.LIFT_Z     # object height after pickup (> T.PICKUP_Z)
    TABLE_Z = T.TABLE_Z   # object height after drop (<= T.DROPPED_Z)

    def __init__(self, env):
        self.env = env
        self.agent = env.agent
        self.info = None
        self.history = []

    # ---- State ----
    @property
    def task_index(self) -> int:
        return int(getattr(self.env, "timestep", 0))

    @property
    def task_names(self):
        return [t.get("name") for t in self.env.task_list]

    def first_online_index(self) -> int:
        for i, t in enumerate(self.env.task_list):
            if not t.get("demonstration", False):
                return i
        raise AssertionError("no online subtask")

    @property
    def success(self) -> bool:
        return bool(self.info["success"].item()) if self.info is not None else False

    @property
    def fail(self) -> bool:
        return bool(self.info["fail"].item()) if self.info is not None else False

    # ---- Advancing ----
    def step(self, n: int = 1):
        for _ in range(n):
            _obs, _r, terminated, _tr, self.info = self.env.step(None)
            self.history.append((self.success, self.fail, bool(terminated.item())))
        return self.info

    def skip_demo(self, elapsed: int | None = None):
        """Move the subtask pointer to the first online subtask (the demo segment is executed by motion planning, which cannot run on CPU),
        and fill in the observable state left by the demo function at the end of the demo (after_demo, reset_in_proecess)."""
        self.env.timestep = self.first_online_index()
        if hasattr(self.env, "after_demo"):
            self.env.after_demo = True
        if hasattr(self.env, "reset_in_proecess"):
            self.env.reset_in_proecess = False
        if elapsed is not None:
            self.env._elapsed_steps = torch.tensor([elapsed], dtype=torch.int32)

    # ---- World actions ----
    def tcp_to(self, x, y, z):
        self.agent.tcp.move_to(x, y, z)

    def grasp(self, obj, z: float | None = None):
        """Pick up obj: the robot reports grasping it, and the object rises to z together with the tcp."""
        z = self.LIFT_Z if z is None else z
        x, y, _ = obj.xyz
        self.agent.held = obj
        obj.move_to(z=z)
        self.tcp_to(x, y, z)

    def carry(self, obj, x, y, z: float | None = None):
        z = self.LIFT_Z if z is None else z
        obj.move_to(x, y, z)
        self.tcp_to(x, y, z)

    def release(self, obj, x=None, y=None, z: float | None = None, tcp_z: float = T.TCP_UP_Z):
        """Release and drop: no longer grasped, the object falls to table height, the tcp rises to tcp_z (> T.PICKUP_Z)."""
        z = self.TABLE_Z if z is None else z
        if self.agent.held is obj:
            self.agent.held = None
        obj.move_to(x, y, z)
        ox, oy, _ = obj.xyz
        self.tcp_to(ox, oy, tcp_z)

    def place_on(self, obj, target):
        tx, ty, _ = target.xyz
        self.carry(obj, tx, ty)
        self.release(obj, tx, ty)

    def press(self, button, depth: float = 2 * T.BUTTON_DEPTH):
        button.depth = depth

    def unpress(self, button):
        button.depth = 0.0

    def close_gripper(self):
        q = self.agent.robot.qpos.clone()
        q[0, -2:] = 0.0
        self.agent.robot.qpos = q

    def open_gripper(self):
        q = self.agent.robot.qpos.clone()
        q[0, -2:] = 2 * T.GRIPPER_OPEN
        self.agent.robot.qpos = q

    def set_qpos(self, qpos):
        self.agent.robot.qpos = torch.tensor([np.asarray(qpos, dtype=np.float32).reshape(-1).tolist()])

    def move_robot(self):
        self.agent.robot.qvel = torch.full_like(self.agent.robot.qvel, 1.0)

    def hold_robot(self):
        self.agent.robot.qvel = torch.zeros_like(self.agent.robot.qvel)


class _WrapperView:
    """Mimics how DemonstrationWrapper calls task_goal: ``self.env.unwrapped`` plus attribute pass-through."""

    def __init__(self, env):
        self.env = SimpleNamespace(unwrapped=env)
        self._inner = env

    def __getattr__(self, name):
        return getattr(self._inner, name)


def goal_text(env) -> list:
    """Get this episode's goal language via the real task_goal.get_language_goal."""
    from robomme.robomme_env.utils import task_goal

    return task_goal.get_language_goal(_WrapperView(env), type(env).__name__)


def find_seed(task: str, difficulty: str, predicate, limit: int = 64) -> int:
    """Find the first seed in the offline world for which predicate(env) holds (runs only _load_scene/_initialize_episode; not a simulation reset)."""
    with OfficialWorld(task) as world:
        for seed in range(limit):
            ep = world.make(difficulty, seed=seed)
            if predicate(ep.env):
                return seed
    raise AssertionError(f"{task}/{difficulty}: no layout satisfying the condition within {limit} seeds")
