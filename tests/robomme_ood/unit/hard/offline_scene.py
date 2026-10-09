"""Offline scene (shared fixture for hard-package unit tests): builds no SAPIEN scene, runs the task class's **real** ``__init__`` and ``_load_scene``.

Technique (based on the old blueprint ``test_v5_xhard_pickswing::OfflineScene``, generalized to 16 tasks):

* ``BaseEnv.__init__`` is replaced with a stand-in that only sets a few attributes (``num_envs``, ``device``, ``scene``); the task class's own ``__init__``
  (value sites, SpecRecorder, difficulty normalization) runs unchanged; the tests replicate no value logic;
* ``scene`` is a :class:`FakeScene`: ``create_actor_builder()``/``create_articulation_builder()`` return builders that only record
  collision shapes and initial poses, ``build*`` returns a :class:`FakeActor` with float32 batched poses; **real** builder functions such as ``actors.build_cube``,
  ``build_button`` run as usual, they just end up on the fake builders;
* ``get_actor_obb`` is replaced with the same trimesh path as ManiSkill ``get_component_mesh`` (box/cylinder → local pose → merge →
  entity pose → ``bounding_box_oriented``), raising for static and collision-free bodies just like the real behavior;
* ``TableSceneBuilder`` in each task module is replaced with an empty builder (table and robot do not take part in layout values).

Fidelity is checked by packaged spec replay: when ``native_episode_spec`` is replayed offline, the "original draw" at every value site must equal the frozen value bit for bit
(``mismatches == 0``); any geometry stand-in that differs from reality would make rejection sampling draw more or fewer times and be exposed.

The plan originally put this file at ``tests/robomme_ood/_support/offline_scene.py``; ``_support`` belongs to the main session, so T4 keeps it in this directory for now, and the main session decides whether to move it up.
"""
from __future__ import annotations

import contextlib
import copy
import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh
import trimesh.creation

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import Pose

from robomme_ood.env_record_wrapper import hard_specs

_MP_UTILS = importlib.import_module("mani_skill.examples.motionplanning.base_motionplanner.utils")
from robomme_ood.robomme_env.utils import object_generation as og

ALL_TASKS = hard_specs.ALL_TASKS
# The four runtime items match the evaluation chain (taken from production constants, not rewritten in tests)
RUNTIME = {k: v for k, v in hard_specs.RUNTIME.items()}


def task_module(task: str):
    return importlib.import_module(f"robomme_ood.robomme_env.{task}")


def task_class(task: str):
    return getattr(task_module(task), task)


# ---------------------------------------------------------------- poses and shapes


def _as_sapien_pose(pose) -> sapien.Pose:
    if pose is None:
        return sapien.Pose()
    if isinstance(pose, sapien.Pose):
        return pose
    if isinstance(pose, Pose):
        p = pose.p.reshape(-1, 3)[0].detach().cpu().numpy().astype(np.float32)
        q = pose.q.reshape(-1, 4)[0].detach().cpu().numpy().astype(np.float32)
        return sapien.Pose(p=p, q=q)
    raise TypeError(f"unsupported pose type {type(pose).__name__}")


def _batched(pose: sapien.Pose) -> Pose:
    """In real CPU simulation actor.pose is a float32 ManiSkill Pose with batch dim 1."""
    return Pose.create_from_pq(
        torch.tensor(np.asarray(pose.p, dtype=np.float32).reshape(1, 3)),
        torch.tensor(np.asarray(pose.q, dtype=np.float32).reshape(1, 4)),
    )


class FakeShape:
    """Collision or visual shape: boxes carry ``half_size`` (a float32 array like physx/render boxes), other shapes do not have this attribute."""

    def __init__(self, kind: str, local_pose, **dims):
        self.kind = kind
        self.local_pose = _as_sapien_pose(local_pose)
        self.dims = dims
        if kind == "box":
            self.half_size = np.asarray(dims["half_size"], dtype=np.float32).reshape(3)
        if "radius" in dims:
            self.radius = float(dims["radius"])
        if "half_length" in dims:
            self.half_length = float(dims["half_length"])

    def get_local_pose(self):
        return self.local_pose

    def mesh(self) -> trimesh.Trimesh:
        """The same trimesh primitives as ``mani_skill.utils.geometry.trimesh_utils.get_component_meshes``."""
        if self.kind == "box":
            m = trimesh.creation.box(extents=2 * self.half_size)
        elif self.kind == "cylinder":
            m = trimesh.creation.cylinder(radius=self.radius, height=2 * self.half_length)
        elif self.kind == "capsule":
            m = trimesh.creation.capsule(height=2 * self.half_length, radius=self.radius)
        elif self.kind == "sphere":
            m = trimesh.creation.icosphere(radius=self.radius)
        else:  # pragma: no cover
            raise TypeError(self.kind)
        m.apply_transform(self.local_pose.to_transformation_matrix())
        return m


class _ShapeRecorder:
    """Common builder part: records collision shapes and box visual shapes; all other visual and physical property setters are accepted and ignored."""

    def _init_shapes(self):
        self.shapes: list[FakeShape] = []
        self.visuals: list[FakeShape] = []
        self.name = None

    def add_box_collision(self, pose=None, half_size=(1, 1, 1), *a, **k):
        self.shapes.append(FakeShape("box", pose, half_size=[float(x) for x in half_size]))

    def add_cylinder_collision(self, pose=None, radius=1.0, half_length=1.0, *a, **k):
        self.shapes.append(FakeShape("cylinder", pose, radius=float(radius), half_length=float(half_length)))

    def add_capsule_collision(self, pose=None, radius=1.0, half_length=1.0, *a, **k):
        self.shapes.append(FakeShape("capsule", pose, radius=float(radius), half_length=float(half_length)))

    def add_sphere_collision(self, pose=None, radius=1.0, *a, **k):
        self.shapes.append(FakeShape("sphere", pose, radius=float(radius)))

    def add_box_visual(self, pose=None, half_size=(1, 1, 1), *a, **k):
        self.visuals.append(FakeShape("box", pose, half_size=[float(x) for x in half_size]))

    def set_name(self, name):
        self.name = name

    def __getattr__(self, name):
        # Other add_*_visual and set_* physics/render properties: accepted and ignored; unknown collision shapes are always rejected (to prevent silently dropping shapes)
        if name.startswith(("add_", "set_")) and "collision" not in name:
            return lambda *a, **k: None
        raise AttributeError(f"{type(self).__name__} has no stand-in method {name}")


# Rigid body kind → real physx component class (find_component_by_type uses issubclass, same as sapien)
_COMPONENT_CLASS = {
    "dynamic": physx.PhysxRigidDynamicComponent,
    "kinematic": physx.PhysxRigidDynamicComponent,
    "static": physx.PhysxRigidStaticComponent,
    "link": physx.PhysxArticulationLinkComponent,
}


class FakeJointDesc:
    def __init__(self, name, pose_in_parent: sapien.Pose, pose_in_child: sapien.Pose):
        self.name = name
        self._pip, self._pic = pose_in_parent, pose_in_child

    def get_name(self):
        return self.name

    def get_pose_in_parent(self):
        return self._pip

    def get_pose_in_child(self):
        return self._pic

    def __getattr__(self, name):
        if name.startswith("set_"):
            return lambda *a, **k: None
        raise AttributeError(f"FakeJointDesc has no stand-in method {name}")


class FakeRigidComponent:
    def __init__(self, owner: "FakeActor"):
        self.owner = owner

    @property
    def pose(self) -> sapien.Pose:
        return self.owner._pose

    def get_collision_shapes(self):
        return list(self.owner._fake_shapes)

    def get_entity(self):
        return self.owner._entity

    def get_joint(self):
        return self.owner._fake_joint

    def get_parent(self):
        parent = self.owner._fake_parent
        return None if parent is None else parent._component


class FakeRenderBody:
    def __init__(self, visuals):
        self.render_shapes = list(visuals)


class FakeEntity:
    def __init__(self, owner: "FakeActor"):
        self.owner = owner
        self.name = owner.name

    def find_component_by_type(self, cls):
        real = _COMPONENT_CLASS[self.owner.px_body_type]
        return self.owner._component if issubclass(real, cls) else None

    def get_components(self):
        return [self.owner._component, FakeRenderBody(self.owner._fake_visuals)]

    @property
    def pose(self):
        return self.owner._pose


class FakeActor:
    """actor/link stand-in: name, float32 batched pose, rigid body kind, collision and visual shapes, and the ``_objs[0]`` entity interface."""

    def __init__(self, name, pose: sapien.Pose, body: str, shapes, visuals=(), initial_pose=None,
                 joint=None, parent=None):
        self.name = name
        self._pose = pose
        self.px_body_type = body
        self._fake_shapes = list(shapes)
        self._fake_visuals = list(visuals)
        self._fake_joint = joint
        self._fake_parent = parent
        self._component = FakeRigidComponent(self)
        self._entity = FakeEntity(self)
        # ManiSkill: Actor._objs is sapien.Entity; Link._objs is PhysxArticulationLinkComponent
        self._objs = [self._component] if body == "link" else [self._entity]
        # ManiSkill Actor.initial_pose: initial pose given by the builder (Pose.create converts to batched)
        self.initial_pose = None if initial_pose is None else Pose.create(initial_pose)

    @property
    def pose(self) -> Pose:
        return _batched(self._pose)

    @pose.setter
    def pose(self, value):
        self._pose = _as_sapien_pose(value)

    def set_pose(self, value):
        self.pose = value

    def get_pose(self):
        return self._pose

    def get_name(self):
        return self.name

    def set_linear_velocity(self, *a):
        pass

    def set_angular_velocity(self, *a):
        pass

    def __repr__(self):
        return f"FakeActor({self.name!r})"


class FakeActorBuilder(_ShapeRecorder):
    def __init__(self, scene):
        self._init_shapes()
        self.scene = scene
        self.initial_pose = None

    def set_initial_pose(self, pose):
        self.initial_pose = pose

    def _build(self, name, body):
        init = self.initial_pose if self.initial_pose is not None else sapien.Pose()
        actor = FakeActor(name if name is not None else self.name, _as_sapien_pose(init), body,
                          self.shapes, self.visuals, initial_pose=init)
        self.scene.actors.append(actor)
        return actor

    def build(self, name=None, *a, **k):
        return self._build(name, "dynamic")

    def build_dynamic(self, name=None, *a, **k):
        return self._build(name, "dynamic")

    def build_kinematic(self, name=None, *a, **k):
        return self._build(name, "kinematic")

    def build_static(self, name=None, *a, **k):
        return self._build(name, "static")


class FakeLinkBuilder(_ShapeRecorder):
    def __init__(self, parent):
        self._init_shapes()
        self.parent = parent
        self.joint_name = None
        self.pose_in_parent = sapien.Pose()
        self.pose_in_child = sapien.Pose()

    def set_joint_name(self, name):
        self.joint_name = name

    def set_joint_properties(self, type=None, limits=None, pose_in_parent=None, pose_in_child=None, **k):
        self.joint_type = type
        self.pose_in_parent = _as_sapien_pose(pose_in_parent)
        self.pose_in_child = _as_sapien_pose(pose_in_child)


class FakeArticulation:
    def __init__(self, name, root_pose: sapien.Pose, link_builders: list[FakeLinkBuilder]):
        self.name = name
        self._pose = root_pose
        made: dict[int, FakeActor] = {}
        self.links: list[FakeActor] = []
        self.joints: list[FakeJointDesc] = []
        for lb in link_builders:
            parent = None if lb.parent is None else made[id(lb.parent)]
            if parent is None:
                pose = root_pose
            else:
                # Joint zero position: child = parent · pose_in_parent · pose_in_child⁻¹
                pose = parent._pose * lb.pose_in_parent * lb.pose_in_child.inv()
            joint = FakeJointDesc(lb.joint_name or "", lb.pose_in_parent, lb.pose_in_child)
            link = FakeActor(lb.name, pose, "link", lb.shapes, lb.visuals, joint=joint, parent=parent)
            made[id(lb)] = link
            self.links.append(link)
            if lb.joint_name is not None:
                self.joints.append(joint)
        # Degrees of freedom = number of non-fixed joints (the button's prismatic joint is 1; the stick's two segments are fixed joints, 0)
        self.dof = sum(1 for lb in link_builders if lb.joint_name is not None and getattr(lb, "joint_type", None) != "fixed")
        self._fake_qpos = [0.0] * len(self.joints)

    @property
    def pose(self) -> Pose:
        return _batched(self._pose)

    @pose.setter
    def pose(self, value):
        self.set_pose(value)

    def set_pose(self, value):
        """When the root pose changes, each link translates and rotates rigidly with the root (joints stay at zero)."""
        new = _as_sapien_pose(value)
        delta = new * self._pose.inv()
        for link in self.links:
            link._pose = delta * link._pose
        self._pose = new

    def get_pose(self):
        return self._pose

    def get_qpos(self):
        """Joint positions (batch dim 1); button press depth = -qpos, rewritten via :meth:`set_qpos` by the truth-table world stand-in."""
        return torch.tensor([self._fake_qpos], dtype=torch.float32)

    @property
    def qpos(self):
        return self.get_qpos()

    def set_qpos(self, qpos):
        self._fake_qpos = [float(x) for x in torch.as_tensor(qpos, dtype=torch.float32).reshape(-1)]

    def get_links(self):
        return list(self.links)

    def get_joints(self):
        return list(self.joints)

    def get_active_joints(self):
        return list(self.joints)

    def get_name(self):
        return self.name

    def __getattr__(self, name):
        if name.startswith("set_"):
            return lambda *a, **k: None
        raise AttributeError(f"FakeArticulation has no stand-in method {name}")


class FakeArticulationBuilder:
    def __init__(self, scene):
        self.scene = scene
        self.initial_pose = None
        self.link_builders: list[FakeLinkBuilder] = []

    def set_initial_pose(self, pose):
        self.initial_pose = pose

    def create_link_builder(self, parent=None):
        lb = FakeLinkBuilder(parent)
        self.link_builders.append(lb)
        return lb

    def build(self, name=None, fix_root_link=None, *a, **k):
        art = FakeArticulation(name, _as_sapien_pose(self.initial_pose), self.link_builders)
        self.scene.articulations.append(art)
        return art

    def __getattr__(self, name):
        if name.startswith("set_"):
            return lambda *a, **k: None
        raise AttributeError(f"FakeArticulationBuilder has no stand-in method {name}")


class FakeScene:
    """Scene stand-in just sufficient for ``_load_scene``; records the actors and articulations built."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.gpu_sim_enabled = False
        self.actors: list[FakeActor] = []
        self.articulations: list[FakeArticulation] = []

    def create_actor_builder(self):
        return FakeActorBuilder(self)

    def create_articulation_builder(self):
        return FakeArticulationBuilder(self)

    def actor_by_name(self, name):
        hits = [a for a in self.actors if a.name == name]
        if len(hits) != 1:
            raise KeyError(f"{name}: {len(hits)} found")
        return hits[0]


def fake_get_actor_obb(actor, to_world_frame=True, vis=False):
    """Same path as the real ``get_actor_obb``: take the entity's ``PhysxRigidDynamicComponent`` (static bodies have none → raises AttributeError at
    ``get_component_meshes(None)`` just like reality), merge collision meshes (no shapes → ``assert mesh is not None`` fails),
    apply the entity pose and take ``bounding_box_oriented``."""
    comp = actor._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent)
    if comp is None:
        raise AttributeError("'NoneType' object has no attribute 'get_collision_shapes'")
    meshes = [s.mesh() for s in comp.get_collision_shapes()]
    assert meshes, f"can not get actor mesh for {actor}"
    vs, fs, n = [], [], 0
    for m in meshes:
        vs.append(m.vertices)
        fs.append(m.faces + n)
        n += m.vertices.shape[0]
    mesh = trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    if to_world_frame:
        mesh.apply_transform(comp.pose.to_transformation_matrix())
    return mesh.bounding_box_oriented


TCP_UP_Z = 0.25  # initial TCP height (far from the table and all verdict thresholds)


def _pose_at(xyz, q=(1.0, 0.0, 0.0, 0.0)) -> Pose:
    return Pose.create_from_pq(torch.tensor([list(map(float, xyz))], dtype=torch.float32),
                               torch.tensor([list(map(float, q))], dtype=torch.float32))


#: Arm base pose (ManiSkill TableSceneBuilder.initialize places Panda at x=-0.615; InsertPeg reads it to set orientation)
ROBOT_BASE_XYZ = (-0.615, 0.0, 0.0)


class FakeRobot:
    def __init__(self):
        self.pose = _pose_at(ROBOT_BASE_XYZ)
        self.qpos = torch.zeros((1, 9), dtype=torch.float32)
        self.qvel = torch.zeros((1, 9), dtype=torch.float32)

    def get_qpos(self):
        return self.qpos

    def get_qvel(self):
        return self.qvel

    def set_qpos(self, qpos):
        self.qpos = torch.as_tensor(qpos, dtype=torch.float32).reshape(1, -1)

    def set_qvel(self, qvel):
        self.qvel = torch.as_tensor(qvel, dtype=torch.float32).reshape(1, -1)


class FakeAgent:
    def __init__(self):
        self.robot = FakeRobot()
        self.tcp = SimpleNamespace(pose=_pose_at((0.0, 0.0, TCP_UP_Z)))
        self.held = None

    @property
    def tcp_pose(self):
        return self.tcp.pose

    def reset(self, qpos=None):
        if qpos is not None:
            self.robot.set_qpos(qpos)

    def is_grasping(self, obj, *a, **k):
        return torch.tensor([obj is self.held])


class FakeRenderMaterial:
    """Stand-in for ``sapien.render.RenderMaterial``: the real constructor starts a render context (measured about 0.6 s each), and layout values do not read materials."""

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        if name.startswith("set_"):
            return lambda *a, **k: None
        raise AttributeError(f"FakeRenderMaterial has no stand-in method {name}")


class _FakeTableSceneBuilder:
    def __init__(self, env=None, robot_init_qpos_noise=0, **k):
        self.env = env

    def build(self, *a, **k):
        return None

    def initialize(self, *a, **k):
        return None


def _fake_base_init(self, *args, **kwargs):
    """Stand-in for ``BaseEnv.__init__``: builds no simulation, only sets the few attributes ``_load_scene`` needs."""
    self.num_envs = 1
    self.device = torch.device("cpu")
    self._sim_device = self.device
    self.robot_uids = kwargs.get("robot_uids")
    self._fake_base_kwargs = dict(kwargs)
    self.scene = FakeScene()
    # Real BaseEnv runs _load_agent before _load_scene; some original-tier tasks read agent while building the scene (e.g. PickHighlight's button failure criterion)
    self.agent = FakeAgent()


def _modules_with(name: str):
    mods = [og]
    for task in ALL_TASKS:
        mods.append(task_module(task))
    for extra in ("unmask_distractors", "unmask_swap_xhard", "xhard_home_site", "bin_collision",
                  "unmask_distractor_sampler"):
        mods.append(importlib.import_module(f"robomme_ood.robomme_env.utils.{extra}"))
    return [m for m in mods if hasattr(m, name)]


@contextlib.contextmanager
def offline_scene():
    """Install/uninstall the offline stand-ins (in-process, reversible; not written to disk)."""
    saved: list[tuple[Any, str, Any]] = []

    def patch(obj, name, value):
        saved.append((obj, name, obj.__dict__[name] if name in obj.__dict__ else getattr(obj, name)))
        setattr(obj, name, value)

    try:
        patch(BaseEnv, "__init__", _fake_base_init)
        patch(sapien.render, "RenderMaterial", FakeRenderMaterial)
        # Some code under test does ``from ...base_motionplanner.utils import get_actor_obb`` inside function bodies, so the source module must be replaced too
        for mod in [_MP_UTILS, *_modules_with("get_actor_obb")]:
            patch(mod, "get_actor_obb", fake_get_actor_obb)
        for mod in _modules_with("TableSceneBuilder"):
            patch(mod, "TableSceneBuilder", _FakeTableSceneBuilder)
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


def make_offline(task: str, *, seed: int, difficulty: str, sampling_config=None, spec=None, load: bool = True,
                 **extra):
    """Instantiate the task class with the evaluation chain's parameters (four runtime items + seed + difficulty + sampling_config + native_episode_spec),
    then run the real ``_load_scene`` once. Must be called inside :func:`offline_scene`."""
    cls = task_class(task)
    kwargs = dict(RUNTIME)
    kwargs.update(seed=int(seed), difficulty=difficulty)
    if sampling_config is not None:
        kwargs["sampling_config"] = sampling_config
    if spec is not None:
        kwargs["native_episode_spec"] = spec
    kwargs.update(extra)
    env = cls(**kwargs)
    if load:
        env._load_scene({})
    return env


def run_offline(task: str, **kw):
    with offline_scene():
        return make_offline(task, **kw)


# ---------------------------------------------------------------- packaged specs


_SPECS_CACHE: dict[str, tuple[dict, list[dict]]] = {}


def _load(tier: str) -> tuple[dict, list[dict]]:
    if tier not in _SPECS_CACHE:
        _SPECS_CACHE[tier] = hard_specs.load_specs(hard_specs.packaged_specs_path(tier), check_fingerprint=False)
    return _SPECS_CACHE[tier]


def packaged(tier: str) -> tuple[dict, list[dict]]:
    """Packaged specs of a tier (fully validated by production ``load_specs``); cached in-process, callers get a deep copy."""
    header, rows = _load(tier)
    return copy.deepcopy(header), copy.deepcopy(rows)


def delivered_cells() -> list[tuple[str, str]]:
    """(task, tier) pairs actually delivered in V9, taken from the production constant ``V9_CELLS``."""
    return sorted(hard_specs.V9_CELLS)


def tiers_of(task: str) -> list[str]:
    return [tier for (t, tier) in delivered_cells() if t == task]


def cells_of(*tasks: str) -> list[tuple[str, str]]:
    return [(t, tier) for (t, tier) in delivered_cells() if t in tasks]


def delivered_rows(task: str, tier: str, n: int | None = None) -> tuple[dict, list[dict]]:
    """Formal episodes (``delivered``) of this packaged cell, in ascending candidate order (same as the builder's ordering); ``n`` takes the first n rows (deep copy)."""
    header, rows = _load(tier)
    chosen = sorted((r for r in rows if r["task"] == task and hard_specs.delivered(r)),
                    key=lambda r: int(r["candidate"]))
    chosen = chosen if n is None else chosen[:n]
    return copy.deepcopy(header), copy.deepcopy(chosen)


def replay_row(task: str, tier: str, header: dict, row: dict):
    """Replay a packaged spec row into the offline ``_load_scene`` with the evaluation chain's parameters; return env."""
    return run_offline(task, seed=row["seed"], difficulty=tier, sampling_config=header["sampling_config"][task],
                       spec=row["spec"])


def export_spec(task: str, tier: str, seed: int, header: dict | None = None):
    """Run the offline ``_load_scene`` once in export mode; return (env, spec document)."""
    if header is None:
        header, _ = packaged(tier)
    env = run_offline(task, seed=seed, difficulty=tier, sampling_config=header["sampling_config"][task])
    return env, env._spec.to_dict()
