"""离线场景（hard 包单元测试公共夹具）：不建 SAPIEN 场景，跑任务类**真实的** ``__init__`` 与 ``_load_scene``。

手法（以旧蓝本 ``test_v5_xhard_pickswing::OfflineScene`` 为基础，推广到 16 个任务）：

* ``BaseEnv.__init__`` 换成只设几个属性的替身（``num_envs``、``device``、``scene``），任务类自己的 ``__init__``
  （取值点、SpecRecorder、难度归一）原样执行——测试不复刻任何取值逻辑；
* ``scene`` 是 :class:`FakeScene`：``create_actor_builder()``／``create_articulation_builder()`` 返回只记录
  碰撞形状与初始位姿的构建器，``build*`` 返回带 float32 批维位姿的 :class:`FakeActor`；``actors.build_cube``、
  ``build_button`` 等**真实**构建函数照常执行，只是最终落到假构建器上；
* ``get_actor_obb`` 换成与 ManiSkill ``get_component_mesh`` 同一条 trimesh 路径（盒子／圆柱 → 局部位姿 → 合并 →
  实体位姿 → ``bounding_box_oriented``），对静态体与无碰撞体照真实行为抛错；
* 各任务模块里的 ``TableSceneBuilder`` 换成空构建器（桌面与机器人不参与布局取值）。

保真度的检验就是包内规格回放：离线回放 ``native_episode_spec`` 时每个取值点的「原抽样」都要与冻结值逐位相等
（``mismatches == 0``），任何一处几何替身与真实不符都会让拒绝采样多抽或少抽而暴露出来。

本文件计划原写在 ``tests/robomme_hard/_support/offline_scene.py``；``_support`` 归主会话，T4 先放在本目录，是否上移由主会话定。
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

from robomme_hard.env_record_wrapper import hard_specs

_MP_UTILS = importlib.import_module("mani_skill.examples.motionplanning.base_motionplanner.utils")
from robomme_hard.robomme_env.utils import object_generation as og

ALL_TASKS = hard_specs.ALL_TASKS
# 运行时四项与评估链一致（取自生产常量，不在测试里另写）
RUNTIME = {k: v for k, v in hard_specs.RUNTIME.items()}


def task_module(task: str):
    return importlib.import_module(f"robomme_hard.robomme_env.{task}")


def task_class(task: str):
    return getattr(task_module(task), task)


# ---------------------------------------------------------------- 位姿与形状


def _as_sapien_pose(pose) -> sapien.Pose:
    if pose is None:
        return sapien.Pose()
    if isinstance(pose, sapien.Pose):
        return pose
    if isinstance(pose, Pose):
        p = pose.p.reshape(-1, 3)[0].detach().cpu().numpy().astype(np.float32)
        q = pose.q.reshape(-1, 4)[0].detach().cpu().numpy().astype(np.float32)
        return sapien.Pose(p=p, q=q)
    raise TypeError(f"不支持的位姿类型 {type(pose).__name__}")


def _batched(pose: sapien.Pose) -> Pose:
    """真实 CPU 仿真里 actor.pose 是 float32、带批维 1 的 ManiSkill Pose。"""
    return Pose.create_from_pq(
        torch.tensor(np.asarray(pose.p, dtype=np.float32).reshape(1, 3)),
        torch.tensor(np.asarray(pose.q, dtype=np.float32).reshape(1, 4)),
    )


class FakeShape:
    """碰撞或可视形状：盒体带 ``half_size``（与 physx／render 盒体同为 float32 数组），其余形状不带该属性。"""

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
        """与 ``mani_skill.utils.geometry.trimesh_utils.get_component_meshes`` 同一套 trimesh 原语。"""
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
    """构建器公共部分：记录碰撞形状与盒体可视形状；其余视觉与物理属性设置一律接受并忽略。"""

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
        # 其余 add_*_visual 与 set_* 物理／渲染属性：接受并忽略；未知的碰撞形状一律拒绝（防止静默丢形状）
        if name.startswith(("add_", "set_")) and "collision" not in name:
            return lambda *a, **k: None
        raise AttributeError(f"{type(self).__name__} 没有替身方法 {name}")


# 刚体种类 → 真实 physx 组件类（find_component_by_type 用 issubclass 判定，与 sapien 一致）
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
        raise AttributeError(f"FakeJointDesc 没有替身方法 {name}")


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
    """actor／link 替身：名字、float32 批维位姿、刚体种类、碰撞与可视形状，以及 ``_objs[0]`` 实体接口。"""

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
        # ManiSkill：Actor._objs 是 sapien.Entity；Link._objs 是 PhysxArticulationLinkComponent
        self._objs = [self._component] if body == "link" else [self._entity]
        # ManiSkill Actor.initial_pose：构建器给定的初始位姿（Pose.create 转成批维）
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
                # 关节零位：child = parent · pose_in_parent · pose_in_child⁻¹
                pose = parent._pose * lb.pose_in_parent * lb.pose_in_child.inv()
            joint = FakeJointDesc(lb.joint_name or "", lb.pose_in_parent, lb.pose_in_child)
            link = FakeActor(lb.name, pose, "link", lb.shapes, lb.visuals, joint=joint, parent=parent)
            made[id(lb)] = link
            self.links.append(link)
            if lb.joint_name is not None:
                self.joints.append(joint)
        # 自由度 = 非固定关节数（按钮的移动副 1 个；杆的两段是固定关节，0 个）
        self.dof = sum(1 for lb in link_builders if lb.joint_name is not None and getattr(lb, "joint_type", None) != "fixed")
        self._fake_qpos = [0.0] * len(self.joints)

    @property
    def pose(self) -> Pose:
        return _batched(self._pose)

    @pose.setter
    def pose(self, value):
        self.set_pose(value)

    def set_pose(self, value):
        """根位姿改动时各 link 随根刚性平移旋转（关节保持零位）。"""
        new = _as_sapien_pose(value)
        delta = new * self._pose.inv()
        for link in self.links:
            link._pose = delta * link._pose
        self._pose = new

    def get_pose(self):
        return self._pose

    def get_qpos(self):
        """关节位置（批维 1）；按钮按下深度 = -qpos，由真值表的世界替身经 :meth:`set_qpos` 改写。"""
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
        raise AttributeError(f"FakeArticulation 没有替身方法 {name}")


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
        raise AttributeError(f"FakeArticulationBuilder 没有替身方法 {name}")


class FakeScene:
    """只够 ``_load_scene`` 用的场景替身；记录建出的 actor 与 articulation。"""

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
            raise KeyError(f"{name}: {len(hits)} 个")
        return hits[0]


def fake_get_actor_obb(actor, to_world_frame=True, vis=False):
    """与真实 ``get_actor_obb`` 同路径：取实体上的 ``PhysxRigidDynamicComponent``（静态体取不到 → 与真实一样在
    ``get_component_meshes(None)`` 处抛 AttributeError），合并碰撞网格（无形状 → ``assert mesh is not None`` 失败），
    乘实体位姿后取 ``bounding_box_oriented``。"""
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


TCP_UP_Z = 0.25  # TCP 初始高度（远离桌面与全部判定阈值）


def _pose_at(xyz, q=(1.0, 0.0, 0.0, 0.0)) -> Pose:
    return Pose.create_from_pq(torch.tensor([list(map(float, xyz))], dtype=torch.float32),
                               torch.tensor([list(map(float, q))], dtype=torch.float32))


#: 机械臂基座位姿（ManiSkill TableSceneBuilder.initialize 把 Panda 放在 x=-0.615 处；InsertPeg 读它定朝向）
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
    """``sapien.render.RenderMaterial`` 替身：真实构造器要起渲染上下文（实测每次约 0.6 s），布局取值不读材质。"""

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        if name.startswith("set_"):
            return lambda *a, **k: None
        raise AttributeError(f"FakeRenderMaterial 没有替身方法 {name}")


class _FakeTableSceneBuilder:
    def __init__(self, env=None, robot_init_qpos_noise=0, **k):
        self.env = env

    def build(self, *a, **k):
        return None

    def initialize(self, *a, **k):
        return None


def _fake_base_init(self, *args, **kwargs):
    """``BaseEnv.__init__`` 替身：不建仿真，只设 ``_load_scene`` 需要的几个属性。"""
    self.num_envs = 1
    self.device = torch.device("cpu")
    self._sim_device = self.device
    self.robot_uids = kwargs.get("robot_uids")
    self._fake_base_kwargs = dict(kwargs)
    self.scene = FakeScene()
    # 真实 BaseEnv 在 _load_scene 之前先 _load_agent；原三档有的任务在建场时就读 agent（如 PickHighlight 的按钮失败判据）
    self.agent = FakeAgent()


def _modules_with(name: str):
    mods = [og]
    for task in ALL_TASKS:
        mods.append(task_module(task))
    for extra in ("unmask_distractors", "unmask_swap_xhard", "xhard_home_site", "bin_collision",
                  "unmask_distractor_sampler"):
        mods.append(importlib.import_module(f"robomme_hard.robomme_env.utils.{extra}"))
    return [m for m in mods if hasattr(m, name)]


@contextlib.contextmanager
def offline_scene():
    """装上／卸下离线替身（进程内、可恢复；不落盘）。"""
    saved: list[tuple[Any, str, Any]] = []

    def patch(obj, name, value):
        saved.append((obj, name, obj.__dict__[name] if name in obj.__dict__ else getattr(obj, name)))
        setattr(obj, name, value)

    try:
        patch(BaseEnv, "__init__", _fake_base_init)
        patch(sapien.render, "RenderMaterial", FakeRenderMaterial)
        # 被测代码有的在函数体内 ``from ...base_motionplanner.utils import get_actor_obb``，源模块也要换
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
    """按评估链的参数（runtime 四项 + seed + difficulty + sampling_config + native_episode_spec）实例化任务类，
    再跑一次真实 ``_load_scene``。必须在 :func:`offline_scene` 内调用。"""
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


# ---------------------------------------------------------------- 包内规格


_SPECS_CACHE: dict[str, tuple[dict, list[dict]]] = {}


def _load(tier: str) -> tuple[dict, list[dict]]:
    if tier not in _SPECS_CACHE:
        _SPECS_CACHE[tier] = hard_specs.load_specs(hard_specs.packaged_specs_path(tier), check_fingerprint=False)
    return _SPECS_CACHE[tier]


def packaged(tier: str) -> tuple[dict, list[dict]]:
    """包内某档规格（经生产 ``load_specs`` 全量校验）；同进程缓存，调用方拿到的是深拷贝。"""
    header, rows = _load(tier)
    return copy.deepcopy(header), copy.deepcopy(rows)


def delivered_cells() -> list[tuple[str, str]]:
    """V9 实际交付的 (任务, 档)，取自生产常量 ``V9_CELLS``。"""
    return sorted(hard_specs.V9_CELLS)


def tiers_of(task: str) -> list[str]:
    return [tier for (t, tier) in delivered_cells() if t == task]


def cells_of(*tasks: str) -> list[tuple[str, str]]:
    return [(t, tier) for (t, tier) in delivered_cells() if t in tasks]


def delivered_rows(task: str, tier: str, n: int | None = None) -> tuple[dict, list[dict]]:
    """包内该格的正式局（``delivered``），按 candidate 升序（与 builder 的排序一致）；``n`` 取前 n 行（深拷贝）。"""
    header, rows = _load(tier)
    chosen = sorted((r for r in rows if r["task"] == task and hard_specs.delivered(r)),
                    key=lambda r: int(r["candidate"]))
    chosen = chosen if n is None else chosen[:n]
    return copy.deepcopy(header), copy.deepcopy(chosen)


def replay_row(task: str, tier: str, header: dict, row: dict):
    """把包内一行规格按评估链参数回放到离线 ``_load_scene``，返回 env。"""
    return run_offline(task, seed=row["seed"], difficulty=tier, sampling_config=header["sampling_config"][task],
                       spec=row["spec"])


def export_spec(task: str, tier: str, seed: int, header: dict | None = None):
    """导出模式跑一次离线 ``_load_scene``；返回 (env, spec 文档)。"""
    if header is None:
        header, _ = packaged(tier)
    env = run_offline(task, seed=seed, difficulty=tier, sampling_config=header["sampling_config"][task])
    return env, env._spec.to_dict()
