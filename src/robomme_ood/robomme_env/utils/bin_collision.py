"""容器／方块之间的真实碰撞盒判据（NEW_VALUE_INJECTION_TEST_PLAN 第三节与第 8.1～8.3 节）。

本模块只做几何，不建 actor、不采样、不碰 SAPIEN 场景，因此外部 CPU 规格生成器与仿真
运行时可以共用同一份判据，避免「筛查用一套几何、实跑用另一套」的双真值。

三层接口：

* :func:`bin_shape_specs` / :func:`cube_shape_specs` —— 与
  ``object_generation.py::build_bin`` 同源的局部盒体描述（容器 6 个盒体、方块 1 个）。
* :func:`check_bin_layout` —— 静态判据：全部对象对逐盒对做三维分离轴检测，
  ``g > ε`` 才算分开，``0 ≤ g ≤ ε`` 记接触、``g < 0`` 记穿入，两者都排除。
* :func:`check_swap_sweep` —— 连续判据：复刻
  ``statechange.py::swap_flat_two_lane`` 的弯道与四元数插值，用区间二分证明整段路径
  始终分开；证明不出来一律按 ``uncertified`` 排除，不改成抽帧放行。
* :func:`check_multi_swap_sweep` —— V5 新增：同一窗口多对同时交换的联合连续判据，复用同一个
  区间二分证明；带默认开启的认证预筛（401 个采样点 + Lipschitz 界，只跳过已证明分离的对）。
  :func:`check_swap_sweep_prefiltered` 是它的单对包装；:func:`static_box_state` 等 helper 构造
  任意有向盒体作静止障碍（按钮底座、干扰容器等）。:func:`check_swap_sweep` 本身不预筛、判定不变。

判据数值全部定死在模块常量里：``EPS_M``、``MAX_DEPTH``、``MAX_INTERVALS``、
``DEGENERATE_NORM``。不提供放宽阈值的开关。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "BinCollisionError",
    "SpecBindingError",
    "nearest_partner_index",
    "CollisionRejection",
    "ShapeSpec",
    "ObjectState",
    "EPS_M",
    "MAX_DEPTH",
    "MAX_INTERVALS",
    "DEGENERATE_NORM",
    "LANE_OFFSET",
    "bin_shape_specs",
    "cube_shape_specs",
    "bin_actor_pose",
    "cube_actor_pose",
    "quat_to_matrix",
    "euler_xyz_to_quat",
    "sat_gap",
    "check_pair_static",
    "check_bin_layout",
    "check_swap_sweep",
    "PREFILTER_SAMPLES",
    "PREFILTER_MARGIN_M",
    "check_multi_swap_sweep",
    "check_swap_sweep_prefiltered",
    "static_box_state",
    "static_rect_state",
    "static_state_from_obb2d",
    "button_base_state",
    "check_bin_state",
    "shape_specs_from_actor",
    "object_state_from_actor",
]


# ── 固定判据数值 ─────────────────────────────────────────────────────────────
#: 分离阈值，米。1e-6 米 = 0.001 毫米；不再额外加安全间隙。
EPS_M = 1e-6
#: 单个盒对、单段路径的区间二分最大深度。
MAX_DEPTH = 20
#: 单个盒对、单段路径最多考察的区间个数。
MAX_INTERVALS = 4096
#: 四元数线性混合的范数下限；低于它认为插值退化，无法给出角速度上界。
DEGENERATE_NORM = 1e-6
#: ``swap_flat_two_lane`` 在两个视频任务里的实际弯道侧移，米。
LANE_OFFSET = 0.07
#: ``swap_flat_two_lane`` 判定「起终点重合、法向取零」的阈值，与原函数一致。
COINCIDENT_XY = 1e-9


class BinCollisionError(RuntimeError):
    """规格或运行状态未通过碰撞判据。``rejection`` 里带结构化的拒绝证据。"""

    def __init__(self, rejection: "CollisionRejection") -> None:
        super().__init__(rejection.summary())
        self.rejection = rejection


class SpecBindingError(RuntimeError):
    """运行时实测的对象／动作与规格预写的不一致，对应七类结果里的「实际对象／动作不符」。

    最典型的是交换搭档：规格预写的 ``partner`` 是设计口径（按前一段收尾后的名义位姿算），
    而 ``step`` 在交换开始那一刻按**实际**位姿重算最近邻。计划第五节步骤 0a 明确要求
    「不符即失败，禁止换搭档」——所以这里抛错中止该样本，绝不现场改用实际搭档继续跑。

    ``detail`` 里带完整的候选距离表，便于事后判断是临界等距还是真的错绑。
    """

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


@dataclass(frozen=True)
class CollisionRejection:
    """一次拒绝的完整证据。``reason`` 只取三种具名值。

    * ``contact``：``g < 0``，两个盒体已经穿入。
    * ``numerical_boundary``：``0 ≤ g ≤ ε``（含正好相切的 ``g == 0``），或判定值非有限。
    * ``uncertified``：深度／区间数耗尽、四元数退化、上界算不出来——无法证明安全。
    """

    reason: str
    stage: str  # "initial" / "sweep" / "state"
    object_a: str
    object_b: str
    shape_a: int
    shape_b: int
    gap_m: float | None = None
    s: float | None = None
    sweep_index: int | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    depth: int | None = None
    intervals_used: int | None = None
    detail: str = ""

    def summary(self) -> str:
        where = f"{self.stage}"
        if self.sweep_index is not None:
            where += f"#{self.sweep_index}"
        gap = "n/a" if self.gap_m is None else f"{self.gap_m:.12g}"
        return (
            f"碰撞排除[{self.reason}] {where} 对象 {self.object_a}(形状{self.shape_a})"
            f" 与 {self.object_b}(形状{self.shape_b}) g={gap}"
            + (f" s={self.s:.12g}" if self.s is not None else "")
            + (f" {self.detail}" if self.detail else "")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "stage": self.stage,
            "object_a": self.object_a,
            "object_b": self.object_b,
            "shape_a": self.shape_a,
            "shape_b": self.shape_b,
            "gap_m": self.gap_m,
            "s": self.s,
            "sweep_index": self.sweep_index,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "depth": self.depth,
            "intervals_used": self.intervals_used,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ShapeSpec:
    """一个盒体在 actor 局部坐标系里的位姿与半尺寸。"""

    local_p: np.ndarray  # (3,)
    local_q: np.ndarray  # (4,) wxyz
    half: np.ndarray  # (3,)

    def vertex_radius(self) -> float:
        """该盒体 8 个顶点到 actor 原点的最大距离，用于旋转项的线速度上界。"""
        rot = quat_to_matrix(self.local_q)
        best = 0.0
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    offset = rot @ (self.half * np.array([sx, sy, sz]))
                    best = max(best, float(np.linalg.norm(self.local_p + offset)))
        return best


@dataclass
class ObjectState:
    """场上一个对象的当前世界位姿与它的全部盒体。"""

    name: str
    p: np.ndarray  # (3,)
    q: np.ndarray  # (4,) wxyz
    shapes: Sequence[ShapeSpec]
    radii: tuple[float, ...] = field(default=())

    def __post_init__(self) -> None:
        self.p = np.asarray(self.p, dtype=np.float64).reshape(3)
        self.q = _normalize_quat(np.asarray(self.q, dtype=np.float64).reshape(4))
        if not self.radii:
            self.radii = tuple(shape.vertex_radius() for shape in self.shapes)


# ── 四元数与旋转 ─────────────────────────────────────────────────────────────
def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= DEGENERATE_NORM:
        raise ValueError(f"四元数退化，范数 {norm}")
    return q / norm


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """wxyz 四元数转 3×3 旋转矩阵（与 SAPIEN／ManiSkill 的 wxyz 顺序一致）。"""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    cos, sin = math.cos(angle), math.sin(angle)
    if axis == "X":
        return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float64)
    if axis == "Y":
        return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float64)
    if axis == "Z":
        return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"未知旋转轴 {axis}")


def euler_xyz_to_quat(angles_rad: Sequence[float]) -> np.ndarray:
    """复刻 ``euler_angles_to_matrix(..., convention="XYZ")`` 再取四元数。

    ManiSkill 的 ``rotation_conversions`` 是 pytorch3d 的搬运版，``"XYZ"`` 表示
    ``R = Rx(a) @ Ry(b) @ Rz(c)``；``build_bin`` 与 ``spawn_fixed_cube`` 都走这条路径。
    """
    a, b, c = (float(v) for v in angles_rad)
    matrix = _axis_rotation("X", a) @ _axis_rotation("Y", b) @ _axis_rotation("Z", c)
    return matrix_to_quat(matrix)


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """3×3 旋转矩阵转 wxyz 四元数，符号取 w ≥ 0，与 pytorch3d 的实现一致。"""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    if quat[0] < 0.0:
        quat = -quat
    return _normalize_quat(quat)


# ── 与 build_bin / spawn_random_cube 同源的几何描述 ───────────────────────────
def bin_shape_specs(cube_half_size: float) -> tuple[ShapeSpec, ...]:
    """容器的 6 个盒体，逐字复刻 ``build_bin`` 的 ``poses`` 与 ``half_sizes``。

    ⚠ ``build_bin`` 的注释说「底板加四壁」，源码另外还建了一个中央方块，实际是 6 个。
    """
    inner_side = cube_half_size * 2.5
    wall_thickness = 0.005
    wall_height = cube_half_size * 2.5
    floor_thickness = 0.004

    inner_half = inner_side * 0.5
    t = wall_thickness * 0.5
    h = wall_height * 0.5
    tf = floor_thickness * 0.5

    bottom_half = [inner_half + t, inner_half + t, tf]
    lr_wall_half = [t, inner_half + t, h]
    fb_wall_half = [inner_half + t, t, h]

    base_z = tf
    offset = inner_half + t
    z_wall = tf + h

    identity = np.array([1.0, 0.0, 0.0, 0.0])
    local = [
        ([0.0, 0.0, 0.0], [cube_half_size] * 3),
        ([0.0, 0.0, base_z], bottom_half),
        ([-offset, 0.0, z_wall], lr_wall_half),
        ([+offset, 0.0, z_wall], lr_wall_half),
        ([0.0, -offset, z_wall], fb_wall_half),
        ([0.0, +offset, z_wall], fb_wall_half),
    ]
    return tuple(
        ShapeSpec(np.array(p, dtype=np.float64), identity.copy(), np.array(half, dtype=np.float64))
        for p, half in local
    )


def cube_shape_specs(half_size: float) -> tuple[ShapeSpec, ...]:
    """方块的单个盒体：``actors.build_cube`` 的碰撞体就在 actor 原点。"""
    return (
        ShapeSpec(
            np.zeros(3, dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.full(3, float(half_size), dtype=np.float64),
        ),
    )


def bin_actor_pose(xy: Sequence[float], z_rotation_deg: float, cube_half_size: float) -> tuple[np.ndarray, np.ndarray]:
    """复刻 ``build_bin`` 的 ``builder.set_initial_pose``：翻转 180° 扣在桌面上。"""
    wall_height = cube_half_size * 2.5
    floor_thickness = 0.004
    h = wall_height * 0.5
    tf = floor_thickness * 0.5
    p = np.array([float(xy[0]), float(xy[1]), tf + 2.0 * h], dtype=np.float64)
    q = euler_xyz_to_quat([math.pi, 0.0, math.radians(float(z_rotation_deg))])
    return p, q


def cube_actor_pose(xy: Sequence[float], yaw_rad: float, half_size: float) -> tuple[np.ndarray, np.ndarray]:
    """复刻 ``spawn_random_cube`` 的落点：底面贴桌，绕 z 轴 yaw。"""
    p = np.array([float(xy[0]), float(xy[1]), float(half_size)], dtype=np.float64)
    q = euler_xyz_to_quat([0.0, 0.0, float(yaw_rad)])
    return p, q


# ── 静态判据：三维分离轴 ─────────────────────────────────────────────────────
def _world_box(state_p: np.ndarray, state_rot: np.ndarray, shape: ShapeSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """actor 世界位姿 × 形状局部位姿 → 世界盒体（中心、列为轴的 3×3、半尺寸）。"""
    center = state_p + state_rot @ shape.local_p
    axes = state_rot @ quat_to_matrix(shape.local_q)
    return center, axes, shape.half


def _radius(half: np.ndarray, axes: np.ndarray, n: np.ndarray) -> float:
    return float(np.sum(half * np.abs(axes.T @ n)))


def sat_gap(
    center_a: np.ndarray,
    axes_a: np.ndarray,
    half_a: np.ndarray,
    center_b: np.ndarray,
    axes_b: np.ndarray,
    half_b: np.ndarray,
) -> float:
    """单个盒对的分离轴判定值 ``g = max_n gap(n)``。

    候选轴 15 根：双方各 3 个面法向 + 9 个边叉积；叉积范数 ≤ 1e-12 的退化轴跳过。
    ``g`` 是分离轴判定值，一般不等于最短欧氏距离；非有限数按不安全处理（返回 NaN）。
    """
    delta = center_b - center_a
    best = -np.inf
    for k in range(3):
        for axis in (axes_a[:, k], axes_b[:, k]):
            gap = abs(float(np.dot(delta, axis))) - _radius(half_a, axes_a, axis) - _radius(half_b, axes_b, axis)
            if gap > best:
                best = gap
    for i in range(3):
        for j in range(3):
            axis = np.cross(axes_a[:, i], axes_b[:, j])
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                continue
            axis = axis / norm
            gap = abs(float(np.dot(delta, axis))) - _radius(half_a, axes_a, axis) - _radius(half_b, axes_b, axis)
            if gap > best:
                best = gap
    if not math.isfinite(best):
        return float("nan")
    return float(best)


def _classify(gap: float) -> str | None:
    """把判定值翻成拒绝原因；``None`` 表示这一对分开。"""
    if not math.isfinite(gap):
        return "numerical_boundary"
    if gap < 0.0:
        return "contact"  # 穿入
    if gap <= EPS_M:
        return "numerical_boundary"  # 接触或落在 [0, ε] 的数值边界带，含正好相切的 g == 0
    return None


def check_pair_static(
    obj_a: ObjectState,
    obj_b: ObjectState,
    *,
    stage: str = "initial",
    sweep_index: int | None = None,
) -> tuple[float, CollisionRejection | None]:
    """一对对象的全部盒对静态检查，返回最小判定值与拒绝证据。

    ⚠ 遍历全部盒对后才出结论，证据指向**判定值最小**的那一对形状，而不是遍历里
    碰上的第一对。否则同一个几何状态会因形状顺序不同报出不同的 ``reason``：
    两个容器中心距 0.04 时最小 ``g = −0.01``（前后壁真穿入），但中央方块与前壁那一对
    恰好 ``g = 0``，早退会把一次真穿入报成数值边界。
    """
    rot_a = quat_to_matrix(obj_a.q)
    rot_b = quat_to_matrix(obj_b.q)
    worst = np.inf
    worst_pair: tuple[int, int] | None = None
    worst_finite = True
    for ia, shape_a in enumerate(obj_a.shapes):
        box_a = _world_box(obj_a.p, rot_a, shape_a)
        for ib, shape_b in enumerate(obj_b.shapes):
            box_b = _world_box(obj_b.p, rot_b, shape_b)
            gap = sat_gap(*box_a, *box_b)
            if not math.isfinite(gap):
                # 非有限判定值一律按不安全处理，且优先于任何有限值成为证据
                return float("nan"), CollisionRejection(
                    reason="numerical_boundary",
                    stage=stage,
                    object_a=obj_a.name,
                    object_b=obj_b.name,
                    shape_a=ia,
                    shape_b=ib,
                    gap_m=None,
                    sweep_index=sweep_index,
                    detail="分离轴判定值非有限",
                )
            if gap < worst:
                worst, worst_pair = gap, (ia, ib)
    if worst_pair is None:
        return float("nan"), CollisionRejection(
            reason="uncertified",
            stage=stage,
            object_a=obj_a.name,
            object_b=obj_b.name,
            shape_a=-1,
            shape_b=-1,
            sweep_index=sweep_index,
            detail="对象缺少碰撞形状",
        )
    reason = _classify(worst)
    if reason is None:
        return float(worst), None
    return float(worst), CollisionRejection(
        reason=reason,
        stage=stage,
        object_a=obj_a.name,
        object_b=obj_b.name,
        shape_a=worst_pair[0],
        shape_b=worst_pair[1],
        gap_m=float(worst),
        sweep_index=sweep_index,
    )


def check_bin_layout(
    objects: Sequence[ObjectState],
    *,
    stage: str = "initial",
    raise_on_reject: bool = False,
    exhaustive: bool = False,
) -> tuple[float, CollisionRejection | None]:
    """全部对象两两之间的静态检查；任一对被拒绝，整体拒绝。

    返回 ``(最小判定值, 拒绝证据或 None)``；证据指向判定值最小的那一对对象与形状，
    与遍历顺序无关。``raise_on_reject`` 为真时改抛 :class:`BinCollisionError`，
    供运行时检查点直接中止该样本。

    ⚠ **返回的最小值在两种模式下语义不同**：

    * ``exhaustive=False``（默认，运行时用）：先用包围球粗筛跳过必然分离的对象对，
      一次距离计算替掉 36 次 SAT。**判定**（排除与否）完全不受影响——被跳过的对必然
      分离——但返回的最小值只是「**精算过的那些对**里的最小」。某个被跳过的对真实间隙
      可能比它更小，只是同样安全。
    * ``exhaustive=True``（复现核对用）：不粗筛，逐对精算，返回的是真正的全场最小 g。
      ``COLLISION_REPRODUCE`` 要逐步对照保存下来的 ``sat_gap_m``，必须走这条。
    """
    worst = np.inf
    coarse_worst = np.inf
    worst_rejection: CollisionRejection | None = None
    # 每个对象的保守包围球半径：盒体顶点到 actor 原点的最大距离
    radii = [max(item.radii) for item in objects]
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            # 粗筛：球心距减两个半径已经大于 ε 时，这一对的 36 个盒对必然分离，
            # 一次距离计算就能替掉 36 次 SAT。只跳过必然通过的对，判据不变。
            clearance = float(np.linalg.norm(objects[i].p - objects[j].p)) - radii[i] - radii[j]
            if not exhaustive and clearance > EPS_M:
                # 与 check_swap_sweep 同样的道理：包围球间隙是真实 g 的保守下界，
                # 不能混进 worst，否则「最危险对象对的最小 g」会被一个远处的对压低
                coarse_worst = min(coarse_worst, clearance)
                continue
            gap, rejection = check_pair_static(objects[i], objects[j], stage=stage)
            if rejection is not None and not math.isfinite(gap):
                if raise_on_reject:
                    raise BinCollisionError(rejection)
                return gap, rejection
            if gap < worst:
                worst, worst_rejection = gap, rejection
    if worst_rejection is not None:
        if raise_on_reject:
            raise BinCollisionError(worst_rejection)
        return float(worst), worst_rejection
    if math.isfinite(worst):
        return float(worst), None
    # 全部对象对都在粗筛里过掉了：退回包围球下界，仍然是「已证明分离」
    return (float(coarse_worst) if math.isfinite(coarse_worst) else float("nan")), None


def check_bin_state(
    objects: Sequence[ObjectState],
    *,
    stage: str = "state",
    raise_on_reject: bool = False,
) -> tuple[float, CollisionRejection | None]:
    """运行时某一时刻的只读复核，与 :func:`check_bin_layout` 同判据，只换 ``stage`` 标签。"""
    return check_bin_layout(objects, stage=stage, raise_on_reject=raise_on_reject)


# ── 连续判据：区间二分证明 ───────────────────────────────────────────────────
def _lane_endpoints(a_xy: np.ndarray, b_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """复刻 ``swap_flat_two_lane`` 的主方向与左法向；起终点重合时法向取零。"""
    delta = b_xy - a_xy
    normal = np.array([-delta[1], delta[0]], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm > COINCIDENT_XY:
        normal = normal / norm
    else:
        normal = np.zeros(2, dtype=np.float64)
    return delta, normal


def _quat_at(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """A 的 ``qa(s) = normalize((1−s)qa0 + s qb0)``；保留原符号与归一化线性混合。"""
    blended = (1.0 - s) * q0 + s * q1
    norm = float(np.linalg.norm(blended))
    if not math.isfinite(norm) or norm <= DEGENERATE_NORM:
        raise ValueError("四元数线性混合退化")
    return blended / norm


def _min_blend_norm(q0: np.ndarray, dq: np.ndarray, lo: float, hi: float) -> float:
    """``m = min_{s∈[lo,hi]} ||q0 + s·Δq||``：线段到原点的最近距离，端点或投影点取到。"""
    dd = float(np.dot(dq, dq))
    candidates = [lo, hi]
    if dd > 0.0:
        star = -float(np.dot(q0, dq)) / dd
        if lo < star < hi:
            candidates.append(star)
    return min(float(np.linalg.norm(q0 + s * dq)) for s in candidates)


@dataclass
class _Mover:
    """交换中一个对象在参数 ``s`` 上的位姿函数与速度上界所需的量。"""

    name: str
    shapes: Sequence[ShapeSpec]
    radii: Sequence[float]
    xy0: np.ndarray
    z: float
    delta: np.ndarray  # 主方向位移（B−A），本对象按 sign 取用
    normal: np.ndarray
    sign: float  # A 取 +1，B 取 −1
    q0: np.ndarray
    q1: np.ndarray

    def pose_at(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        offset = LANE_OFFSET * math.sin(math.pi * s)
        xy = self.xy0 + self.sign * (self.delta * s + self.normal * offset)
        return np.array([xy[0], xy[1], self.z], dtype=np.float64), _quat_at(self.q0, self.q1, s)

    def bounding_sphere(self) -> tuple[np.ndarray, float]:
        """整段路径上该对象全部盒体点的保守包围球。

        actor 原点轨迹 ``xy0 + sign·(δ·s + n·0.07 sin(πs))`` 包在以中点为心、
        半径 ``0.5‖δ‖ + 0.07`` 的球内；再加上盒体顶点到原点的最大距离即可。
        """
        mid_xy = self.xy0 + self.sign * (self.delta * 0.5 + self.normal * LANE_OFFSET)
        center = np.array([mid_xy[0], mid_xy[1], self.z], dtype=np.float64)
        radius = 0.5 * float(np.linalg.norm(self.delta)) + LANE_OFFSET + max(self.radii)
        return center, radius

    def speed_bound(self, lo: float, hi: float, shape_index: int) -> float:
        """该盒体上任意点在 ``[lo,hi]`` 内对 ``s`` 的线速度上界。"""
        translation = float(np.linalg.norm(self.delta)) + LANE_OFFSET * math.pi
        dq = self.q1 - self.q0
        m = _min_blend_norm(self.q0, dq, lo, hi)
        if not math.isfinite(m) or m <= DEGENERATE_NORM:
            raise ValueError("四元数插值退化，算不出角速度上界")
        omega = 2.0 * float(np.linalg.norm(dq)) / m
        return translation + self.radii[shape_index] * omega


@dataclass
class _Static:
    """交换中不动的旁观对象；速度上界恒为 0。"""

    name: str
    shapes: Sequence[ShapeSpec]
    radii: Sequence[float]
    p: np.ndarray
    q: np.ndarray

    def pose_at(self, s: float) -> tuple[np.ndarray, np.ndarray]:  # noqa: ARG002
        return self.p, self.q

    def bounding_sphere(self) -> tuple[np.ndarray, float]:
        return self.p, max(self.radii)

    def speed_bound(self, lo: float, hi: float, shape_index: int) -> float:  # noqa: ARG002
        return 0.0


def _prove_pair(
    left: Any,
    right: Any,
    ia: int,
    ib: int,
    *,
    stage: str,
    sweep_index: int | None,
) -> tuple[float, CollisionRejection | None]:
    """对一个盒对做区间二分证明；返回 ``(观察到的最小 g, 拒绝证据或 None)``。"""
    shape_a = left.shapes[ia]
    shape_b = right.shapes[ib]

    def gap_at(s: float) -> float:
        pa, qa = left.pose_at(s)
        pb, qb = right.pose_at(s)
        return sat_gap(
            *_world_box(pa, quat_to_matrix(qa), shape_a),
            *_world_box(pb, quat_to_matrix(qb), shape_b),
        )

    def reject(reason: str, s: float | None, gap: float | None, lo: float, hi: float, depth: int, used: int, detail: str) -> CollisionRejection:
        return CollisionRejection(
            reason=reason,
            stage=stage,
            object_a=left.name,
            object_b=right.name,
            shape_a=ia,
            shape_b=ib,
            gap_m=gap if gap is not None and math.isfinite(gap) else None,
            s=s,
            sweep_index=sweep_index,
            interval_low=lo,
            interval_high=hi,
            depth=depth,
            intervals_used=used,
            detail=detail,
        )

    worst = np.inf
    used = 0

    # 先查两端：端点都通过也不能放行，但端点不通过可以立刻拒绝。
    for s in (0.0, 1.0):
        try:
            gap = gap_at(s)
        except ValueError as exc:
            return float("nan"), reject("uncertified", s, None, 0.0, 1.0, 0, used, str(exc))
        used += 1
        if math.isfinite(gap):
            worst = min(worst, gap)
        reason = _classify(gap)
        if reason is not None:
            return gap, reject(reason, s, gap, s, s, 0, used, "端点判定")

    # 从根区间中点开始，左子区间先于右子区间。
    stack: list[tuple[float, float, int]] = [(0.0, 1.0, 0)]
    while stack:
        lo, hi, depth = stack.pop()
        if used >= MAX_INTERVALS:
            return (
                float(worst) if math.isfinite(worst) else float("nan"),
                reject("uncertified", None, None, lo, hi, depth, used, f"区间数超过上限 {MAX_INTERVALS}"),
            )
        if depth > MAX_DEPTH:
            return (
                float(worst) if math.isfinite(worst) else float("nan"),
                reject("uncertified", None, None, lo, hi, depth, used, f"二分深度超过上限 {MAX_DEPTH}"),
            )
        mid = 0.5 * (lo + hi)
        try:
            gap = gap_at(mid)
            bound_a = left.speed_bound(lo, hi, ia)
            bound_b = right.speed_bound(lo, hi, ib)
        except ValueError as exc:
            return (
                float(worst) if math.isfinite(worst) else float("nan"),
                reject("uncertified", mid, None, lo, hi, depth, used, str(exc)),
            )
        used += 1
        if math.isfinite(gap):
            worst = min(worst, gap)
        reason = _classify(gap)
        if reason is not None:
            return gap, reject(reason, mid, gap, lo, hi, depth, used, "中点判定")
        margin = (bound_a + bound_b) * (hi - lo) * 0.5
        if not math.isfinite(margin):
            return (
                float(worst) if math.isfinite(worst) else float("nan"),
                reject("uncertified", mid, gap, lo, hi, depth, used, "速度上界非有限"),
            )
        if gap > EPS_M + margin:
            continue  # 整个区间已证明分离
        # 右子区间后进先出地压栈，保证左子区间先被证明。
        stack.append((mid, hi, depth + 1))
        stack.append((lo, mid, depth + 1))

    return (float(worst) if math.isfinite(worst) else float("nan")), None


def check_swap_sweep(
    moving_a: ObjectState,
    moving_b: ObjectState,
    bystanders: Sequence[ObjectState] = (),
    *,
    sweep_index: int | None = None,
    stage: str = "sweep",
    raise_on_reject: bool = False,
) -> tuple[float, CollisionRejection | None]:
    """整段交换路径的连续检查：交换双方之间、双方各自与每个旁观对象之间。

    路径按 ``swap_flat_two_lane`` 的实际语义复算：``A`` 沿 ``+0.07 sin(πs)`` 的左法向弯道
    去 ``B`` 的位置，``B`` 沿 ``−`` 号弯道去 ``A`` 的位置，四元数互换并做归一化线性混合。
    旁观对象在整段里被原函数逐帧按住不动，因此速度上界取 0。

    ⚠ 不抽帧：每个盒对都必须由区间二分证明整段分离，证明不出来按 ``uncertified`` 拒绝。
    ⚠ 与 :func:`check_bin_layout` 不同，本函数在第一个被证否的盒对上就返回——盒对数量是
    静态检查的几十倍且每对都要二分，跑完全部只为挑「最严重」的一对不划算。因此返回的
    证据是「第一个证否的盒对」，不保证是判定值最小的那一对。
    """
    a_xy = moving_a.p[:2]
    b_xy = moving_b.p[:2]
    delta, normal = _lane_endpoints(a_xy, b_xy)

    mover_a = _Mover(
        name=moving_a.name,
        shapes=moving_a.shapes,
        radii=moving_a.radii,
        xy0=a_xy.copy(),
        z=float(moving_a.p[2]),
        delta=delta,
        normal=normal,
        sign=1.0,
        q0=moving_a.q.copy(),
        q1=moving_b.q.copy(),
    )
    mover_b = _Mover(
        name=moving_b.name,
        shapes=moving_b.shapes,
        radii=moving_b.radii,
        xy0=b_xy.copy(),
        z=float(moving_b.p[2]),
        delta=delta,
        normal=normal,
        sign=-1.0,
        q0=moving_b.q.copy(),
        q1=moving_a.q.copy(),
    )
    statics = [
        _Static(name=item.name, shapes=item.shapes, radii=item.radii, p=item.p.copy(), q=item.q.copy())
        for item in bystanders
    ]

    pairs: list[tuple[Any, Any]] = [(mover_a, mover_b)]
    for item in statics:
        pairs.append((mover_a, item))
        pairs.append((mover_b, item))

    worst = np.inf
    coarse_worst = np.inf
    for left, right in pairs:
        # 粗筛：两个保守包围球都分开时，这一对的 36 个盒对必然整段分离，跳过二分。
        # 只跳过必然通过的对，判据本身不变；旁观容器多半在这一步就过掉，二分次数大幅下降。
        # ⚠ 被跳过的对**不进 worst**：包围球间隙是真实 g 的保守下界，混进去会把
        # 「最危险对象对的最小 g」压成一个离得很远的对的下界，报告与图都会误导。
        center_l, radius_l = left.bounding_sphere()
        center_r, radius_r = right.bounding_sphere()
        clearance = float(np.linalg.norm(center_l - center_r)) - radius_l - radius_r
        if clearance > EPS_M:
            coarse_worst = min(coarse_worst, clearance)
            continue
        for ia in range(len(left.shapes)):
            for ib in range(len(right.shapes)):
                gap, rejection = _prove_pair(left, right, ia, ib, stage=stage, sweep_index=sweep_index)
                if rejection is not None:
                    if raise_on_reject:
                        raise BinCollisionError(rejection)
                    return gap, rejection
                if math.isfinite(gap):
                    worst = min(worst, gap)
    if math.isfinite(worst):
        return float(worst), None
    # 全部对象对都在粗筛里过掉了：没有精算值，退回包围球下界，仍然是「已证明分离」
    return (float(coarse_worst) if math.isfinite(coarse_worst) else float("nan")), None


# ── V5：多对同时交换的联合连续判据与认证预筛（只加不改） ─────────────────────
#: 认证预筛在 s∈[0,1] 上的等距采样点数（V5 计划 2.5 / L23）。
PREFILTER_SAMPLES = 401
#: 认证预筛的放行余量，米。预筛证出的「竖直圆柱间隙下界」必须大于它才跳过精确证明。
#: 取 1 毫米而不是 ``EPS_M``：让被跳过的对与「精确证明也一定能证出来」的对之间留出三个数量级的缓冲，
#: 把「真实分离但分离轴判定值落在数值边界带、或二分耗尽」这种理论上的分歧挤到不可能的区域（见
#: :func:`check_multi_swap_sweep` 的说明）。它只决定跳不跳精确证明，不参与任何拒绝判定。
PREFILTER_MARGIN_M = 1e-3


def _swap_movers(moving_a: ObjectState, moving_b: ObjectState) -> tuple[_Mover, _Mover]:
    """按 :func:`check_swap_sweep` 完全相同的方式构造一对交换者（逐字段一致，保证单对时逐位相同）。"""
    a_xy = moving_a.p[:2]
    b_xy = moving_b.p[:2]
    delta, normal = _lane_endpoints(a_xy, b_xy)
    mover_a = _Mover(
        name=moving_a.name,
        shapes=moving_a.shapes,
        radii=moving_a.radii,
        xy0=a_xy.copy(),
        z=float(moving_a.p[2]),
        delta=delta,
        normal=normal,
        sign=1.0,
        q0=moving_a.q.copy(),
        q1=moving_b.q.copy(),
    )
    mover_b = _Mover(
        name=moving_b.name,
        shapes=moving_b.shapes,
        radii=moving_b.radii,
        xy0=b_xy.copy(),
        z=float(moving_b.p[2]),
        delta=delta,
        normal=normal,
        sign=-1.0,
        q0=moving_b.q.copy(),
        q1=moving_a.q.copy(),
    )
    return mover_a, mover_b


def _quat_to_matrix_batch(q: np.ndarray) -> np.ndarray:
    """``(N,4)`` 的 wxyz 四元数批量转 ``(N,3,3)`` 旋转矩阵，公式与 :func:`quat_to_matrix` 相同。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - z * w)
    out[:, 0, 2] = 2 * (x * z + y * w)
    out[:, 1, 0] = 2 * (x * y + z * w)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - x * w)
    out[:, 2, 0] = 2 * (x * z - y * w)
    out[:, 2, 1] = 2 * (y * z + x * w)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def _local_vertices(shapes: Sequence[ShapeSpec]) -> np.ndarray:
    """全部盒体的 8 个顶点在 actor 局部坐标系里的位置，``(8·形状数, 3)``。"""
    signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    blocks = []
    for shape in shapes:
        rot = quat_to_matrix(shape.local_q)
        blocks.append(shape.local_p[None, :] + (signs * shape.half[None, :]) @ rot.T)
    return np.concatenate(blocks, axis=0)


@dataclass
class _PrefilterTrack:
    """一个对象在预筛采样点上的竖直包围圆柱：原点 XY 轨迹、圆柱半径与二者合起来的 Lipschitz 常数。

    ``xy[i]`` 与 ``rho[i]`` 是 ``s = PREFILTER_S[i]`` 时 actor 原点的 XY 与「全部盒体顶点到原点竖直轴的最大
    水平距离」。对象整个落在以 ``xy[i]`` 为轴、半径 ``rho[i]`` 的无限高竖直圆柱里（凸包由顶点张成）。
    ``lipschitz`` 是 ``‖xy(s)‖`` 与 ``rho(s)`` 对 ``s`` 的变化率上界之和：原点平移速度上界
    ``‖δ‖ + 0.07π`` 加上旋转引起的顶点速度上界 ``max(radii)·2‖Δq‖/m``（与 ``_Mover.speed_bound`` 同式，
    ``m`` 取整段 ``[0,1]`` 上的最小混合范数）。静止物为 0。
    """

    xy: np.ndarray  # (N, 2)
    rho: np.ndarray  # (N,)
    lipschitz: float


def _prefilter_samples() -> np.ndarray:
    return np.linspace(0.0, 1.0, PREFILTER_SAMPLES)


def _prefilter_track(obj: Any, samples: np.ndarray) -> _PrefilterTrack | None:
    """为交换者或静止物构造预筛圆柱；四元数混合退化（算不出角速度上界）时返回 ``None``，该对象不参与预筛。"""
    vertices = _local_vertices(obj.shapes)
    if isinstance(obj, _Static):
        rot = quat_to_matrix(obj.q)
        rho = float(np.max(np.linalg.norm((vertices @ rot.T)[:, :2], axis=1)))
        n = samples.shape[0]
        return _PrefilterTrack(
            xy=np.repeat(np.asarray(obj.p, dtype=np.float64)[None, :2], n, axis=0),
            rho=np.full(n, rho, dtype=np.float64),
            lipschitz=0.0,
        )
    if not isinstance(obj, _Mover):
        raise TypeError(f"预筛不认识的对象类型 {type(obj).__name__}")
    dq = obj.q1 - obj.q0
    m = _min_blend_norm(obj.q0, dq, 0.0, 1.0)
    if not math.isfinite(m) or m <= DEGENERATE_NORM:
        # 退化的对象交给精确证明去报 uncertified，预筛绝不替它放行
        return None
    s = samples[:, None]
    offset = LANE_OFFSET * np.sin(np.pi * s)
    xy = obj.xy0[None, :] + obj.sign * (obj.delta[None, :] * s + obj.normal[None, :] * offset)
    blended = (1.0 - s) * obj.q0[None, :] + s * obj.q1[None, :]
    blended = blended / np.linalg.norm(blended, axis=1, keepdims=True)
    rots = _quat_to_matrix_batch(blended)
    horizontal = np.einsum("nij,vj->nvi", rots[:, :2, :], vertices)
    rho = np.max(np.linalg.norm(horizontal, axis=2), axis=1)
    translation = float(np.linalg.norm(obj.delta)) + LANE_OFFSET * math.pi
    omega = 2.0 * float(np.linalg.norm(dq)) / m
    lipschitz = translation + max(obj.radii) * omega
    if not (np.all(np.isfinite(xy)) and np.all(np.isfinite(rho)) and math.isfinite(lipschitz)):
        return None
    return _PrefilterTrack(xy=xy, rho=rho, lipschitz=float(lipschitz))


def _prefilter_clearance(left: _PrefilterTrack, right: _PrefilterTrack, step: float) -> float:
    """两个竖直圆柱在整段 ``s∈[0,1]`` 上的水平间隙下界。

    ``G(s) = ‖xy_l(s) − xy_r(s)‖ − rho_l(s) − rho_r(s)`` 是 ``L = L_l + L_r`` -Lipschitz 的；在相邻两个采样点
    ``[s_i, s_i + h]`` 之间 ``G(s) ≥ max(G_i − L(s−s_i), G_{i+1} − L(s_{i+1}−s)) ≥ (G_i + G_{i+1})/2 − L·h/2``。
    对全部小区间取最小即整段下界；下界为正就证明了两个圆柱（从而两个对象）整段分离。
    """
    gaps = np.linalg.norm(left.xy - right.xy, axis=1) - left.rho - right.rho
    if gaps.shape[0] < 2:
        return float("nan")
    worst_mid = float(np.min(0.5 * (gaps[:-1] + gaps[1:])))
    bound = worst_mid - (left.lipschitz + right.lipschitz) * step * 0.5
    return bound if math.isfinite(bound) else float("nan")


def check_multi_swap_sweep(
    pairs: Sequence[tuple[ObjectState, ObjectState]],
    bystanders: Sequence[ObjectState] = (),
    *,
    sweep_index: int | None = None,
    stage: str = "sweep",
    raise_on_reject: bool = False,
    prefilter: bool = True,
    stats: dict[str, int] | None = None,
) -> tuple[float, CollisionRejection | None]:
    """同一窗口里多对同时交换的连续检查（V5 计划 2.5：外环随内环同步交换；2.15：按钮底座作静止障碍）。

    ``pairs`` 里每一项 ``(A, B)`` 都按 ``swap_flat_two_lane(lane_offset=0.07, smooth=True)`` 在**同一个窗口**
    里对换：同一控制步上所有对的进度 ``α``（smoothstep 之后）相同，因此全体位姿都是同一个参数 ``s`` 的函数，
    可以直接用 :func:`_prove_pair` 证明（它本来就能证两个都在动的对象）。``bystanders`` 是整段静止的对象，
    可以是容器、方块，也可以是 :func:`static_box_state` / :func:`static_rect_state` /
    :func:`static_state_from_obb2d` / :func:`button_base_state` 构造的任意有向盒体。

    检查的对象对与顺序：

    1. 每一对自身（交换双方），按 ``pairs`` 顺序；
    2. 跨对的交换者两两（交换者按 ``A1, B1, A2, B2, …`` 排序后取 ``i < j`` 且不同对）；
    3. 每个静止物依次对全部交换者（先静止物、后交换者，与 :func:`check_swap_sweep` 的旁观对象顺序一致）。

    **只有一对时，检查的对象对、顺序、粗筛与每个盒对的证明都与** :func:`check_swap_sweep` **完全相同**；
    ``prefilter=False`` 时返回值逐位相同（单测大量随机样例核对）。

    ``prefilter=True``（默认，L23 已定开启）时，在包围球粗筛之后、精确证明之前再做一层认证预筛：对每个对象
    在 ``s`` 上等距取 ``PREFILTER_SAMPLES = 401`` 个点，算竖直包围圆柱的水平间隙，用 Lipschitz 界把样本间的
    空隙补成整段下界（:func:`_prefilter_clearance`）；下界大于 ``PREFILTER_MARGIN_M`` 的对象对**已被证明整段
    分离**，跳过逐盒对的二分证明。预筛**只会跳过、不会拒绝**：拒绝一律来自 :func:`_prove_pair`，按同样的
    对象对顺序遇到的第一个被证否的盒对给出，所以拒绝证据与不预筛时逐位相同。

    ⚠ 与 :func:`check_swap_sweep` 的包围球粗筛同理，被跳过的对**不进返回的最小 g**；通过时返回值是
    「精算过的那些盒对里的最小 g」，全部对象对都被跳过时退回粗筛/预筛下界里的最小值。预筛开关因此可能
    改变**通过时**返回的数，但不改变判定与拒绝证据。
    ⚠ 理论上唯一可能的分歧：某对真实间隙 > 1 mm，但精确证明因分离轴判定值 ≤ ε 或二分耗尽而证不出来——
    这与原函数的包围球粗筛是同一个前提（粗筛跳过的对同样没有经过精确证明），1 mm 余量让它在实际几何下
    不会出现。四元数混合退化的交换者不参与预筛，照旧交给精确证明报 ``uncertified``。

    ``stats`` 传入一个字典时，就地累加计数：``object_pairs``（考察的对象对数）、``coarse_skipped``（包围球
    粗筛跳过）、``prefilter_skipped``（认证预筛跳过）、``proved_object_pairs``（进入精确证明的对象对）、
    ``proved_shape_pairs``（实际完成证明或被证否的盒对数）。

    返回 ``(最小判定值, 拒绝证据或 None)``，与 :func:`check_swap_sweep` 同结构；``raise_on_reject`` 为真时
    改抛 :class:`BinCollisionError`。同一个对象（按 ``name``）不得同时出现在两处，否则抛 ``ValueError``。
    """
    pair_list = [tuple(item) for item in pairs]
    if not pair_list:
        raise ValueError("check_multi_swap_sweep 至少需要一对交换对象")
    names: list[str] = []
    for item in pair_list:
        if len(item) != 2:
            raise ValueError(f"每一对必须恰好两个对象，收到 {len(item)} 个")
        names.extend(obj.name for obj in item)
    names.extend(obj.name for obj in bystanders)
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"对象名重复出现（同一对象不能既在两对里、或既交换又静止）：{duplicated}")

    movers: list[_Mover] = []
    pair_of: list[int] = []
    for index, (moving_a, moving_b) in enumerate(pair_list):
        mover_a, mover_b = _swap_movers(moving_a, moving_b)
        movers.extend((mover_a, mover_b))
        pair_of.extend((index, index))
    statics = [
        _Static(name=item.name, shapes=item.shapes, radii=item.radii, p=item.p.copy(), q=item.q.copy())
        for item in bystanders
    ]

    checks: list[tuple[Any, Any]] = [(movers[2 * k], movers[2 * k + 1]) for k in range(len(pair_list))]
    for i in range(len(movers)):
        for j in range(i + 1, len(movers)):
            if pair_of[i] != pair_of[j]:
                checks.append((movers[i], movers[j]))
    for item in statics:
        for mover in movers:
            checks.append((mover, item))

    counts = {
        "object_pairs": 0,
        "coarse_skipped": 0,
        "prefilter_skipped": 0,
        "proved_object_pairs": 0,
        "proved_shape_pairs": 0,
    }
    samples = _prefilter_samples() if prefilter else None
    step = float(samples[1] - samples[0]) if samples is not None else 0.0
    tracks: dict[int, _PrefilterTrack | None] = {}

    def track_of(obj: Any) -> _PrefilterTrack | None:
        key = id(obj)
        if key not in tracks:
            tracks[key] = _prefilter_track(obj, samples)
        return tracks[key]

    def flush_stats() -> None:
        if stats is not None:
            for key, value in counts.items():
                stats[key] = stats.get(key, 0) + value

    worst = np.inf
    coarse_worst = np.inf
    for left, right in checks:
        counts["object_pairs"] += 1
        # 第一层：与 check_swap_sweep 完全相同的整段包围球粗筛
        center_l, radius_l = left.bounding_sphere()
        center_r, radius_r = right.bounding_sphere()
        clearance = float(np.linalg.norm(center_l - center_r)) - radius_l - radius_r
        if clearance > EPS_M:
            coarse_worst = min(coarse_worst, clearance)
            counts["coarse_skipped"] += 1
            continue
        # 第二层：认证预筛（只跳过已证明分离的对）
        if samples is not None:
            track_l = track_of(left)
            track_r = track_of(right)
            if track_l is not None and track_r is not None:
                bound = _prefilter_clearance(track_l, track_r, step)
                if math.isfinite(bound) and bound > PREFILTER_MARGIN_M:
                    coarse_worst = min(coarse_worst, bound)
                    counts["prefilter_skipped"] += 1
                    continue
        # 第三层：逐盒对区间二分证明，与 check_swap_sweep 同一个 _prove_pair
        counts["proved_object_pairs"] += 1
        for ia in range(len(left.shapes)):
            for ib in range(len(right.shapes)):
                gap, rejection = _prove_pair(left, right, ia, ib, stage=stage, sweep_index=sweep_index)
                counts["proved_shape_pairs"] += 1
                if rejection is not None:
                    flush_stats()
                    if raise_on_reject:
                        raise BinCollisionError(rejection)
                    return gap, rejection
                if math.isfinite(gap):
                    worst = min(worst, gap)
    flush_stats()
    if math.isfinite(worst):
        return float(worst), None
    return (float(coarse_worst) if math.isfinite(coarse_worst) else float("nan")), None


def check_swap_sweep_prefiltered(
    moving_a: ObjectState,
    moving_b: ObjectState,
    bystanders: Sequence[ObjectState] = (),
    *,
    sweep_index: int | None = None,
    stage: str = "sweep",
    raise_on_reject: bool = False,
    prefilter: bool = True,
    stats: dict[str, int] | None = None,
) -> tuple[float, CollisionRejection | None]:
    """单对交换的带预筛包装：等于 ``check_multi_swap_sweep([(moving_a, moving_b)], bystanders, ...)``。

    :func:`check_swap_sweep` 本身保持原样、从不预筛；想给单对检查提速（如 Swap 两环境 reset 时的内环对内环
    预判、VideoRepick 的搭档可行性过滤）就改调本函数。``prefilter=False`` 时与 :func:`check_swap_sweep`
    返回值逐位相同；``prefilter=True`` 时判定与拒绝证据相同，只有「通过时的最小 g」可能不同。
    """
    return check_multi_swap_sweep(
        [(moving_a, moving_b)],
        bystanders,
        sweep_index=sweep_index,
        stage=stage,
        raise_on_reject=raise_on_reject,
        prefilter=prefilter,
        stats=stats,
    )


# ── V5：构造静止障碍（任意有向盒体／矩形）────────────────────────────────────
def static_box_state(
    name: str,
    center: Sequence[float],
    half_size: Sequence[float],
    *,
    yaw_rad: float = 0.0,
) -> ObjectState:
    """一个绕 z 轴转 ``yaw_rad`` 的实心盒体作静止物：世界中心 ``center (3,)``、半尺寸 ``half_size (3,)``。

    actor 原点就放在盒体中心、局部位姿为单位，因此包围球与预筛圆柱都是紧的。
    """
    center_arr = np.asarray(center, dtype=np.float64).reshape(3)
    half_arr = np.asarray(half_size, dtype=np.float64).reshape(3)
    if not (np.all(np.isfinite(center_arr)) and np.all(np.isfinite(half_arr))) or np.any(half_arr <= 0.0):
        raise ValueError(f"静止盒体参数非法：center={center_arr.tolist()} half={half_arr.tolist()}")
    shape = ShapeSpec(np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]), half_arr)
    return ObjectState(name=name, p=center_arr, q=euler_xyz_to_quat([0.0, 0.0, float(yaw_rad)]), shapes=(shape,))


def static_rect_state(
    name: str,
    center_xy: Sequence[float],
    half_xy: Sequence[float],
    *,
    yaw_rad: float = 0.0,
    z_range: tuple[float, float],
) -> ObjectState:
    """桌面上一个有向矩形沿 z 拉伸成的静止盒体：XY 中心、XY 半尺寸、绕 z 的朝向，外加必须显式给出的
    ``z_range = (z_low, z_high)``（米）。``z_range`` 故意不给默认值：障碍多高直接决定会不会挡住某个物体，
    写错高度会静默放行，必须由调用方按实际几何填写（例如「只挡地面上的东西」可以给 ``(0.0, 1.0)``）。
    """
    z_low, z_high = (float(v) for v in z_range)
    if not (math.isfinite(z_low) and math.isfinite(z_high)) or z_high <= z_low:
        raise ValueError(f"z_range 非法：{z_range}")
    cx, cy = (float(v) for v in center_xy)
    hx, hy = (float(v) for v in half_xy)
    return static_box_state(
        name,
        [cx, cy, 0.5 * (z_low + z_high)],
        [hx, hy, 0.5 * (z_high - z_low)],
        yaw_rad=yaw_rad,
    )


def static_state_from_obb2d(
    name: str,
    obb2d: tuple[Sequence[float], Any, Sequence[float]],
    *,
    z_range: tuple[float, float],
    tol: float = 1e-6,
) -> ObjectState:
    """把仓库放置逻辑用的二维 OBB 三元组 ``(c (2,), A (2×2，列为轴), h (2,))`` 转成静止盒体。

    三元组格式与 ``object_generation._trimesh_box_to_obb2d`` / ``create_button_obb`` 的返回值一致。
    ``A`` 的两列必须是单位正交向量（误差 ``tol`` 内），否则抛 ``ValueError``——倾斜物体投影出来的非正交轴
    不能当成矩形。第二列允许与第一列成左手系（翻转 180° 扣放的容器投影就是这样），矩形关于轴对称，
    只取第一列定朝向。
    """
    c, axes, half = obb2d
    axes_arr = np.asarray(axes, dtype=np.float64).reshape(2, 2)
    gram = axes_arr.T @ axes_arr
    if not np.all(np.isfinite(gram)) or float(np.max(np.abs(gram - np.eye(2)))) > tol:
        raise ValueError(f"二维 OBB 的轴不是单位正交的：A={axes_arr.tolist()}")
    yaw = math.atan2(float(axes_arr[1, 0]), float(axes_arr[0, 0]))
    return static_rect_state(name, c, half, yaw_rad=yaw, z_range=z_range)


def button_base_state(
    name: str,
    center_xy: Sequence[float],
    *,
    scale: float = 1.0,
    base_half: Sequence[float] = (0.025, 0.025, 0.005),
) -> ObjectState:
    """复刻 ``object_generation.build_button`` 的按钮**底座**碰撞盒：半尺寸 ``base_half × scale``，
    轴对齐，底面贴桌（中心 z = 半高）。

    只含底座，不含上面那个圆柱按帽（碰撞模型只有盒体）；V5 计划 L54 要求把底座 OBB 当交换扫掠的静止
    障碍。``center_xy`` 必须是 ``build_button`` 最终使用的中心（随机偏移与规格回注之后的值）。
    """
    half = np.asarray(base_half, dtype=np.float64).reshape(3) * float(scale)
    return static_box_state(name, [float(center_xy[0]), float(center_xy[1]), float(half[2])], half)


def nearest_partner_index(reference: Sequence[float], candidates: Sequence[tuple[int, Sequence[float]]]) -> tuple[int, list[tuple[int, float]]]:
    """按 ``step`` 的原语义扫描最近邻，返回 ``(选中的序号, 全部候选的距离表)``。

    复刻原实现的两个细节：判定用 ``dist < closest_dist`` **严格小于**，候选按传入顺序
    （即生成顺序）遍历，因此等距时先出现的胜出——这正是 ``tie_break=first_in_spawn_order``。
    距离表原样返回，供不一致时留证：临界等距和真的错绑要能分得开。
    """
    best_index = -1
    best_dist = float("inf")
    table: list[tuple[int, float]] = []
    origin = np.asarray(reference, dtype=np.float64)[:2]
    for index, position in candidates:
        dist = float(np.linalg.norm(origin - np.asarray(position, dtype=np.float64)[:2]))
        table.append((index, dist))
        if dist < best_dist:
            best_dist = dist
            best_index = index
    return best_index, table


# ── 运行时：从真实 actor 读盒体 ──────────────────────────────────────────────
def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def shape_specs_from_actor(actor: Any) -> tuple[ShapeSpec, ...]:
    """读 actor 上每个 ``PhysxCollisionShapeBox`` 的 ``half_size`` 与 ``local_pose``。

    形状顺序按引擎返回顺序保留，不排序、不合并；遇到非盒体形状立即报错，
    不用整体 AABB 替代——那只能初筛，不能当判据。
    """
    entity = getattr(actor, "_objs", None)
    if entity:
        entity = entity[0]
    else:
        entity = actor
    components = None
    for attr in ("find_component_by_type", "components"):
        if hasattr(entity, attr):
            break
    try:
        from sapien.physx import PhysxRigidBaseComponent  # type: ignore

        component = entity.find_component_by_type(PhysxRigidBaseComponent)
        components = component.get_collision_shapes()
    except Exception as exc:  # pragma: no cover - 只在无 SAPIEN 环境下触发
        raise BinCollisionError(
            CollisionRejection(
                reason="uncertified",
                stage="geometry",
                object_a=str(getattr(actor, "name", actor)),
                object_b="-",
                shape_a=-1,
                shape_b=-1,
                detail=f"读不到碰撞形状：{exc}",
            )
        ) from exc

    specs: list[ShapeSpec] = []
    for index, shape in enumerate(components):
        half = getattr(shape, "half_size", None)
        if half is None:
            raise BinCollisionError(
                CollisionRejection(
                    reason="uncertified",
                    stage="geometry",
                    object_a=str(getattr(actor, "name", actor)),
                    object_b="-",
                    shape_a=index,
                    shape_b=-1,
                    detail=f"第 {index} 个形状不是盒体：{type(shape).__name__}",
                )
            )
        pose = shape.local_pose
        specs.append(
            ShapeSpec(
                np.asarray(pose.p, dtype=np.float64).reshape(3),
                np.asarray(pose.q, dtype=np.float64).reshape(4),
                np.asarray(half, dtype=np.float64).reshape(3),
            )
        )
    return tuple(specs)


def object_state_from_actor(actor: Any, name: str | None = None) -> ObjectState:
    """把一个真实 actor 的当前世界位姿与真实盒体打包成 :class:`ObjectState`。"""
    pose = actor.pose if hasattr(actor, "pose") else actor.get_pose()
    p = _to_numpy(pose.p)[:3]
    q = _to_numpy(pose.q)[:4]
    return ObjectState(
        name=name or str(getattr(actor, "name", "actor")),
        p=p,
        q=q,
        shapes=shape_specs_from_actor(actor),
    )


def object_states_from_actors(actors_map: Iterable[tuple[str, Any]]) -> list[ObjectState]:
    return [object_state_from_actor(actor, name) for name, actor in actors_map]
