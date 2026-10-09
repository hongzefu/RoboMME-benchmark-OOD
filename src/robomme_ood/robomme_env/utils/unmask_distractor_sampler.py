"""V5 xhard：四个 Unmask 环境共用的干扰容器统一采样器与独立停放点（NEWTASK_RELEASE_V5_PLAN 2.2 / L13 / L14）。

只被 xhard 分支调用；原三档从不进入本模块，所以不影响 V0/V1。本模块**不改**任何既有共用函数
（``object_generation.spawn_random_bin``、``statechange.*``、``bin_collision.check_swap_sweep`` 都只读引用），
V4 的 ``unmask_distractors.spawn_ring_distractor_bins`` 与 ``unmask_swap_xhard.sample_distractors`` 原样保留，
由各环境在接入时改调本模块。

一、统一采样器（L13：一套实现、一套 schema，随机流由调用方注入）
-----------------------------------------------------------------
逐容器拒绝采样（每个容器 ``max_trials`` 次，V5 统一 1024）：

1. 在 ``[-R, R]²``（``R = ring_max_abs_xy[1]``）上均匀抽 ``x``、``y``（2 次 ``rand``）；
2. 拒绝 ``max(|x|,|y|)`` 不在 ``[r_in, r_out]`` 的点（max-norm 环带）；
3. 精确 8 角点可见：容器按「任意 yaw 的外接正方形」取底面与顶面 8 个角点投到前视相机，全部落在画面内才接受
   （与 V4 VU/BU 的 ``visible_in_camera(bin_corners(...))`` 同一判据，比 Swap 两环境 V4 的线性近似严格）；
4. OBB 间距：候选中心到每个障碍 2D OBB 的点距须 ``≥ 采样半边 0.0275 + min_gap``，``min_gap = cube_half_size ×
   min_gap_factor``（V5 统一 0.75，即 0.015）；已放的干扰容器以**精确**矩形（由 x、y、yaw 算出，外扩 min_gap）进障碍；
5. 通过后抽 yaw（1 次 ``rand``，``u·90°``）；
6. 若调用方给了 ``extra_reject``（Swap 两环境的扫掠 / 路径规则），在 yaw 之后调用，返回真即拒绝、继续下一次尝试。
   回调**不许**抽随机数。

全部容器放完后依次抽：含 cube 个数 ``randint(lo, hi+1)`` → ``cube_bins = randperm(N)[:n]`` →
``color_order = randperm(3)``；颜色规则 ``balanced_cycle``：第 j 个 cube 用 ``DISTRACTOR_COLORS[color_order[j % 3]]``，
名字 ``distractor_cube_<j>_<colour>``。这与 V4 VU/BU 的抽样次序与次数逐项相同（V4 颜色是 ``randperm(3)[:n]``，
同一次 randperm），不给回调时整个随机调用序列就是 V4 VU/BU 的序列。

二、记录纪律（N18）
-------------------
采样函数 :func:`sample_distractor_layout` 是**纯几何**的：只抽随机数、不碰 recorder、不建 actor，
Swap 的 reset 规划可以先拿它判可行、整段重抽，接受后再由 :func:`commit_distractor_layout` 统一走
``recorder.value``（每个取值点只调一次），尝试次数与失败原因走 ``recorder.record``。
:func:`resample_distractor_layout` 把「整段重抽 → 接受 → 记录」串成一个驱动。回放冻结规格时，
``commit`` 会按同一规则复核冻结布局（N17 精神），违反即抛 :class:`SceneGenerationError`。

统一 schema（``spec_prefix`` 默认 ``objects.distractors``，与 V4 VU/BU 的路径一致）::

    objects.distractors.requested      record  请求个数（commit 时写）
    objects.distractors.bins.<i>       value   [x, y, yaw_deg]      decision_key …ring_max_abs_xy
    objects.distractors.placed         record  实际个数
    objects.distractors.trials         record  每个容器用掉的尝试次数
    objects.distractors.cube_count     value   含 cube 个数         decision_key …cube_count_range
    objects.distractors.cube_bins      value   第 j 个 cube 所在容器 decision_key …count
    objects.distractors.color_order    value   randperm(3)          decision_key …color_rule
    objects.distractors.cube_colors    record  第 j 个 cube 的颜色名
    objects.distractors.cube_names     record  第 j 个 cube 的 actor 名
    layout.distractor_layout_attempts  record  （整段重抽模式）被接受的是第几次尝试
    layout.distractor_layout_failures  record  （整段重抽模式）此前每次失败的原因

三、独立停放点（L14）
---------------------
揭示 / 交换窗口里，V4 把所有容器与被藏 cube 都传送到同一点 (10,10,10)，22～24 个物体互相穿插，
接触求解把单步耗时从 38 ms 拉到 108～417 ms（计划 2.2 陷阱 1）。这里给每个物体一个独立的画面外停放点：
按组（干扰容器 / 干扰 cube / 内环容器 / 内环被藏 cube）分行、组内按序号等距排开，相邻点相距
:data:`XHARD_PARK_PITCH_M` = 0.5 m（物体最大外廓约 0.085 m），与 (10,10,10) 也隔开，彼此永不接触。
:func:`lift_and_park_back_to_original` / :func:`lift_and_park_onto` 与 ``statechange`` 里对应函数的时间线逐步相同
（窗口起点、半窗落回步、终点放回规则都不变），唯一差别是「远处」换成调用方给的停放点；缓存用独立的属性名，
与 ``statechange`` 的缓存互不干扰。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .bin_collision import bin_actor_pose, object_state_from_actor, quat_to_matrix
from .SceneGenerationError import SceneGenerationError
from .unmask_distractors import (
    DISTRACTOR_BIN_PREFIX,
    DISTRACTOR_CUBE_PREFIX,
    bin_corners,
    bin_geometry,
    in_ring,
    visible_in_camera,
)
from .xhard import DISTRACTOR_COLORS

# ── 配置 ────────────────────────────────────────────────────────────────────────
#: 唯一支持的颜色规则（L12 (a)）：三色平衡轮转。
COLOR_RULE_BALANCED_CYCLE = "balanced_cycle"
#: 统一配置的键（四个环境的 ``decision.xhard.distractor`` 都用这一套键名）。
DISTRACTOR_CFG_KEYS = (
    "count",
    "ring_max_abs_xy",
    "cube_count_range",
    "color_pool",
    "color_rule",
    "min_gap_factor",
    "max_trials",
)

# 计划 2.2 的密度推导常量（L6 / L7 / L10）：VU/BU 内部 8 个容器 / 0.4×0.4 = 50 个/m²；
# 环带宽 W = 1/√ρ；环带与内部范围之间留 g = 0.015（xhard min_gap）；0.0275 是容器采样半边。
V5_INNER_DENSITY_PER_M2 = 50.0
V5_RING_GAP_M = 0.015
V5_RING_BAND_WIDTH_M = 1.0 / math.sqrt(V5_INNER_DENSITY_PER_M2)

#: 四个环境的 V5 xhard 干扰配置（统一键）。环境文件的 ``XHARD_DISTRACTOR`` 应直接深拷贝这里的值，
#: 数量与环带按 V5 计划 2.2 的公式确定。
V5_DISTRACTOR_PRESETS: dict[str, dict[str, Any]] = {
    # 2.3：贴身环带 [0.2 + g + 0.0275, 0.2 + g + W − 0.0275]，N = floor(50·0.3027 + 0.5) = 15，含 cube [7,8]
    "VideoUnmask": {
        "count": 15,
        "ring_max_abs_xy": [0.2425, 0.3289],
        "cube_count_range": [7, 8],
        "color_pool": [c["name"] for c in DISTRACTOR_COLORS],
        "color_rule": COLOR_RULE_BALANCED_CYCLE,
        "min_gap_factor": 0.75,
        "max_trials": 1024,
    },
    # 2.4：同一环带，按钮挡 5.7% ⇒ A_usable 0.2853，N = 14，含 cube [7,7]
    "ButtonUnmask": {
        "count": 14,
        "ring_max_abs_xy": [0.2425, 0.3289],
        "cube_count_range": [7, 7],
        "color_pool": [c["name"] for c in DISTRACTOR_COLORS],
        "color_rule": COLOR_RULE_BALANCED_CYCLE,
        "min_gap_factor": 0.75,
        "max_trials": 1024,
    },
    # 2.6 / L16 (b)：沿用 V4 环带 [0.2675, 0.45]，10 个，含 cube [5,5]
    "VideoUnmaskSwap": {
        "count": 10,
        "ring_max_abs_xy": [0.2675, 0.45],
        "cube_count_range": [5, 5],
        "color_pool": [c["name"] for c in DISTRACTOR_COLORS],
        "color_rule": COLOR_RULE_BALANCED_CYCLE,
        "min_gap_factor": 0.75,
        "max_trials": 1024,
    },
    # 2.7：与 VUS 相同
    "ButtonUnmaskSwap": {
        "count": 10,
        "ring_max_abs_xy": [0.2675, 0.45],
        "cube_count_range": [5, 5],
        "color_pool": [c["name"] for c in DISTRACTOR_COLORS],
        "color_rule": COLOR_RULE_BALANCED_CYCLE,
        "min_gap_factor": 0.75,
        "max_trials": 1024,
    },
}


@dataclass(frozen=True)
class DistractorConfig:
    """校验过的统一配置。"""

    count: int
    ring: tuple[float, float]
    cube_count_range: tuple[int, int]
    color_rule: str
    min_gap_factor: float
    max_trials: int


def parse_distractor_cfg(cfg: dict | DistractorConfig) -> DistractorConfig:
    """把 ``decision.xhard.distractor`` 子树校验成 :class:`DistractorConfig`；不认识或缺少的键都直接报错。"""
    if isinstance(cfg, DistractorConfig):
        return cfg
    keys = set(cfg)
    missing = [k for k in DISTRACTOR_CFG_KEYS if k not in keys]
    extra = sorted(keys - set(DISTRACTOR_CFG_KEYS))
    if missing or extra:
        raise ValueError(f"干扰配置键不符：缺 {missing}，多 {extra}（统一键为 {list(DISTRACTOR_CFG_KEYS)}）")
    count = int(cfg["count"])
    # V7 定值表：VideoUnmask／ButtonUnmask 的 xhard1 贴身环带干扰为 0（0928 方案 §3.2.2、第二部分 §1.7）
    if count < 0:
        raise ValueError(f"count 必须 ≥ 0，收到 {cfg['count']}")
    ring = tuple(float(v) for v in cfg["ring_max_abs_xy"])
    if len(ring) != 2 or not (0.0 < ring[0] < ring[1]):
        raise ValueError(f"ring_max_abs_xy 非法：{cfg['ring_max_abs_xy']}")
    lo, hi = (int(v) for v in cfg["cube_count_range"])
    # V5：颜色平衡轮转后 cube 个数不再受色数限制，只要求 0 ≤ lo ≤ hi ≤ count（计划 2.3）
    if not 0 <= lo <= hi <= count:
        raise ValueError(f"cube_count_range {cfg['cube_count_range']} 必须满足 0 ≤ lo ≤ hi ≤ count={count}")
    pool = [c["name"] for c in DISTRACTOR_COLORS]
    if list(cfg["color_pool"]) != pool:
        # 色池是全局决策 B2（黄/青/品红），也被 PickXtimes/SwingXtimes 共用，不许按环境改
        raise ValueError(f"color_pool 必须等于全局干扰色池 {pool}，收到 {cfg['color_pool']}")
    if cfg["color_rule"] != COLOR_RULE_BALANCED_CYCLE:
        raise ValueError(f"color_rule 只支持 {COLOR_RULE_BALANCED_CYCLE!r}，收到 {cfg['color_rule']!r}")
    gap_factor = float(cfg["min_gap_factor"])
    if not (math.isfinite(gap_factor) and gap_factor >= 0.0):
        raise ValueError(f"min_gap_factor 必须是非负有限数，收到 {cfg['min_gap_factor']}")
    max_trials = int(cfg["max_trials"])
    if max_trials < 1:
        raise ValueError(f"max_trials 必须 ≥ 1，收到 {cfg['max_trials']}")
    return DistractorConfig(count, (ring[0], ring[1]), (lo, hi), COLOR_RULE_BALANCED_CYCLE, gap_factor, max_trials)


# ── 几何 ────────────────────────────────────────────────────────────────────────
Obb2d = tuple[np.ndarray, np.ndarray, np.ndarray]  # (中心 (2,), 列为轴的 2×2, 半边 (2,))


def bin_outer_half(cube_half_size: float) -> float:
    """容器外廓（正方形）半边：内口半边 + 壁厚（与 ``build_bin`` 同源，cube_half_size=0.02 时为 0.03）。"""
    return cube_half_size * 2.5 * 0.5 + 0.005


def _yaw_axes_from_quat(q) -> np.ndarray:
    """直立（含翻转 180°）物体的 2D 轴：取旋转矩阵里水平投影最长的两列并单位化。"""
    rot = quat_to_matrix(q)
    cols = [rot[:2, k] for k in range(3)]
    norms = [float(np.linalg.norm(c)) for c in cols]
    order = sorted(range(3), key=lambda k: -norms[k])[:2]
    return np.stack([cols[k] / norms[k] for k in order], axis=1)


def bin_obb2d(x: float, y: float, yaw_deg: float, cube_half_size: float, pad: float = 0.0) -> Obb2d:
    """由 ``(x, y, yaw)`` 精确算出容器 2D OBB（``build_bin`` 翻转 180° 后绕 z 转 yaw），外扩 ``pad``。"""
    _p, q = bin_actor_pose([x, y], yaw_deg, cube_half_size)
    half = bin_outer_half(cube_half_size) + float(pad)
    return (np.array([float(x), float(y)], dtype=np.float64), _yaw_axes_from_quat(q), np.array([half, half]))


def actor_obb2d_exact(actor, pad: float = 0.0) -> Obb2d:
    """按 actor 的真实碰撞盒算 2D 包围矩形（外扩 ``pad``）。

    不走 ``get_actor_obb`` → ``_trimesh_box_to_obb2d``：那条路径对近似正方体会把竖直轴排进前两列，
    2D 框退化成线段（计划 2.0①）。这里直接用 actor 的位姿与 ``PhysxCollisionShapeBox``：
    直立物体取水平投影最长的两根轴，否则退回世界轴对齐框（任何姿态下都是保守包围）。
    """
    state = object_state_from_actor(actor)
    rot = quat_to_matrix(state.q)
    # 直立（含翻转 180°）：某一根局部轴与世界 z 轴平行，其余两根的水平投影就是精确的 2D 轴
    upright = float(np.max(np.abs(rot[2, :]))) > 1.0 - 1e-6
    axes = _yaw_axes_from_quat(state.q) if upright else np.eye(2)
    corners = []
    for shape in state.shapes:
        center = state.p + rot @ shape.local_p
        box_axes = rot @ quat_to_matrix(shape.local_q)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append((center + box_axes @ (shape.half * np.array([sx, sy, sz])))[:2])
    local = (np.asarray(corners) - state.p[:2]) @ axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    center = state.p[:2] + axes @ ((lo + hi) / 2.0)
    return center, axes, (hi - lo) / 2.0 + float(pad)


def obstacle_obbs(avoid: Sequence[Any], min_gap: float) -> list[Obb2d]:
    """把避让列表转成 2D OBB 列表；收集口径与 V4 一致：

    * 预制三元组 ``(c, A, h)``（如按钮的 ``create_button_obb``，已含安全区）原样使用；
    * ``(actor, pad)`` 按 ``pad`` 外扩；裸 actor 按 ``min_gap`` 外扩；
    * actor 用 :func:`actor_obb2d_exact` 精确求框；读不到盒体形状时退回 V4 的 trimesh 路径，再不行就忽略
      （与 V4 ``_obstacle_obbs`` 相同：没有物理网格的对象不参与避让）。
    """
    out: list[Obb2d] = []
    for item in avoid:
        if isinstance(item, tuple):
            if len(item) == 3 and isinstance(item[0], np.ndarray) and isinstance(item[1], np.ndarray):
                out.append((np.asarray(item[0], np.float64)[:2], np.asarray(item[1], np.float64),
                            np.asarray(item[2], np.float64)[:2]))
                continue
            actor, pad = item
        else:
            actor, pad = item, min_gap
        try:
            out.append(actor_obb2d_exact(actor, float(pad)))
            continue
        except Exception:  # noqa: BLE001 非盒体形状等：退回 V4 路径
            pass
        try:
            from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb

            from .object_generation import _trimesh_box_to_obb2d

            out.append(_trimesh_box_to_obb2d(get_actor_obb(actor, to_world_frame=True, vis=False), extra_pad=float(pad)))
        except Exception:  # noqa: BLE001 与 V4 一致：没有物理网格的对象忽略
            pass
    return out


def point_hits_obbs(pos: np.ndarray, obbs: Sequence[Obb2d], reach: float) -> bool:
    """点到任一 OBB 的距离 < ``reach`` 即真（与 V4 ``unmask_distractors._hits`` 同一判据）。"""
    for c_obs, a_obs, h_obs in obbs:
        local = a_obs.T @ (pos - c_obs)
        closest = c_obs + a_obs @ np.clip(local, -h_obs, h_obs)
        if np.linalg.norm(pos - closest) < reach:
            return True
    return False


def bin_visible(x: float, y: float, cube_half_size: float) -> bool:
    """精确 8 角点可见判据（与 V4 VU/BU 相同）。"""
    _half, reach_any_yaw, height = bin_geometry(cube_half_size)
    return visible_in_camera(bin_corners(float(x), float(y), reach_any_yaw, height))


# ── 放置结果 ────────────────────────────────────────────────────────────────────
@dataclass
class DistractorLayout:
    """一次完整的干扰布局（纯几何，不含 actor）。"""

    bins: list[tuple[float, float, float]]  # 逐容器 (x, y, yaw_deg)
    cube_count: int
    cube_bins: list[int]  # 第 j 个 cube 所在容器的序号
    color_order: list[int]  # randperm(3)，下标指向 DISTRACTOR_COLORS
    trials: list[int] = field(default_factory=list)  # 每个容器用掉的尝试次数（只作留痕）

    @property
    def count(self) -> int:
        return len(self.bins)

    @property
    def cube_color_indices(self) -> list[int]:
        """balanced_cycle：第 j 个 cube 用 ``color_order[j % 3]``。"""
        n = len(self.color_order)
        return [int(self.color_order[j % n]) for j in range(len(self.cube_bins))]

    @property
    def cube_colors(self) -> list[str]:
        return [DISTRACTOR_COLORS[i]["name"] for i in self.cube_color_indices]

    @property
    def bin_names(self) -> list[str]:
        return [f"{DISTRACTOR_BIN_PREFIX}_{i}" for i in range(len(self.bins))]

    @property
    def cube_names(self) -> list[str]:
        return [f"{DISTRACTOR_CUBE_PREFIX}_{j}_{color}" for j, color in enumerate(self.cube_colors)]

    def bin_obbs(self, cube_half_size: float, pad: float = 0.0) -> list[Obb2d]:
        return [bin_obb2d(x, y, yaw, cube_half_size, pad) for x, y, yaw in self.bins]

    def same_geometry(self, other: "DistractorLayout") -> bool:
        """位置、cube 映射与颜色是否逐值相同（不比 trials）。"""
        return (
            [list(map(float, b)) for b in self.bins] == [list(map(float, b)) for b in other.bins]
            and int(self.cube_count) == int(other.cube_count)
            and [int(v) for v in self.cube_bins] == [int(v) for v in other.cube_bins]
            and [int(v) for v in self.color_order] == [int(v) for v in other.color_order]
        )

    def to_spec(self) -> dict[str, Any]:
        """统一 schema 的纯数据视图（与 ``commit_distractor_layout`` 写进规格的字段一一对应）。"""
        return {
            "bins": [[float(x), float(y), float(yaw)] for x, y, yaw in self.bins],
            "cube_count": int(self.cube_count),
            "cube_bins": [int(v) for v in self.cube_bins],
            "color_order": [int(v) for v in self.color_order],
            "cube_colors": self.cube_colors,
            "cube_names": self.cube_names,
            "bin_names": self.bin_names,
        }


class DistractorPlacementError(SceneGenerationError):
    """某个容器在 ``max_trials`` 次内找不到合法位置（候选级拒绝）。``placed`` 为已放下的个数。"""

    def __init__(self, message: str, placed: int):
        super().__init__(message)
        self.placed = int(placed)


#: 额外拒绝回调：``(序号 i, x, y, yaw_deg, 已放容器列表) -> 是否拒绝``；不许抽随机数。
ExtraReject = Callable[[int, float, float, float, Sequence[tuple[float, float, float]]], bool]


def sample_distractor_layout(
    cfg: dict | DistractorConfig,
    *,
    obstacles: Sequence[Obb2d],
    generator: torch.Generator,
    cube_half_size: float,
    extra_reject: ExtraReject | None = None,
) -> DistractorLayout:
    """纯几何地抽一整套干扰布局（不建 actor、不碰 recorder）。

    ``obstacles`` 是已在场对象的 2D OBB（用 :func:`obstacle_obbs` 从避让列表转来；actor 已按 min_gap 外扩）。
    随机调用顺序见模块文档；某个容器放不下即抛 :class:`DistractorPlacementError`。
    """
    c = parse_distractor_cfg(cfg)
    half, _reach_any_yaw, _height = bin_geometry(cube_half_size)
    min_gap = float(cube_half_size) * c.min_gap_factor
    reject_reach = half + min_gap
    span = c.ring[1]
    obbs = list(obstacles)
    bins: list[tuple[float, float, float]] = []
    trials: list[int] = []
    for i in range(c.count):
        placed = None
        used = 0
        for _ in range(c.max_trials):
            used += 1
            x = float(torch.rand(1, generator=generator).item() * 2.0 * span - span)
            y = float(torch.rand(1, generator=generator).item() * 2.0 * span - span)
            if not in_ring(x, y, c.ring):
                continue
            if not bin_visible(x, y, cube_half_size):
                continue
            if point_hits_obbs(np.array([x, y], dtype=np.float64), obbs, reject_reach):
                continue
            yaw = float(torch.rand(1, generator=generator).item() * 90.0)
            if extra_reject is not None and extra_reject(i, x, y, yaw, list(bins)):
                continue
            placed = (x, y, yaw)
            break
        if placed is None:
            raise DistractorPlacementError(
                f"xhard 干扰容器放不满：请求 {c.count} 个，第 {i} 个在 {c.max_trials} 次尝试内无可行位置", placed=len(bins)
            )
        bins.append(placed)
        trials.append(used)
        obbs.append(bin_obb2d(placed[0], placed[1], placed[2], cube_half_size, pad=min_gap))

    lo, hi = c.cube_count_range
    n_cubes = int(torch.randint(lo, hi + 1, (1,), generator=generator).item())
    cube_bins = torch.randperm(c.count, generator=generator)[:n_cubes].tolist()
    color_order = torch.randperm(len(DISTRACTOR_COLORS), generator=generator).tolist()
    return DistractorLayout(bins=bins, cube_count=n_cubes, cube_bins=cube_bins, color_order=color_order, trials=trials)


def verify_distractor_layout(
    layout: DistractorLayout,
    cfg: dict | DistractorConfig,
    *,
    obstacles: Sequence[Obb2d],
    cube_half_size: float,
    extra_reject: ExtraReject | None = None,
) -> list[str]:
    """按采样的同一套规则复核一整套布局，返回违反项（空列表即合法）；不抽随机数。"""
    c = parse_distractor_cfg(cfg)
    half, _r, _h = bin_geometry(cube_half_size)
    min_gap = float(cube_half_size) * c.min_gap_factor
    reach = half + min_gap
    problems: list[str] = []
    if len(layout.bins) != c.count:
        problems.append(f"容器个数 {len(layout.bins)} ≠ {c.count}")
    obbs = list(obstacles)
    for i, (x, y, yaw) in enumerate(layout.bins):
        if not in_ring(x, y, c.ring):
            problems.append(f"容器 {i} 不在环带 {list(c.ring)}：({x:.4f},{y:.4f})")
        if not bin_visible(x, y, cube_half_size):
            problems.append(f"容器 {i} 不在画面内：({x:.4f},{y:.4f})")
        if point_hits_obbs(np.array([x, y], dtype=np.float64), obbs, reach):
            problems.append(f"容器 {i} 与障碍或已放容器间距不足：({x:.4f},{y:.4f})")
        if not 0.0 <= float(yaw) <= 90.0:
            problems.append(f"容器 {i} 的 yaw {yaw} 不在 [0,90]")
        if extra_reject is not None and extra_reject(i, float(x), float(y), float(yaw), list(layout.bins[:i])):
            problems.append(f"容器 {i} 被调用方的额外规则拒绝")
        obbs.append(bin_obb2d(x, y, yaw, cube_half_size, pad=min_gap))
    lo, hi = c.cube_count_range
    if not lo <= int(layout.cube_count) <= hi:
        problems.append(f"含 cube 个数 {layout.cube_count} 不在 [{lo},{hi}]")
    cube_bins = [int(v) for v in layout.cube_bins]
    if len(cube_bins) != int(layout.cube_count) or len(set(cube_bins)) != len(cube_bins) or any(
        not 0 <= v < len(layout.bins) for v in cube_bins
    ):
        problems.append(f"cube_bins {cube_bins} 与个数 {layout.cube_count} / 容器数 {len(layout.bins)} 不符")
    if sorted(int(v) for v in layout.color_order) != list(range(len(DISTRACTOR_COLORS))):
        problems.append(f"color_order {layout.color_order} 不是 0..{len(DISTRACTOR_COLORS) - 1} 的排列")
    return problems


def commit_distractor_layout(
    layout: DistractorLayout,
    *,
    cfg: dict | DistractorConfig,
    recorder,
    obstacles: Sequence[Obb2d],
    cube_half_size: float,
    spec_prefix: str = "objects.distractors",
    decision_prefix: str = "xhard.distractor",
    extra_reject: ExtraReject | None = None,
) -> DistractorLayout:
    """把**已被接受**的布局经 ``recorder`` 落进规格，返回真正用于建场景的布局。

    导出模式返回原布局；回注模式返回冻结值拼成的布局，并按同一规则复核（N17 精神），违反即抛
    :class:`SceneGenerationError`。每个取值点只调一次 ``recorder.value``（N18）。
    """
    c = parse_distractor_cfg(cfg)
    recorder.record(f"{spec_prefix}.requested", c.count)
    bins = []
    for i, entry in enumerate(layout.bins):
        x, y, yaw = recorder.value(f"{spec_prefix}.bins.{i}", [float(v) for v in entry],
                                   decision_key=f"{decision_prefix}.ring_max_abs_xy")
        bins.append((float(x), float(y), float(yaw)))
    recorder.record(f"{spec_prefix}.placed", len(bins))
    recorder.record(f"{spec_prefix}.trials", [int(v) for v in layout.trials])
    cube_count = int(recorder.value(f"{spec_prefix}.cube_count", int(layout.cube_count),
                                    decision_key=f"{decision_prefix}.cube_count_range"))
    cube_bins = [int(v) for v in recorder.value(f"{spec_prefix}.cube_bins", [int(v) for v in layout.cube_bins],
                                                decision_key=f"{decision_prefix}.count")]
    color_order = [int(v) for v in recorder.value(f"{spec_prefix}.color_order", [int(v) for v in layout.color_order],
                                                  decision_key=f"{decision_prefix}.color_rule")]
    final = DistractorLayout(bins=bins, cube_count=cube_count, cube_bins=cube_bins, color_order=color_order,
                             trials=list(layout.trials))
    # 回注时冻结布局要按本档的障碍与额外规则（如本档内环扫掠）复核
    if getattr(recorder, "replaying", False):
        problems = verify_distractor_layout(final, c, obstacles=obstacles, cube_half_size=cube_half_size,
                                            extra_reject=extra_reject)
        if problems:
            raise SceneGenerationError("回放的冻结干扰布局违反 V5 规则：" + "；".join(problems))
    recorder.record(f"{spec_prefix}.cube_colors", final.cube_colors)
    recorder.record(f"{spec_prefix}.cube_names", final.cube_names)
    return final


#: 整段重抽模式的接受回调：``layout -> (是否接受, 接受时的附带结果 / 拒绝时的原因字符串)``。
#: 回调可以从同一随机流继续抽（如 Swap 外环发起者的 ``randperm``），这是调用方随机流设计的一部分。
AcceptFn = Callable[[DistractorLayout], tuple[bool, Any]]


def resample_distractor_layout(
    cfg: dict | DistractorConfig,
    *,
    obstacles: Sequence[Obb2d],
    generator: torch.Generator,
    cube_half_size: float,
    recorder,
    accept: AcceptFn,
    max_attempts: int,
    extra_reject: ExtraReject | None = None,
    spec_prefix: str = "objects.distractors",
    decision_prefix: str = "xhard.distractor",
    attempts_path: str = "layout.distractor_layout_attempts",
    failures_path: str = "layout.distractor_layout_failures",
    require_replay_match: bool = True,
) -> tuple[DistractorLayout, Any]:
    """整段重抽驱动：最多 ``max_attempts`` 次「放置 → accept」，第一次被接受的布局才走 ``recorder.value``（N18）。

    * 某次放置放不满（:class:`DistractorPlacementError`）记为一次失败尝试（原因 ``placement``），继续重抽；
    * ``accept`` 返回 ``(False, 原因)`` 同样记为失败尝试；
    * 被接受后 ``record`` 尝试序号与此前的失败原因，再 :func:`commit_distractor_layout`；
    * 回注模式下若冻结布局与本次被接受的布局不同（说明代码或规则已变），``accept`` 的附带结果已不对应冻结布局，
      ``require_replay_match`` 为真时抛 :class:`SceneGenerationError`；
    * 全部失败时 ``record`` 失败原因并抛 :class:`SceneGenerationError`（候选级重抽）。
    """
    attempts = int(max_attempts)
    if attempts < 1:
        raise ValueError(f"max_attempts 必须 ≥ 1，收到 {max_attempts}")
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            layout = sample_distractor_layout(cfg, obstacles=obstacles, generator=generator,
                                              cube_half_size=cube_half_size, extra_reject=extra_reject)
        except DistractorPlacementError:
            failures.append("placement")
            continue
        ok, payload = accept(layout)
        if not ok:
            failures.append(str(payload) if payload else "rejected")
            continue
        recorder.record(attempts_path, attempt)
        recorder.record(failures_path, list(failures))
        final = commit_distractor_layout(layout, cfg=cfg, recorder=recorder, obstacles=obstacles,
                                         cube_half_size=cube_half_size, spec_prefix=spec_prefix,
                                         decision_prefix=decision_prefix, extra_reject=extra_reject)
        if require_replay_match and getattr(recorder, "replaying", False) \
                and not final.same_geometry(layout):
            raise SceneGenerationError("回放的冻结干扰布局与本次重抽被接受的布局不一致，规划结果无法对应冻结布局")
        return final, payload
    recorder.record(attempts_path, attempts)
    recorder.record(failures_path, list(failures))
    raise SceneGenerationError(f"干扰布局 {attempts} 次整段重抽都不可行：{failures}")


# ── 建 actor ─────────────────────────────────────────────────────────────────────
def build_distractor_actors(env, layout: DistractorLayout, *, hidden_half_size: float):
    """按布局建干扰容器与 cube，返回 ``(bins, cubes)``；第 j 个 cube 放在 ``bins[cube_bins[j]]`` 的 XY。

    容器名 ``distractor_bin_<i>``、cube 名 ``distractor_cube_<j>_<colour>``（带序号，颜色重复也不会重名）；
    **不设 ``bin_<i>`` 属性**，避免被按 ``bin_<i>`` 扫描的揭示 / 交换逻辑误收。
    """
    from .object_generation import build_bin, spawn_fixed_cube

    names = layout.bin_names + layout.cube_names
    if len(set(names)) != len(names):
        raise SceneGenerationError(f"干扰 actor 名字重复：{names}")
    bins = [
        build_bin(env, callsign=name, position=[x, y, 0.002], z_rotation_deg=yaw)
        for name, (x, y, yaw) in zip(layout.bin_names, layout.bins)
    ]
    cubes = []
    for j, (b_idx, name, c_idx) in enumerate(zip(layout.cube_bins, layout.cube_names, layout.cube_color_indices)):
        x, y, _yaw = layout.bins[int(b_idx)]
        cubes.append(spawn_fixed_cube(
            env,
            position=[float(x), float(y)],
            half_size=hidden_half_size,
            color=DISTRACTOR_COLORS[int(c_idx)]["rgba"],
            name_prefix=name,
            yaw=0.0,
            dynamic=True,
        ))
    return bins, cubes


def distractor_cube_bin_pairs(layout: DistractorLayout, bins: Sequence[Any], cubes: Sequence[Any]):
    """``[(cube_j, 它所在的容器)]``；Swap 两环境交换窗口里外环 cube 跟随容器时用。"""
    return [(cubes[j], bins[int(b_idx)]) for j, b_idx in enumerate(layout.cube_bins)]


def spawn_distractor_layout(
    env,
    *,
    cfg: dict | DistractorConfig,
    avoid: list,
    generator: torch.Generator,
    recorder,
    hidden_half_size: float,
    spec_prefix: str = "objects.distractors",
    decision_prefix: str = "xhard.distractor",
):
    """VU / BU 用的一站式入口：避让列表 → 纯几何采样 → 记录 → 建 actor；返回 ``(bins, cubes, layout)``。

    与 V4 ``spawn_ring_distractor_bins`` 相同的约定：调用方用主场景 generator、放在全部既有取值点之后（N5）；
    放下的容器追加进 ``avoid``；放不满直接抛 :class:`SceneGenerationError`（不许静默截断）。
    """
    c = parse_distractor_cfg(cfg)
    chs = float(env.cube_half_size)
    obstacles = obstacle_obbs(avoid, chs * c.min_gap_factor)
    try:
        layout = sample_distractor_layout(c, obstacles=obstacles, generator=generator, cube_half_size=chs)
    except DistractorPlacementError as exc:
        recorder.record(f"{spec_prefix}.requested", c.count)
        recorder.record(f"{spec_prefix}.placed", exc.placed)
        raise
    layout = commit_distractor_layout(layout, cfg=c, recorder=recorder, obstacles=obstacles, cube_half_size=chs,
                                      spec_prefix=spec_prefix, decision_prefix=decision_prefix)
    bins, cubes = build_distractor_actors(env, layout, hidden_half_size=hidden_half_size)
    avoid.extend(bins)
    return bins, cubes, layout


# ── 独立停放点（L14）──────────────────────────────────────────────────────────────
#: 停放网格原点。与 statechange 的 (10,10,10) 隔开 ≥ 14 m，在前视相机背后、远离桌面与机械臂。
XHARD_PARK_ORIGIN = (20.0, 20.0, 10.0)
#: 相邻停放点的间距（米）；物体最大外廓对角约 0.085 m，窗口内每个控制步都被传送回停放点、自由下落不到 2 cm。
XHARD_PARK_PITCH_M = 0.5
#: 每行停放点个数。
XHARD_PARK_ROW_LEN = 16
#: 每组最多停放点个数（占 4 行）。
XHARD_PARK_GROUP_CAPACITY = 64
#: 分组：干扰容器、干扰 cube、内环容器、内环被藏 cube 各占一块，互不重叠。
XHARD_PARK_GROUPS = ("distractor_bin", "distractor_cube", "bin", "hidden_cube")


def xhard_park_point(group: str, index: int) -> np.ndarray:
    """``group`` 组第 ``index`` 个物体的停放点（float32 xyz）。纯函数，不依赖调用次序。"""
    if group not in XHARD_PARK_GROUPS:
        raise ValueError(f"未知停放组 {group!r}，可选 {XHARD_PARK_GROUPS}")
    index = int(index)
    if not 0 <= index < XHARD_PARK_GROUP_CAPACITY:
        raise ValueError(f"停放序号 {index} 超出每组容量 {XHARD_PARK_GROUP_CAPACITY}")
    rows_per_group = XHARD_PARK_GROUP_CAPACITY // XHARD_PARK_ROW_LEN
    row = XHARD_PARK_GROUPS.index(group) * rows_per_group + index // XHARD_PARK_ROW_LEN
    col = index % XHARD_PARK_ROW_LEN
    x0, y0, z0 = XHARD_PARK_ORIGIN
    return np.array([x0 + col * XHARD_PARK_PITCH_M, y0 + row * XHARD_PARK_PITCH_M, z0], dtype=np.float32)


def _pose_quat(obj) -> np.ndarray:
    quat = obj.pose.q if hasattr(obj, "pose") else obj.get_pose().q
    if hasattr(quat, "detach"):
        quat = quat.detach().cpu().numpy()
    return np.asarray(quat, dtype=np.float32).flatten()


def _pose_xyz(obj) -> np.ndarray:
    p = obj.pose.p if hasattr(obj, "pose") else obj.get_pose().p
    if hasattr(p, "detach"):
        p = p.detach().cpu().numpy()
    return np.asarray(p, dtype=np.float64).reshape(-1)[:3]


def _teleport(obj, target_pos, quat) -> None:
    """与 statechange 里的 ``_teleport`` 同语义：设位姿并清零速度，失败只吞掉。"""
    import sapien

    try:
        obj.set_pose(sapien.Pose(p=[float(v) for v in target_pos], q=quat))
        try:
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def lift_and_park_back_to_original(env, obj, start_step: int, end_step: int, cur_step: int, park_xyz) -> None:
    """``statechange.lift_and_drop_objects_back_to_original`` 的停放点版本，时间线逐步相同：

    ``[start, drop)`` 每步传送到 ``park_xyz``，``drop = min(end, start + max(1, (end-start)//2))`` 那一步放回
    窗口首次进入时记下的原位（含原四元数），其余步不动。缓存属性 ``_xhard_park_cache``，与 statechange 的分开。
    """
    start_step, end_step, cur_step = int(start_step), int(end_step), int(cur_step)
    if not hasattr(env, "_xhard_park_cache"):
        env._xhard_park_cache = {}
    cache_all = env._xhard_park_cache
    key = (id(obj), start_step, end_step)
    if cur_step > end_step:
        cache_all.pop(key, None)
        return
    if cur_step < start_step:
        return
    cache = cache_all.get(key)
    if cache is None:
        duration = max(1, end_step - start_step)
        cache = {
            "origin": _pose_xyz(obj).astype(np.float32),
            "quat": _pose_quat(obj),
            "park": np.asarray(park_xyz, dtype=np.float32).reshape(3),
            "drop_step": min(end_step, start_step + max(1, duration // 2)),
        }
        cache_all[key] = cache
    if cur_step < cache["drop_step"]:
        _teleport(obj, cache["park"], cache["quat"])
        return
    if cur_step == cache["drop_step"]:
        _teleport(obj, cache["origin"], cache["quat"])
        return
    cache_all.pop(key, None)


def lift_and_park_onto(env, obj_a, obj_b, start_step: int, end_step: int, cur_step: int, park_xyz) -> None:
    """``statechange.lift_and_drop_objectA_onto_objectB`` 的停放点版本，时间线逐步相同：

    ``[start, end)`` 每步把 ``obj_a`` 传送到 ``park_xyz``；``end`` 那一步放到 ``obj_b`` 当前 XY、``obj_a`` 窗口首次进入时的高度。
    缓存属性 ``_xhard_park_onto_cache``。
    """
    start_step, end_step, cur_step = int(start_step), int(end_step), int(cur_step)
    if not hasattr(env, "_xhard_park_onto_cache"):
        env._xhard_park_onto_cache = {}
    cache_all = env._xhard_park_onto_cache
    key = (id(obj_a), id(obj_b), start_step, end_step)
    if cur_step < start_step:
        return
    if cur_step > end_step:
        cache_all.pop(key, None)
        return
    cache = cache_all.get(key)
    if cache is None:
        cache = {
            "quat_a": _pose_quat(obj_a),
            "park": np.asarray(park_xyz, dtype=np.float32).reshape(3),
            "origin_z": float(_pose_xyz(obj_a)[2]),
        }
        cache_all[key] = cache
    if cur_step < end_step:
        _teleport(obj_a, cache["park"], cache["quat_a"])
        return
    bx, by, _bz = _pose_xyz(obj_b)
    _teleport(obj_a, np.array([bx, by, cache["origin_z"]], dtype=np.float32), cache["quat_a"])
    cache_all.pop(key, None)


def reveal_actors_parked(env, actors: Sequence[Any], *, group: str, start_step: int, end_step: int, cur_step: int) -> None:
    """对一组 actor 逐个做 :func:`lift_and_park_back_to_original`，第 i 个停在 ``xhard_park_point(group, i)``。"""
    for index, actor in enumerate(actors):
        if actor is None:
            continue
        lift_and_park_back_to_original(env, actor, start_step, end_step, cur_step, xhard_park_point(group, index))


def reveal_distractor_bins_parked(env, *, start_step: int, end_step: int, cur_step: int) -> None:
    """``unmask_distractors.reveal_distractor_bins`` 的停放点版本：窗口与落回步不变，每个干扰容器一个停放点。"""
    reveal_actors_parked(env, list(getattr(env, "distractor_bins", None) or []), group="distractor_bin",
                         start_step=start_step, end_step=end_step, cur_step=cur_step)


def park_cubes_onto_bins(env, pairs: Sequence[tuple[Any, Any]], *, group: str, start_step: int, end_step: int,
                         cur_step: int) -> None:
    """对 ``[(cube, bin)]`` 逐对做 :func:`lift_and_park_onto`，第 j 个 cube 停在 ``xhard_park_point(group, j)``。

    Swap 两环境：内环 ``cube_bin_pairs`` 用 ``group="hidden_cube"``，外环 cube 用 ``group="distractor_cube"``。
    """
    for index, (cube, bin_actor) in enumerate(pairs):
        if cube is None or bin_actor is None:
            continue
        lift_and_park_onto(env, cube, bin_actor, start_step, end_step, cur_step, xhard_park_point(group, index))
