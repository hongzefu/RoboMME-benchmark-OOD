"""两个 UnmaskSwap 环境（VideoUnmaskSwap / ButtonUnmaskSwap）xhard 档的共用件。

依据 NEWTASK_RELEASE_V4_PLAN 2.7①③、2.10、2.11、2.21：

* **交换窗口**：原三档每段交换 50 步、首段起点 64；xhard 的 swap 速度 ×1.5（B4）
  ⇒ ``round(50 / 1.5) = 33`` 步一段，起点 64 不变。:func:`scaled_window_steps` 在倍率为 1 时
  **原样返回整数基数**，原三档的调度与改动前逐项相同。
* **干扰容器**（B3 / B13 / H1）：3 个额外容器放在现 region 之外的外环
  ``max(|x|,|y|) ∈ [0.2675, 0.45]``、相机可见处，1~2 个内含「其他颜色」方块（B2 色池），
  单独存 ``distractor_bins``，**不进 ``spawned_bins``**（天然不被选作交换搭档、不进揭示动画、
  不进最近邻），但**必须进碰撞检查**：本模块在 reset 时按确定性的交换序列预演每一段扫掠，
  候选位置与任何一段扫掠相交就拒绝重抽；运行时的初态与扫掠检查另由环境显式并入干扰容器。
* **随机流（N5）**：干扰容器的全部抽样走**独立的专用流**（种子 = 本局 seed + 固定盐），
  主流一次都不多抽——原三档根本不进这里，xhard 的主流取值序列也与不加干扰容器时相同。

本模块的 S5/O4 入口只在新值档分支被调用；原三档不经过这些规划函数。

V5（NEWTASK_RELEASE_V5_PLAN 2.5～2.7，L13～L23）在文件末尾新增一节，两个环境的 xhard 改为调用它：

* **干扰容器**改走统一采样器（``unmask_distractor_sampler``）：V4 环带 ``[0.2675, 0.45]``、10 个、含 cube ``[5, 5]``、
  OBB 间距、精确 8 角点可见、1024 次、带序号命名、``cube_bins`` 映射；仍走独立流 ``distractor_generator(seed)``，
  仍**不进 ``spawned_bins``、不用 ``bin_<i>`` 命名**。
* **外环随内环同步交换**：reset 时规划（:func:`plan_distractor_swaps`），每个内环窗口恰好一次外环交换，发起者按
  ``randperm(count)`` 全体轮转（L17）、不可行按排列顺延（L18）、搭档为干扰容器里的 XY 最近邻、路径全程在画面内且离
  内环容器圆距 ≥ 0.04、BUS 离按钮中心 ≥ 0.122（L19）、lane 0.07（L21）；某窗全不可行整段重抽，最多 16 次。
* **碰撞**：reset 时内环对内环扫掠预判（L20），外环规划用两对联合连续证明 ``check_multi_swap_sweep``；运行时
  :func:`joint_sweep_from_actual` 从实际位姿复核两对联合；三处都开认证预筛（L23）。
* 上面 V4 的 ``XHARD_DISTRACTOR`` / ``sample_distractors`` / ``build_distractors`` / ``visible_on_camera`` 已不被环境调用，
  原样保留（旧单测仍锁着它们的语义），不影响任何行为。
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from .bin_collision import (
    EPS_M,
    LANE_OFFSET,
    PREFILTER_MARGIN_M,
    CollisionRejection,
    ObjectState,
    ShapeSpec,
    _lane_endpoints,
    _prefilter_clearance,
    _prefilter_samples,
    _prefilter_track,
    _prove_pair,
    _Static,
    _swap_movers,
    bin_actor_pose,
    bin_shape_specs,
    check_multi_swap_sweep,
    check_swap_sweep,
    check_swap_sweep_prefiltered,
    object_state_from_actor,
)
from .SceneGenerationError import SceneGenerationError
from .episode_spec import EpisodeSpecError
from . import swap_uniform as _su
from .unmask_distractor_sampler import (
    V5_DISTRACTOR_PRESETS,
    DistractorLayout,
    build_distractor_actors,
    distractor_cube_bin_pairs,
    obstacle_obbs,
    parse_distractor_cfg,
    resample_distractor_layout,
)
from .unmask_distractors import BASE_CAMERA_EYE, BASE_CAMERA_FOV, BASE_CAMERA_TARGET, _camera_axes, bin_geometry
from .xhard import DISTRACTOR_COLORS

# ── 交换窗口（B4）──────────────────────────────────────────────────────────────
#: 首段交换的起点（控制步）；原三档与 xhard 相同，预交换锁定段 [0, 64) 的终点必须等于它。
SWAP_WINDOW_START = 64
#: 原三档每段交换的步数（速度倍率 1）。
SWAP_WINDOW_STEPS = 50
#: xhard 的交换速度倍率（用户原文「swap 速度 x1.5」）。
NEWVALUE_SWAP_SPEED_MULTIPLIER = 1.5


def validate_hidden_bin_selection(indices: Sequence[int], *, permutation_size: int = 3) -> list[int]:
    """复核 M5(b) 的藏物槽位：恰选前三个容器且不得选择恒空的 ``bin_3``。"""
    raw = list(indices)
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in raw):
        raise EpisodeSpecError(f"objects.selected 必须是整数槽位，收到 {raw!r}；M5(b) 要求 bin_3 恒空")
    values = [int(v) for v in raw]
    if (len(values) != permutation_size or len(set(values)) != len(values)
            or any(v < 0 or v >= permutation_size for v in values)):
        raise EpisodeSpecError(
            f"objects.selected 必须是 0..{permutation_size - 1} 的完整不重复选择，收到 {values!r}；M5(b) 要求 bin_3 恒空"
        )
    return values


def scaled_window_steps(base_steps: int, multiplier) -> int:
    """按速度倍率把每段交换步数取整：``round(base / multiplier)``。

    倍率恰为 1 时不做浮点运算、原样返回 ``int(base_steps)``，保证原三档逐字不变。
    """
    m = float(multiplier)
    if not math.isfinite(m) or m <= 0:
        raise ValueError(f"swap_speed_multiplier 必须是正有限数，收到 {multiplier!r}")
    if m == 1.0:
        return int(base_steps)
    steps = int(round(float(base_steps) / m))
    if steps < 1:
        raise ValueError(f"倍率 {multiplier} 使每段交换步数小于 1")
    return steps


# ── 干扰容器（B3 / B13）────────────────────────────────────────────────────────
#: V4 旧快照中的三容器干扰配置，仅用于 release-aware 旧版导出。
LEGACY_V4_DISTRACTOR = {
    # 额外容器个数（B3：3 个）
    "count": 3,
    # 其中装方块的个数，闭区间（B13：3 个里 1~2 个有）
    "with_cube_range": [1, 2],
    # 外环 max(|x|,|y|) 的上下界（B13 实测推荐值）
    "ring_half_extent": [0.2675, 0.45],
    # 干扰容器外接圆与其他对象外接圆之间的最小间隙，米（B13 推导外环下界时用的 min_gap）
    "min_gap": 0.04,
    # 方块颜色池（B2：黄／青／品红，全局共用）
    "colors": [item["name"] for item in DISTRACTOR_COLORS],
}

#: 干扰容器每个的拒绝采样预算；用尽即抛 SceneGenerationError（2.2④：不许静默截断）。
DISTRACTOR_MAX_TRIALS = 512
#: 专用随机流的种子盐：种子 = seed + 盐，与主流完全分开（N5）。
DISTRACTOR_STREAM_SALT = 0x5D157AC7

# 相机可见 ∩ 桌面（B13 实测表，前视相机 eye=[0.3,0,0.4]、target=[0,0,-0.2]、fov 90°）：
# x ≤ 0.43；可见 y 半宽随 x 线性收窄，表中 7 个点可由 0.49 − 0.45·x 逐点复现（误差 ≤ 0.005）。
VISIBLE_X_MAX = 0.43
VISIBLE_Y_AT_X0 = 0.49
VISIBLE_Y_SLOPE = 0.45


def bin_footprint_radius(cube_half_size: float) -> float:
    """容器外廓（正方形）外接圆半径；与 build_bin 同源：外廓半边 = (2.5·h + 0.005)/2。"""
    half = (cube_half_size * 2.5 + 0.005) * 0.5
    return half * math.sqrt(2.0)


def visible_on_camera(x: float, y: float, radius: float) -> bool:
    """以外接圆保守判断整个容器落在前视相机可见 ∩ 桌面内。"""
    far_x = x + radius
    if far_x > VISIBLE_X_MAX:
        return False
    return abs(y) + radius <= VISIBLE_Y_AT_X0 - VISIBLE_Y_SLOPE * far_x


def distractor_generator(seed: int) -> torch.Generator:
    """干扰容器专用随机流（N5：主流一次都不多抽）。"""
    generator = torch.Generator()
    generator.manual_seed((int(seed) + DISTRACTOR_STREAM_SALT) % (2**63))
    return generator


def solve_hold_obj_xhard(env, planner, static_steps: int) -> None:
    """xhard 专用的原地等待：与 ``solve_hold_obj(close=False)`` 同语义，只吞 ``AttributeError``。

    共享函数 ``utils/subgoal_planner_func.py::solve_hold_obj`` 用裸 ``except:`` 包住
    ``planner.open_gripper()``：xhard 打开运行时碰撞检查（H1）后，``env.step`` 在交换开始时抛出的
    ``BinCollisionError`` 会被吞掉，``elapsed_steps`` 不前进，等待循环永不结束（本机实测挂满外部超时）。
    这里让碰撞拒绝（及其他一切非 ``AttributeError`` 异常）原样上抛，由 ``_worker`` 归为任务性失败。
    共享函数按 N12 不就地修；原三档继续用原函数，新值档使用本函数。
    """
    start_step = int(getattr(env, "elapsed_steps", 0))
    target_step = start_step + static_steps
    while int(getattr(env, "elapsed_steps", 0)) < target_step:
        try:
            planner.open_gripper()
        except AttributeError:
            pass
    return None


def predict_swap_sweeps(env, partner_axes: Sequence[int]) -> list[tuple[ObjectState, ObjectState]]:
    """按 ``step`` 的同一语义预演全部交换段，返回每段起态 ``(发起者, 搭档)``。

    发起者取 ``swap_pair{k}_idx1``；搭档是交换开始时 XY 最近的另一个 ``spawned_bins``
    （严格小于、并列取生成序靠前者，与 ``step`` 的扫描一致）；每段结束后两者位姿互换
    （``swap_flat_two_lane`` 的终态：位置与四元数都换到对方的起态）。干扰容器不在
    ``spawned_bins`` 里，因此不影响搭档选择——预演结果与是否有干扰容器无关。
    """
    bins = list(env.spawned_bins)
    states = [object_state_from_actor(actor, f"bin_{index}") for index, actor in enumerate(bins)]
    positions = [np.asarray(env._get_actor_position(actor), dtype=np.float32) for actor in bins]
    axes = list(partner_axes)
    sweeps = []
    for k in range(int(env.swap_times)):
        initiator = getattr(env, f"swap_pair{k + 1}_idx1")
        a = next(index for index, actor in enumerate(bins) if actor is initiator)
        b, best = None, float("inf")
        for index, position in enumerate(positions):
            if index == a:
                continue
            dist = np.linalg.norm(positions[a][axes] - position[axes])
            if dist < best:
                b, best = index, dist
        if b is None:
            break
        sweeps.append((states[a], states[b]))
        state_a, state_b = states[a], states[b]
        states[a] = ObjectState(name=state_a.name, p=state_b.p.copy(), q=state_b.q.copy(), shapes=state_a.shapes)
        states[b] = ObjectState(name=state_b.name, p=state_a.p.copy(), q=state_a.q.copy(), shapes=state_b.shapes)
        positions[a], positions[b] = positions[b].copy(), positions[a].copy()
    return sweeps


def sample_distractors(
    *,
    generator: torch.Generator,
    cfg: dict,
    obstacles: Sequence[tuple[Sequence[float], float]],
    sweeps: Sequence[tuple[ObjectState, ObjectState]],
    recorder,
    cube_half_size: float,
) -> dict[str, Any]:
    """在外环里拒绝采样全部干扰容器位置，返回 ``{"placements": [...], "cube_colors": [...]}``。

    ``obstacles``：已在场对象的 ``(xy, 外接圆半径)``（容器、按钮等）；
    ``sweeps``：:func:`predict_swap_sweeps` 的结果，候选与任一段扫掠相交即拒绝（H1）。
    每个取值点都经 ``recorder``（SpecRecorder）走一遍；请求数与实际数不等直接抛错。
    """
    lo, hi = (int(v) for v in cfg["with_cube_range"])
    count = int(cfg["count"])
    if not 0 <= lo <= hi <= count:
        raise ValueError(f"with_cube_range {cfg['with_cube_range']} 必须落在 [0, count={count}] 内")
    inner, outer = (float(v) for v in cfg["ring_half_extent"])
    min_gap = float(cfg["min_gap"])
    pool = list(cfg["colors"])
    known = {item["name"] for item in DISTRACTOR_COLORS}
    if not set(pool) <= known:
        raise ValueError(f"干扰色 {pool} 超出全局色池 {sorted(known)}")

    n_with_cube = recorder.value(
        "objects.distractors.n_with_cube",
        int(torch.randint(lo, hi + 1, (1,), generator=generator).item()),
        decision_key="xhard.distractor.with_cube_range",
    )
    order = torch.randperm(len(pool), generator=generator).tolist()
    cube_colors = recorder.value(
        "objects.distractors.cube_colors",
        [pool[i] for i in order][: int(n_with_cube)],
        decision_key="xhard.distractor.colors",
    )

    radius = bin_footprint_radius(cube_half_size)
    shapes = bin_shape_specs(cube_half_size)
    occupied = [(np.asarray(xy, dtype=np.float64)[:2], float(r)) for xy, r in obstacles]
    recorder.record("layout.distractors_requested", count)
    placements = []
    for i in range(count):
        chosen = None
        for _trial in range(DISTRACTOR_MAX_TRIALS):
            x = float((torch.rand(1, generator=generator).item() * 2.0 - 1.0) * outer)
            y = float((torch.rand(1, generator=generator).item() * 2.0 - 1.0) * outer)
            yaw = float(torch.rand(1, generator=generator).item() * 90.0)
            if max(abs(x), abs(y)) < inner:
                continue
            if not visible_on_camera(x, y, radius):
                continue
            here = np.array([x, y], dtype=np.float64)
            if any(np.linalg.norm(here - xy) < r + radius + min_gap for xy, r in occupied):
                continue
            p, q = bin_actor_pose([x, y], yaw, cube_half_size)
            candidate = ObjectState(name=f"distractor_bin_{i}", p=p, q=q, shapes=shapes)
            if any(check_swap_sweep(a, b, [candidate], sweep_index=k, stage="distractor")[1] is not None
                   for k, (a, b) in enumerate(sweeps)):
                continue
            chosen = (x, y, yaw)
            break
        if chosen is None:
            raise SceneGenerationError(
                f"干扰容器 {i} 在 {DISTRACTOR_MAX_TRIALS} 次尝试内找不到合法位置（外环 {inner}~{outer}）"
            )
        x, y, yaw = recorder.value(f"layout.distractors.{i}", list(chosen))
        placements.append({"xy": [float(x), float(y)], "yaw_deg": float(yaw)})
        occupied.append((np.array([x, y], dtype=np.float64), radius))
    recorder.record("layout.distractors_placed", len(placements))
    if len(placements) != count:
        raise SceneGenerationError(f"干扰容器请求 {count} 个、实际 {len(placements)} 个")
    return {"placements": placements, "cube_colors": list(cube_colors)}


def build_distractors(env, layout: dict, build_bin, spawn_fixed_cube, cube_divisor: float = 1.2):
    """按 :func:`sample_distractors` 的结果建干扰容器与方块；前 ``len(cube_colors)`` 个容器装方块。

    命名一律 ``distractor_bin_<i>`` / ``distractor_cube_<色名>``，**不设 ``bin_<i>`` 属性**，
    避免被按 ``bin_<i>`` 扫描的揭示／交换逻辑误收。
    """
    rgba = {item["name"]: item["rgba"] for item in DISTRACTOR_COLORS}
    bins, cubes = [], []
    for i, entry in enumerate(layout["placements"]):
        x, y = entry["xy"]
        bins.append(build_bin(env, callsign=f"distractor_bin_{i}", position=[x, y, 0.002],
                              z_rotation_deg=entry["yaw_deg"]))
    for i, color in enumerate(layout["cube_colors"]):
        x, y = layout["placements"][i]["xy"]
        cubes.append(spawn_fixed_cube(
            env,
            position=[x, y],
            half_size=env.cube_half_size / cube_divisor,
            color=rgba[color],
            name_prefix=f"distractor_cube_{color}",
            yaw=0.0,
            dynamic=True,
        ))
    return bins, cubes


# ════════════════════════════════════════════════════════════════════════════════
# V5（NEWTASK_RELEASE_V5_PLAN 2.5～2.7）：统一采样器 10 个干扰容器 + 外环随内环同步交换
# ════════════════════════════════════════════════════════════════════════════════
#: 本节服务的两个环境。
V5_SWAP_TASKS = ("VideoUnmaskSwap", "ButtonUnmaskSwap")
#: 外环发起者规则（L17 b）：``perm = randperm(count)``，第 k 窗从 ``perm[k % count]`` 起。
OUTER_INITIATOR_RULE = "permutation_cycle"
#: 某窗第一候选不可行时的回退（L18 a）：按排列顺延到下一个发起者，全不行才整段重抽。
OUTER_FALLBACK_RULE = "next_in_permutation"
#: 外环搭档规则（口径 5）：干扰容器里 XY 两轴欧氏距离最近者，严格 <，平局取序号小；reset 时规划。
OUTER_PARTNER_RULE = {
    "selection": "nearest",
    "position_axes": [0, 1],
    "tie_break": "first_in_index_order",
    "population": "distractor_bins",
    "resolve_at": "reset_plan",
}


#: V6（计划 2.2 外环 O4，M7(a)）：每窗在可行对象对里取「两者参与次数 max、再 sum」最小者，平局均匀，
#: 禁止立即撤销（除非别无选择）；放置后序号按 ``randperm(count)`` 重排（跨局均匀）。路径四道判定不放宽。
OUTER_BALANCED_RULE = "balanced_greedy_o4"
OUTER_BALANCED_FALLBACK = "undo_only_if_no_other_feasible"
OUTER_BALANCED_PARTNER_RULE = {
    "selection": "balanced_greedy_with_initiator",
    "score": "max_then_sum_of_participation",
    "tie_break": "uniform_local_generator",
    "population": "distractor_bins",
    "relabel": "randperm_count_after_placement",
    "resolve_at": "reset_plan",
}


def v5_distractor_cfg(task: str) -> dict:
    """``decision.xhard.distractor`` 的 V5 申报值：统一采样器的预设（L16 b：V4 环带、10 个、含 cube [5,5]）。"""
    if task not in V5_SWAP_TASKS:
        raise ValueError(f"只支持 {V5_SWAP_TASKS}，收到 {task!r}")
    return copy.deepcopy(V5_DISTRACTOR_PRESETS[task])


def v6_distractor_cfg(task: str, count: int) -> dict:
    """按 V6 梯度调整外环干扰容器数量；其余 V5 区域、颜色、间距和采样规则逐项沿用。"""
    if task not in V5_SWAP_TASKS:
        raise ValueError(f"只支持 {V5_SWAP_TASKS}，收到 {task!r}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2 or count % 2:
        raise ValueError(f"外环干扰容器数量必须是 ≥2 的偶数，收到 {count!r}")
    cfg = v5_distractor_cfg(task)
    cfg["count"] = count
    cfg["cube_count_range"] = [count // 2, count // 2]
    return cfg


def v5_distractor_swap_cfg(task: str) -> dict:
    """``decision.xhard.distractor_swap`` 的 V5 申报值（计划 2.5 伪码的配置块）。

    相对伪码多三个显式键（实施方自决，均不改变设计意图）：``path_samples``（路径约束的采样点数，401，与认证预筛
    同一密度）、``plan_pad_m``（外环规划时每个干扰容器的碰撞盒外扩 5 mm，吸收运行时位姿与名义的偏差，2.5「关键设计点」）；
    VUS 没有按钮，``min_button_center_dist_m`` 取 ``None``。
    """
    if task not in V5_SWAP_TASKS:
        raise ValueError(f"只支持 {V5_SWAP_TASKS}，收到 {task!r}")
    return {
        "enabled": True,
        "initiator_rule": OUTER_INITIATOR_RULE,
        "fallback": OUTER_FALLBACK_RULE,
        "partner": copy.deepcopy(OUTER_PARTNER_RULE),
        "lane_offset": 0.07,
        "smooth": True,
        "path_constraints": {
            "camera_visible": True,
            "min_inner_circle_clearance_m": 0.04,
            "min_button_center_dist_m": 0.122 if task == "ButtonUnmaskSwap" else None,
        },
        "path_samples": 401,
        "plan_pad_m": 0.005,
        "layout_max_attempts": 16,
    }


@dataclass(frozen=True)
class DistractorSwapConfig:
    """校验过的外环交换配置。"""

    enabled: bool
    lane_offset: float
    camera_visible: bool
    min_inner_circle_clearance_m: float
    min_button_center_dist_m: float | None
    path_samples: int
    plan_pad_m: float
    layout_max_attempts: int
    #: V6（2.2 外环 O4）：``OUTER_INITIATOR_RULE``（V5 排列轮转 + 最近邻）或 ``OUTER_BALANCED_RULE``（均衡贪心 + 序号重排）
    rule: str = "permutation_cycle"


DISTRACTOR_SWAP_CFG_KEYS = ("enabled", "initiator_rule", "fallback", "partner", "lane_offset", "smooth",
                            "path_constraints", "path_samples", "plan_pad_m", "layout_max_attempts")


def parse_distractor_swap_cfg(cfg: dict | DistractorSwapConfig) -> DistractorSwapConfig:
    """校验 ``decision.xhard.distractor_swap``；规则名只认计划定下的那一种，其余数值做范围检查。"""
    if isinstance(cfg, DistractorSwapConfig):
        return cfg
    keys = set(cfg)
    missing = [k for k in DISTRACTOR_SWAP_CFG_KEYS if k not in keys]
    extra = sorted(keys - set(DISTRACTOR_SWAP_CFG_KEYS))
    if missing or extra:
        raise ValueError(f"distractor_swap 键不符：缺 {missing}，多 {extra}")
    if cfg["initiator_rule"] == OUTER_BALANCED_RULE:
        # V6（计划 2.2 外环 O4）：发起者与搭档一起在可行对里按均衡贪心选，回退与搭档规则随之换成 V6 的申报值
        if cfg["fallback"] != OUTER_BALANCED_FALLBACK:
            raise ValueError(f"fallback 只支持 {OUTER_BALANCED_FALLBACK!r}，收到 {cfg['fallback']!r}")
        if dict(cfg["partner"]) != OUTER_BALANCED_PARTNER_RULE:
            raise ValueError(f"partner 只支持 {OUTER_BALANCED_PARTNER_RULE}，收到 {cfg['partner']}")
    else:
        if cfg["initiator_rule"] != OUTER_INITIATOR_RULE:
            raise ValueError(f"initiator_rule 只支持 {OUTER_INITIATOR_RULE!r}，收到 {cfg['initiator_rule']!r}")
        if cfg["fallback"] != OUTER_FALLBACK_RULE:
            raise ValueError(f"fallback 只支持 {OUTER_FALLBACK_RULE!r}，收到 {cfg['fallback']!r}")
        if dict(cfg["partner"]) != OUTER_PARTNER_RULE:
            raise ValueError(f"partner 只支持 {OUTER_PARTNER_RULE}，收到 {cfg['partner']}")
    lane = float(cfg["lane_offset"])
    if lane != LANE_OFFSET:
        # 联合证明（_Mover.pose_at）按 bin_collision.LANE_OFFSET 复算路径，二者必须相同才有证明意义
        raise ValueError(f"lane_offset 必须等于碰撞判据的 LANE_OFFSET={LANE_OFFSET}，收到 {lane}")
    if cfg["smooth"] is not True:
        raise ValueError("smooth 必须为 True（与内环同为 smoothstep）")
    pc = cfg["path_constraints"]
    if set(pc) != {"camera_visible", "min_inner_circle_clearance_m", "min_button_center_dist_m"}:
        raise ValueError(f"path_constraints 键不符：{sorted(pc)}")
    clearance = float(pc["min_inner_circle_clearance_m"])
    button = pc["min_button_center_dist_m"]
    button = None if button is None else float(button)
    samples = int(cfg["path_samples"])
    pad = float(cfg["plan_pad_m"])
    attempts = int(cfg["layout_max_attempts"])
    if not (math.isfinite(clearance) and clearance >= 0.0):
        raise ValueError(f"min_inner_circle_clearance_m 非法：{clearance}")
    if button is not None and not (math.isfinite(button) and button >= 0.0):
        raise ValueError(f"min_button_center_dist_m 非法：{button}")
    if samples < 2:
        raise ValueError(f"path_samples 必须 ≥ 2，收到 {samples}")
    if not (math.isfinite(pad) and pad >= 0.0):
        raise ValueError(f"plan_pad_m 非法：{pad}")
    if attempts < 1:
        raise ValueError(f"layout_max_attempts 必须 ≥ 1，收到 {attempts}")
    return DistractorSwapConfig(bool(cfg["enabled"]), lane, bool(pc["camera_visible"]), clearance, button,
                                samples, pad, attempts, str(cfg["initiator_rule"]))


# ── 内环预演 ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class InnerWindow:
    """第 k 个内环窗口的预演：发起者 ``a``、搭档 ``b``（``spawned_bins`` 序号）与窗口起点时全部内环容器的名义状态。"""

    a: int
    b: int
    states: tuple[ObjectState, ...]


def predict_inner_windows_from_states(states: Sequence[ObjectState], positions: Sequence[Sequence[float]],
                                      initiators: Sequence[int], partner_axes: Sequence[int]) -> list[InnerWindow]:
    """与 :func:`predict_swap_sweeps` 逐字同语义（float32 位置、严格 <、平局取生成序靠前、每段后位姿互换），
    额外返回每窗起点的全部内环状态（联合证明要把其余内环容器当静止旁观者）。"""
    states = list(states)
    pos = [np.asarray(p, dtype=np.float32) for p in positions]
    axes = list(partner_axes)
    windows: list[InnerWindow] = []
    for a in initiators:
        a = int(a)
        b, best = None, float("inf")
        for index, position in enumerate(pos):
            if index == a:
                continue
            dist = np.linalg.norm(pos[a][axes] - position[axes])
            if dist < best:
                b, best = index, dist
        if b is None:
            break
        windows.append(InnerWindow(a=a, b=int(b), states=tuple(states)))
        state_a, state_b = states[a], states[b]
        states[a] = ObjectState(name=state_a.name, p=state_b.p.copy(), q=state_b.q.copy(), shapes=state_a.shapes)
        states[b] = ObjectState(name=state_b.name, p=state_a.p.copy(), q=state_a.q.copy(), shapes=state_b.shapes)
        pos[a], pos[b] = pos[b].copy(), pos[a].copy()
    return windows


def predict_inner_windows(env, partner_axes: Sequence[int]) -> list[InnerWindow]:
    """从环境的 ``spawned_bins`` 与 ``swap_pair{k}_idx1`` 预演全部内环窗口（位姿读实际 actor）。"""
    bins = list(env.spawned_bins)
    states = [object_state_from_actor(actor, f"bin_{index}") for index, actor in enumerate(bins)]
    positions = [np.asarray(env._get_actor_position(actor), dtype=np.float32) for actor in bins]
    initiators = []
    for k in range(int(env.swap_times)):
        initiator = getattr(env, f"swap_pair{k + 1}_idx1")
        initiators.append(next(index for index, actor in enumerate(bins) if actor is initiator))
    return predict_inner_windows_from_states(states, positions, initiators, partner_axes)


def prejudge_inner_windows(windows: Sequence[InnerWindow], *, prefilter: bool = True,
                           stats: dict | None = None) -> tuple[int, CollisionRejection] | None:
    """L20：reset 时逐窗预判内环对内环扫掠（其余内环容器静止），返回第一处拒绝 ``(k, 证据)`` 或 ``None``。

    判定与 V4 运行时的 ``check_swap_sweep`` 相同（单对时 ``check_swap_sweep_prefiltered`` 与之判定逐位一致），
    认证预筛只跳过已证明分离的对（L23）。
    """
    for k, window in enumerate(windows):
        others = [state for j, state in enumerate(window.states) if j not in (window.a, window.b)]
        _gap, rejection = check_swap_sweep_prefiltered(
            window.states[window.a], window.states[window.b], others,
            sweep_index=k, stage="inner_prejudge", prefilter=prefilter, stats=stats,
        )
        if rejection is not None:
            return k, rejection
    return None


# ── 几何小件 ───────────────────────────────────────────────────────────────────
def padded_bin_shapes(cube_half_size: float, pad: float) -> tuple[ShapeSpec, ...]:
    """容器 6 个盒体各向外扩 ``pad``（只用于 reset 规划的余量，运行时复核仍用真实碰撞盒）。"""
    return tuple(
        ShapeSpec(shape.local_p.copy(), shape.local_q.copy(), shape.half + float(pad))
        for shape in bin_shape_specs(cube_half_size)
    )


def distractor_bin_state(index: int, x: float, y: float, yaw_deg: float, cube_half_size: float,
                         shapes: Sequence[ShapeSpec]) -> ObjectState:
    """由布局值 ``(x, y, yaw)`` 构造第 ``index`` 个干扰容器的名义状态（位姿复刻 ``build_bin``）。"""
    p, q = bin_actor_pose([float(x), float(y)], float(yaw_deg), cube_half_size)
    return ObjectState(name=f"distractor_bin_{int(index)}", p=p, q=q, shapes=tuple(shapes))


def path_samples(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(n))


def lane_center_paths(xy_a, xy_b, samples: np.ndarray, lane: float = LANE_OFFSET) -> tuple[np.ndarray, np.ndarray]:
    """``swap_flat_two_lane`` 两个交换者的中心 XY 轨迹（与 ``bin_collision._Mover.pose_at`` 同式）。"""
    a = np.asarray(xy_a, dtype=np.float64)[:2]
    b = np.asarray(xy_b, dtype=np.float64)[:2]
    delta, normal = _lane_endpoints(a, b)
    s = np.asarray(samples, dtype=np.float64)[:, None]
    offset = float(lane) * np.sin(np.pi * s)
    return a + delta * s + normal * offset, b - delta * s - normal * offset


def bins_visible_many(xy: np.ndarray, cube_half_size: float) -> np.ndarray:
    """向量化的精确 8 角点可见判据：与 ``unmask_distractor_sampler.bin_visible``（``visible_in_camera(bin_corners(...))``）
    逐点等价（单测用随机点核对）。``xy`` 形状 ``(N, 2)``，返回 ``(N,)`` 布尔。"""
    _half, reach, height = bin_geometry(cube_half_size)
    corners = np.array([(sx * reach, sy * reach, z) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for z in (0.0, height)])
    eye, forward, right, up = _camera_axes(BASE_CAMERA_EYE, BASE_CAMERA_TARGET)
    tan_half = np.tan(BASE_CAMERA_FOV / 2.0)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    points = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)[:, None, :] + corners[None]
    d = points - eye
    depth = d @ forward
    ok = depth > 1e-6
    safe = np.where(ok, depth, 1.0)
    ok &= np.abs((d @ right) / safe) / tan_half <= 1.0
    ok &= np.abs((d @ up) / safe) / tan_half <= 1.0
    return ok.all(axis=1)


def _min_pointwise(a: np.ndarray, b: np.ndarray) -> float:
    """两条同参数化轨迹（或轨迹与一个点）在同一 ``s`` 上距离的最小值。"""
    return float(np.min(np.linalg.norm(a - b, axis=-1)))


# ── H1：候选干扰容器 × 预演的内环扫掠（只证「候选 × 两个交换者」）──────────────────────
class InnerSweepGuard:
    """干扰容器放置时的额外拒绝：候选（静止）与任一窗的内环交换者扫掠相交即拒绝（V4 H1 的延续）。

    与 ``check_multi_swap_sweep([(a, b)], [候选])`` 里「静止物 × 交换者」那两对的判定逐项相同（同一包围球粗筛、
    同一认证预筛、同一 ``_prove_pair``），只是不再重复证明内环对自身（它已在 L20 预判里证过）——否则每个候选
    都要把内环对重证一遍，reset 慢一个数量级。每窗的交换者与其预筛轨迹在构造时算一次。
    """

    def __init__(self, windows: Sequence[InnerWindow]):
        self._samples = _prefilter_samples()
        self._step = float(self._samples[1] - self._samples[0])
        self._windows = []
        for k, window in enumerate(windows):
            mover_a, mover_b = _swap_movers(window.states[window.a], window.states[window.b])
            self._windows.append((k, [(mover_a, _prefilter_track(mover_a, self._samples)),
                                      (mover_b, _prefilter_track(mover_b, self._samples))]))

    def first_rejection(self, candidate: ObjectState, stage: str = "distractor") -> CollisionRejection | None:
        static = _Static(name=candidate.name, shapes=candidate.shapes, radii=candidate.radii,
                         p=candidate.p.copy(), q=candidate.q.copy())
        track = _prefilter_track(static, self._samples)
        center_r, radius_r = static.bounding_sphere()
        for k, movers in self._windows:
            for mover, mover_track in movers:
                center_l, radius_l = mover.bounding_sphere()
                if float(np.linalg.norm(center_l - center_r)) - radius_l - radius_r > EPS_M:
                    continue
                if mover_track is not None and track is not None:
                    bound = _prefilter_clearance(mover_track, track, self._step)
                    if math.isfinite(bound) and bound > PREFILTER_MARGIN_M:
                        continue
                for ia in range(len(mover.shapes)):
                    for ib in range(len(static.shapes)):
                        _gap, rejection = _prove_pair(mover, static, ia, ib, stage=stage, sweep_index=k)
                        if rejection is not None:
                            return rejection
        return None


# ── 外环规划 ───────────────────────────────────────────────────────────────────
def nearest_distractor(positions_xy: Sequence[Sequence[float]], initiator: int) -> int | None:
    """干扰容器里离 ``initiator`` 最近者（XY、float32、严格 <、平局取序号小），与内环 ``step`` 的扫描同语义。"""
    pos = [np.asarray(p, dtype=np.float32)[:2] for p in positions_xy]
    best, best_dist = None, float("inf")
    for index, position in enumerate(pos):
        if index == initiator:
            continue
        dist = np.linalg.norm(pos[initiator] - position)
        if dist < best_dist:
            best, best_dist = index, dist
    return best


def evaluate_outer_candidate(
    k: int,
    window: InnerWindow,
    o: int,
    p: int,
    outer_states: Sequence[ObjectState],
    *,
    cfg: DistractorSwapConfig,
    cube_half_size: float,
    buttons_xy: Sequence[Sequence[float]] = (),
    stats: dict | None = None,
) -> tuple[bool, str | None, CollisionRejection | None]:
    """第 k 窗候选外环对 ``(o, p)`` 是否可行，返回 ``(可行, 拒绝原因, 碰撞证据)``。按便宜到贵依次查（L19）：

    1. ``vis``：o、p 两条中心路径全程精确可见（``camera_visible`` 为真时）；
    2. ``btn``：（仅 BUS）两条路径离每个按钮中心 ≥ ``min_button_center_dist_m``；
    3. ``inner_clear``：两条路径离内环（本窗两条交换路径的同一时刻位置、其余内环容器的静止位置）的中心距减去两个外接圆
       半径 ≥ ``min_inner_circle_clearance_m``；
    4. ``exact``：``check_multi_swap_sweep([内环对 k, (o, p)], 其余全部静止)`` 通过（干扰容器的碰撞盒已按 ``plan_pad_m`` 外扩）。
    """
    samples = path_samples(cfg.path_samples)
    path_o, path_p = lane_center_paths(outer_states[o].p[:2], outer_states[p].p[:2], samples, cfg.lane_offset)
    if cfg.camera_visible and not bins_visible_many(np.vstack([path_o, path_p]), cube_half_size).all():
        return False, "vis", None
    if cfg.min_button_center_dist_m is not None:
        for button in buttons_xy:
            center = np.asarray(button, dtype=np.float64)[None, :2]
            if min(_min_pointwise(path_o, center), _min_pointwise(path_p, center)) < cfg.min_button_center_dist_m:
                return False, "btn", None
    inner = window.states
    path_a, path_b = lane_center_paths(inner[window.a].p[:2], inner[window.b].p[:2], samples, LANE_OFFSET)
    clearance = min(_min_pointwise(path_o, path_a), _min_pointwise(path_o, path_b),
                    _min_pointwise(path_p, path_a), _min_pointwise(path_p, path_b))
    for j, state in enumerate(inner):
        if j in (window.a, window.b):
            continue
        clearance = min(clearance, _min_pointwise(path_o, state.p[None, :2]), _min_pointwise(path_p, state.p[None, :2]))
    if clearance - 2.0 * bin_footprint_radius(cube_half_size) < cfg.min_inner_circle_clearance_m:
        return False, "inner_clear", None
    bystanders = [state for j, state in enumerate(inner) if j not in (window.a, window.b)]
    bystanders += [state for j, state in enumerate(outer_states) if j not in (o, p)]
    if stats is not None:
        stats["joint_calls"] = stats.get("joint_calls", 0) + 1
    _gap, rejection = check_multi_swap_sweep(
        [(inner[window.a], inner[window.b]), (outer_states[o], outer_states[p])], bystanders,
        sweep_index=k, stage="outer_plan", stats=stats,
    )
    if rejection is not None:
        return False, "exact", rejection
    return True, None, None


@dataclass
class OuterSwapPlan:
    """一次外环规划的结果。``pairs[k] = (o, p)``；``fallbacks[k]`` 为第 k 窗被接受的发起者在排列里顺延了几位（0 = 第一候选）。"""

    ok: bool
    perm: list[int]
    pairs: list[tuple[int, int]] = field(default_factory=list)
    fallbacks: list[int] = field(default_factory=list)
    fail_window: int | None = None
    reasons: dict[str, int] = field(default_factory=dict)
    final_xy: list[list[float]] = field(default_factory=list)


def plan_distractor_swaps(
    layout: DistractorLayout,
    perm: Sequence[int],
    windows: Sequence[InnerWindow],
    *,
    cfg: DistractorSwapConfig,
    cube_half_size: float,
    buttons_xy: Sequence[Sequence[float]] = (),
    stats: dict | None = None,
) -> OuterSwapPlan:
    """reset 时的外环规划（计划 2.5 伪码的窗口循环）。纯几何，不抽随机数。

    对每个内环窗口 k 依次试 ``o = perm[(k + j) % count]``（j = 0..count−1），搭档 ``p`` = 干扰容器中 XY 最近邻；
    第一个满足 :func:`evaluate_outer_candidate` 的 ``(o, p)`` 被接受，名义上对换二者位姿后进入下一窗；
    某窗全部不可行即返回 ``ok=False``（调用方整段重抽干扰布局）。
    """
    count = layout.count
    perm = [int(v) for v in perm]
    if sorted(perm) != list(range(count)):
        raise ValueError(f"外环发起者排列 {perm} 不是 0..{count - 1} 的排列")
    shapes = padded_bin_shapes(cube_half_size, cfg.plan_pad_m)
    outer = [distractor_bin_state(i, x, y, yaw, cube_half_size, shapes) for i, (x, y, yaw) in enumerate(layout.bins)]
    plan = OuterSwapPlan(ok=False, perm=perm)
    for k, window in enumerate(windows):
        chosen = None
        for j in range(count):
            o = perm[(k + j) % count]
            p = nearest_distractor([state.p[:2] for state in outer], o)
            if p is None:
                continue
            ok, reason, _rejection = evaluate_outer_candidate(
                k, window, o, p, outer, cfg=cfg, cube_half_size=cube_half_size, buttons_xy=buttons_xy, stats=stats,
            )
            if ok:
                chosen = (o, p, j)
                break
            plan.reasons[reason] = plan.reasons.get(reason, 0) + 1
        if chosen is None:
            plan.fail_window = k
            return plan
        o, p, j = chosen
        plan.pairs.append((o, p))
        plan.fallbacks.append(j)
        state_o, state_p = outer[o], outer[p]
        outer[o] = ObjectState(name=state_o.name, p=state_p.p.copy(), q=state_p.q.copy(), shapes=state_o.shapes)
        outer[p] = ObjectState(name=state_p.name, p=state_o.p.copy(), q=state_o.q.copy(), shapes=state_p.shapes)
    plan.ok = True
    plan.final_xy = [[float(v) for v in state.p[:2]] for state in outer]
    return plan


def verify_distractor_swap_plan(
    layout: DistractorLayout,
    pairs: Sequence[Sequence[int]],
    windows: Sequence[InnerWindow],
    *,
    cfg: DistractorSwapConfig,
    cube_half_size: float,
    buttons_xy: Sequence[Sequence[float]] = (),
) -> list[str]:
    """独立复核一份外环交换对（不重新选搭档）：窗口数、每窗 p 确为 o 的最近邻、四条可行条件都成立。返回违反项。"""
    problems: list[str] = []
    if len(pairs) != len(windows):
        problems.append(f"外环交换 {len(pairs)} 次 ≠ 内环窗口 {len(windows)} 个")
    shapes = padded_bin_shapes(cube_half_size, cfg.plan_pad_m)
    outer = [distractor_bin_state(i, x, y, yaw, cube_half_size, shapes) for i, (x, y, yaw) in enumerate(layout.bins)]
    for k, (window, pair) in enumerate(zip(windows, pairs)):
        o, p = (int(v) for v in pair)
        if not (0 <= o < len(outer) and 0 <= p < len(outer)) or o == p:
            problems.append(f"第 {k} 窗外环对 {pair} 越界或重复")
            continue
        if nearest_distractor([state.p[:2] for state in outer], o) != p:
            problems.append(f"第 {k} 窗 distractor_bin_{p} 不是 distractor_bin_{o} 的最近邻")
        ok, reason, rejection = evaluate_outer_candidate(k, window, o, p, outer, cfg=cfg, cube_half_size=cube_half_size,
                                                         buttons_xy=buttons_xy)
        if not ok:
            problems.append(f"第 {k} 窗外环对 ({o},{p}) 不可行：{reason}"
                            + ("" if rejection is None else f" {rejection.summary()}"))
        state_o, state_p = outer[o], outer[p]
        outer[o] = ObjectState(name=state_o.name, p=state_p.p.copy(), q=state_p.q.copy(), shapes=state_o.shapes)
        outer[p] = ObjectState(name=state_p.name, p=state_o.p.copy(), q=state_o.q.copy(), shapes=state_p.shapes)
    return problems


# ── reset 入口（两个环境共用；仍由 _spawn_xhard_distractors 调用、仍是 _load_scene 的最后一句）──────────
@dataclass
class SwapDistractorResult:
    """reset 规划与建 actor 的产物，由环境挂到实例属性上。"""

    bins: list
    cubes: list
    cube_bin_pairs: list
    layout: DistractorLayout
    pairs: list[tuple[int, int]]
    predicted_inner_pairs: list[tuple[int, int]]
    timing: dict[str, float]
    stats: dict[str, int]


def plan_swap_distractors(
    *,
    windows: Sequence[InnerWindow],
    obstacles: Sequence[Any],
    buttons_xy: Sequence[Sequence[float]],
    generator: torch.Generator,
    recorder,
    distractor_cfg: dict,
    swap_cfg: dict,
    cube_half_size: float,
    guard_windows: Sequence[InnerWindow] | None = None,
    difficulty: str = "xhard4",
) -> tuple[DistractorLayout, list[tuple[int, int]], dict[str, float], dict[str, int]]:
    """纯几何的 reset 规划（不建 actor，便于离线复算与单测）：L20 预判 → 整段重抽放置 + 外环规划 → 记录。

    ``obstacles`` 是 2D OBB 列表（``unmask_distractor_sampler.obstacle_obbs`` 的产物）。返回
    ``(被接受的布局, 每窗外环对, 墙钟, 计数)``；预判被拒或 16 次都不可行时抛 ``SceneGenerationError``。
    记录纪律见 :func:`spawn_swap_distractors_v5`。
    """
    dcfg = parse_distractor_cfg(distractor_cfg)
    scfg = parse_distractor_swap_cfg(swap_cfg)
    chs = float(cube_half_size)
    buttons_xy = [np.asarray(b, dtype=np.float64)[:2] for b in buttons_xy]
    if scfg.min_button_center_dist_m is not None and not buttons_xy:
        raise ValueError("distractor_swap 申报了按钮中心距，但没有传入按钮")
    timing: dict[str, float] = {}
    stats: dict[str, int] = {}

    t0 = time.perf_counter()
    recorder.record("actions.predicted_inner_swap_pairs", [[int(w.a), int(w.b)] for w in windows])
    prejudge = prejudge_inner_windows(windows)
    timing["inner_prejudge_s"] = time.perf_counter() - t0
    if prejudge is not None:
        k, rejection = prejudge
        recorder.record("layout.inner_sweep_prejudge", {"window": int(k), "rejection": rejection.as_dict()})
        raise SceneGenerationError(f"xhard 内环对内环扫掠在 reset 预判被拒（L20）：{rejection.summary()}")

    if scfg.rule == OUTER_BALANCED_RULE:
        # V6：外环 O4 + 放置后序号重排，单列成 _plan_swap_distractors_balanced（V5 分支下面逐字不动）
        return _plan_swap_distractors_balanced(
            windows=windows, guard_windows=guard_windows, obstacles=obstacles, buttons_xy=buttons_xy,
            generator=generator, recorder=recorder, dcfg=dcfg, scfg=scfg, chs=chs, timing=timing, stats=stats,
            difficulty=difficulty,
        )

    t0 = time.perf_counter()
    guard = InnerSweepGuard(windows if guard_windows is None else guard_windows)
    pad_shapes = padded_bin_shapes(chs, scfg.plan_pad_m)
    attempt_reasons: list[dict[str, int]] = []

    def extra_reject(i, x, y, yaw, _placed):
        # H1：带 plan_pad_m 余量的候选与任一窗内环扫掠相交即拒绝；不抽随机数（回放复核时同样被调用）
        return guard.first_rejection(distractor_bin_state(i, x, y, yaw, chs, pad_shapes)) is not None

    def accept(layout: DistractorLayout):
        if not scfg.enabled:
            return True, None
        perm = torch.randperm(layout.count, generator=generator).tolist()  # 追加在放置与 cube 抽样之后
        plan = plan_distractor_swaps(layout, perm, windows, cfg=scfg, cube_half_size=chs, buttons_xy=buttons_xy,
                                     stats=stats)
        attempt_reasons.append(dict(plan.reasons))
        return (True, plan) if plan.ok else (False, f"window_{plan.fail_window}")

    layout, plan = resample_distractor_layout(
        dcfg, obstacles=obstacles, generator=generator, cube_half_size=chs, recorder=recorder,
        accept=accept, max_attempts=scfg.layout_max_attempts, extra_reject=extra_reject,
    )
    pairs: list[tuple[int, int]] = []
    if scfg.enabled:
        order = [int(v) for v in recorder.value("objects.distractors.swap_order", list(plan.perm),
                                                decision_key="xhard.distractor_swap.initiator_rule")]
        if order != list(plan.perm):
            # 只在回放时可能发生：冻结排列与重抽不同 ⇒ 用冻结排列重新规划并复核（N17）
            plan = plan_distractor_swaps(layout, order, windows, cfg=scfg, cube_half_size=chs, buttons_xy=buttons_xy)
            if not plan.ok:
                raise SceneGenerationError(f"回放复核：冻结的外环发起者排列 {order} 在第 {plan.fail_window} 窗不可行")
        pairs = [(int(o), int(p)) for o, p in plan.pairs]
        recorder.record("actions.distractor_swap_pairs", [[o, p] for o, p in pairs])
        recorder.record("actions.distractor_swap_fallback", [int(v) for v in plan.fallbacks])
    timing["layout_and_plan_s"] = time.perf_counter() - t0
    stats["candidate_rejects"] = sum(sum(r.values()) for r in attempt_reasons)
    for reasons in attempt_reasons:
        for key, value in reasons.items():
            stats[f"reject_{key}"] = stats.get(f"reject_{key}", 0) + int(value)
    return layout, pairs, timing, stats


def spawn_swap_distractors_v5(
    env,
    *,
    generator: torch.Generator,
    partner_axes: Sequence[int],
    button_obbs: Sequence[Any] = (),
    hidden_half_size: float,
) -> SwapDistractorResult:
    """两个 Swap 环境新值档的干扰容器与外环交换规划（计划 2.5 伪码）。

    顺序：内环预演 → L20 内环对内环预判（拒绝即抛真 ``SceneGenerationError``）→ 统一采样器整段重抽（每次放置后
    ``perm = randperm(count)`` 并规划全部窗口，某窗全不可行即重抽，最多 ``layout_max_attempts`` 次）→ 被接受那次才
    ``recorder.value``（N18）→ 外环发起者排列 ``value`` → 交换对与回退 ``record`` → 建 actor。

    ``generator`` 必须是独立流 ``distractor_generator(seed)``：主流一次不多抽。回放冻结规格时整条流程照样重跑，
    冻结布局经 ``commit`` 按同一规则（含 H1 回调，用同一份内环预演）复核，冻结的发起者排列若与重抽不同则用冻结排列
    重新规划并复核可行性（N17），违反即抛 ``SceneGenerationError``。
    """
    decision = env._sampling["decision"][env.difficulty]
    chs = float(env.cube_half_size)
    dcfg = parse_distractor_cfg(decision["distractor"])
    inner_plan = getattr(env, "_newvalue_inner_plan", None)
    if inner_plan is not None:
        # V6（2.2 内环 S5）：内环窗口来自 reset 预规划的序列；H1 守卫覆盖 G 的全部可行槽对（R7）
        windows = planned_inner_windows(env)
        guard_windows = graph_guard_windows(env, inner_plan["graph"])
    else:
        windows = predict_inner_windows(env, partner_axes)
        guard_windows = None
    obstacles = obstacle_obbs(list(env.spawned_bins) + list(button_obbs), chs * dcfg.min_gap_factor)
    layout, pairs, timing, stats = plan_swap_distractors(
        windows=windows, obstacles=obstacles,
        buttons_xy=[np.asarray(c, dtype=np.float64)[:2] for c, _axes, _half in button_obbs],
        generator=generator, recorder=env._spec, distractor_cfg=decision["distractor"],
        swap_cfg=decision["distractor_swap"], cube_half_size=chs, guard_windows=guard_windows,
        difficulty=env.difficulty,
    )
    bins, cubes = build_distractor_actors(env, layout, hidden_half_size=hidden_half_size)
    return SwapDistractorResult(
        bins=bins, cubes=cubes, cube_bin_pairs=distractor_cube_bin_pairs(layout, bins, cubes), layout=layout,
        pairs=pairs, predicted_inner_pairs=[(w.a, w.b) for w in windows], timing=timing, stats=stats,
    )


# ════════════════════════════════════════════════════════════════════════════════
# V6（NEWTASK_RELEASE_V6_PLAN 2.2 / 2.4，M5(b)、M6(a)、M7(a)）：内环 S5 + 外环 O4
# ════════════════════════════════════════════════════════════════════════════════
def v6_inner_swap_plan_cfg(task: str) -> dict:
    """``decision.<tier>.swap_plan_v6`` 的申报值：S5（max→sum 评分）、G 连通作 reset 接受条件。"""
    if task not in V5_SWAP_TASKS:
        raise ValueError(f"只支持 {V5_SWAP_TASKS}，收到 {task!r}")
    return _su.inner_swap_plan_cfg(require_connected=True, score="max_sum")


def v6_distractor_swap_cfg(task: str) -> dict:
    """``decision.<tier>.distractor_swap`` 的 V6 申报值：只换规则三键（O4），路径约束与其余数值逐项沿用 V5（M7(a)）。"""
    cfg = v5_distractor_swap_cfg(task)
    cfg["initiator_rule"] = OUTER_BALANCED_RULE
    cfg["fallback"] = OUTER_BALANCED_FALLBACK
    cfg["partner"] = copy.deepcopy(OUTER_BALANCED_PARTNER_RULE)
    return cfg


def inner_slot_graph(states: Sequence[ObjectState], *, stats: dict | None = None) -> list[list[bool]]:
    """内环 4 个位姿槽的可行槽对图 G：每个无序槽对跑一次 ``check_swap_sweep_prefiltered``（其余内环容器静止）。

    与 L20 预判 :func:`prejudge_inner_windows` 同一判定；槽集合整局不变（每段交换末精确对换位姿），所以一局只需判 6 次。
    """
    states = list(states)

    def feasible(a: int, b: int) -> bool:
        others = [state for j, state in enumerate(states) if j not in (a, b)]
        _gap, rejection = check_swap_sweep_prefiltered(states[a], states[b], others, sweep_index=-1,
                                                       stage="inner_graph", stats=stats)
        return rejection is None

    return _su.slot_pair_graph(len(states), feasible)


def plan_inner_swaps_v6(env, generator: torch.Generator) -> dict:
    """VUS/BUS xhard 内环 S5 的 reset 预规划（计划 2.2 内环 1～3）。只在 xhard 分支、主流全部既有取值点之后调用。

    1. 读实际位姿构造 4 个槽的状态，算 G 并 ``record`` 进 ``layout.inner_swap_graph``；G 不连通抛真
       ``SceneGenerationError``（候选级重抽，计入 reset 拒绝）；
    2. 主流**追加一次** ``objects.swap_plan_seed = randint(0, 2**62)``（``value``），平局打破与重排都在以它播种的局部流上；
    3. S5 规划整段序列，逐次 ``value`` 进 ``actions.swap_pairs.<k> = {"initiator": "bin_a", "partner": "bin_b"}``；
       回放时用冻结值，并按 N17 复核「每次槽对在 G 里可行、没有立即撤销」，违反抛 ``EpisodeSpecError``；
    4. ``record`` ``objects.swap_plan`` = {counts, range, tries, undo}（``UNIFORM=REPORT`` 的逐局来源）。

    返回 ``{"graph", "pairs", "summary"}``，由环境挂到 ``_newvalue_inner_plan``。
    """
    from .episode_spec import EpisodeSpecError

    cfg = _su.parse_inner_swap_plan_cfg(env._sampling["decision"][env.difficulty]["swap_plan_v6"])
    bins = list(env.spawned_bins)
    states = [object_state_from_actor(actor, f"bin_{index}") for index, actor in enumerate(bins)]
    graph = inner_slot_graph(states)
    env._spec.record("layout.inner_swap_graph", [[int(a), int(b)] for a, b in _su.graph_edges(graph)])
    connected = _su.graph_connected(graph)
    env._spec.record("layout.inner_swap_graph_connected", bool(connected))
    if cfg["require_connected_graph"] and not connected:
        raise SceneGenerationError(
            f"{env.difficulty} 内环可行槽对图不连通（可行槽对 {_su.graph_edges(graph)}），按 V6 S5 拒绝本次布局"
        )
    seed = int(env._spec.value(
        "objects.swap_plan_seed",
        int(torch.randint(0, int(cfg["plan_seed_high_exclusive"]), (1,), generator=generator).item()),
        decision_key=f"{env.difficulty}.swap_plan_v6",
    ))
    local = torch.Generator()
    local.manual_seed(seed)
    plan = _su.plan_balanced_swaps(graph, int(env.swap_times), local, score=cfg["score"],
                                   budget=int(cfg["range_retry_budget"]), accept_range=int(cfg["accept_range"]),
                                   forbid_undo=True)
    if plan is None:
        raise SceneGenerationError(f"{env.difficulty} 内环 S5 无法规划 {env.swap_times} 次交换（可行槽对 {_su.graph_edges(graph)}）")
    pairs = []
    for k, (a, b) in enumerate(plan.pairs):
        chosen = env._spec.value(f"actions.swap_pairs.{k}", {"initiator": f"bin_{a}", "partner": f"bin_{b}"},
                                 decision_key=f"{env.difficulty}.swap_plan_v6")
        try:
            pairs.append((int(str(chosen["initiator"]).rsplit("_", 1)[1]), int(str(chosen["partner"]).rsplit("_", 1)[1])))
        except (TypeError, KeyError, ValueError, IndexError) as exc:
            raise EpisodeSpecError(f"{env.difficulty}: actions.swap_pairs.{k} 形状非法：{chosen!r}") from exc
    problems, stats = _su.verify_swap_sequence(graph, pairs, forbid_undo=True)
    if problems:
        raise EpisodeSpecError(f"{env.difficulty} 内环交换序列违反 V6 S5 规则：" + "；".join(problems))
    summary = stats.summary()
    summary["tries"] = int(plan.tries)
    env._spec.record("objects.swap_plan", summary)
    return {"graph": graph, "pairs": pairs, "summary": summary}


def planned_inner_windows(env) -> list[InnerWindow]:
    """按 reset 预规划的 S5 序列（``env._newvalue_inner_plan["pairs"]``）预演全部内环窗口（位姿读实际 actor）。"""
    bins = list(env.spawned_bins)
    states = [object_state_from_actor(actor, f"bin_{index}") for index, actor in enumerate(bins)]
    windows: list[InnerWindow] = []
    for a, b in env._newvalue_inner_plan["pairs"]:
        windows.append(InnerWindow(a=int(a), b=int(b), states=tuple(states)))
        state_a, state_b = states[a], states[b]
        states[a] = ObjectState(name=state_a.name, p=state_b.p.copy(), q=state_b.q.copy(), shapes=state_a.shapes)
        states[b] = ObjectState(name=state_b.name, p=state_a.p.copy(), q=state_a.q.copy(), shapes=state_b.shapes)
    return windows


def graph_guard_windows(env, graph: Sequence[Sequence[bool]]) -> list[InnerWindow]:
    """H1 守卫用的窗口：G 的每条可行边一窗（初始槽状态上交换该槽对）。内环容器盒体相同，任何时刻交换某槽对的扫掠
    都与此相同，所以守卫覆盖 G 中全部可行槽对（计划 2.2 内环 3、V6 调查 R7），是实际序列的超集。"""
    bins = list(env.spawned_bins)
    states = tuple(object_state_from_actor(actor, f"bin_{index}") for index, actor in enumerate(bins))
    return [InnerWindow(a=int(a), b=int(b), states=states) for a, b in _su.graph_edges(graph)]


def relabel_distractor_layout(layout: DistractorLayout, perm: Sequence[int]) -> DistractorLayout:
    """放置后序号重排（V6 O4 的跨局均匀）：新序号 i 的容器 = 放置序 ``perm[i]`` 的容器；cube 映射随之改写。"""
    perm = [int(v) for v in perm]
    if sorted(perm) != list(range(layout.count)):
        raise ValueError(f"序号重排 {perm} 不是 0..{layout.count - 1} 的排列")
    inverse = {old: new for new, old in enumerate(perm)}
    return DistractorLayout(
        bins=[tuple(layout.bins[old]) for old in perm],
        cube_count=int(layout.cube_count),
        cube_bins=[inverse[int(c)] for c in layout.cube_bins],
        color_order=list(layout.color_order),
        trials=[layout.trials[old] for old in perm] if len(layout.trials) == layout.count else list(layout.trials),
    )


def plan_distractor_swaps_balanced(
    layout: DistractorLayout,
    windows: Sequence[InnerWindow],
    generator: torch.Generator,
    *,
    cfg: DistractorSwapConfig,
    cube_half_size: float,
    buttons_xy: Sequence[Sequence[float]] = (),
    stats: dict | None = None,
) -> OuterSwapPlan:
    """V6 外环 O4（纯几何 + 局部流平局）：每窗把全部对象对按「两者参与次数 max、再 sum」升序分组，上一窗用过的对放最末
    （禁止立即撤销，除非别无选择）；第一组里有可行者即在其可行者中均匀抽一对。四道判定仍是 :func:`evaluate_outer_candidate`。

    ``OuterSwapPlan.fallbacks[k]`` 在本规则下记「第 k 窗被接受的是第几组」（0 = 评分最小组）。
    """
    count = layout.count
    shapes = padded_bin_shapes(cube_half_size, cfg.plan_pad_m)
    outer = [distractor_bin_state(i, x, y, yaw, cube_half_size, shapes) for i, (x, y, yaw) in enumerate(layout.bins)]
    plan = OuterSwapPlan(ok=False, perm=list(range(count)))
    cnt = [0] * count
    last = None
    for k, window in enumerate(windows):
        chosen = None
        for g, group in enumerate(_su.balanced_pair_groups(count, cnt, last)):
            feasible = []
            for o, p in group:
                ok, reason, _rejection = evaluate_outer_candidate(
                    k, window, o, p, outer, cfg=cfg, cube_half_size=cube_half_size, buttons_xy=buttons_xy, stats=stats,
                )
                if ok:
                    feasible.append((o, p))
                else:
                    plan.reasons[reason] = plan.reasons.get(reason, 0) + 1
            if feasible:
                o, p = feasible[int(torch.randint(0, len(feasible), (1,), generator=generator).item())]
                chosen = (o, p, g)
                break
        if chosen is None:
            plan.fail_window = k
            return plan
        o, p, g = chosen
        plan.pairs.append((o, p))
        plan.fallbacks.append(g)
        cnt[o] += 1
        cnt[p] += 1
        last = (min(o, p), max(o, p))
        state_o, state_p = outer[o], outer[p]
        outer[o] = ObjectState(name=state_o.name, p=state_p.p.copy(), q=state_p.q.copy(), shapes=state_o.shapes)
        outer[p] = ObjectState(name=state_p.name, p=state_o.p.copy(), q=state_o.q.copy(), shapes=state_p.shapes)
    plan.ok = True
    plan.final_xy = [[float(v) for v in state.p[:2]] for state in outer]
    return plan


def _plan_swap_distractors_balanced(*, windows, guard_windows, obstacles, buttons_xy, generator, recorder, dcfg, scfg,
                                    chs, timing, stats, difficulty):
    """:func:`plan_swap_distractors` 的 V6 分支：放置 + H1（G 全部可行槽对）→ 每次放置后独立流追加
    ``randperm(count)``（序号重排）与一个规划种子 → 重排后的布局上跑 O4 → 被接受那次才 ``value``。

    规格：放置序的布局照旧由 ``commit_distractor_layout`` 落 ``objects.distractors.*``（回放按放置序复核，判据与 V5
    相同）；``objects.distractors.label_perm`` / ``objects.distractors.swap_plan_seed`` 两个新取值点；重排后的
    公开布局 ``record`` 进 ``objects.distractors.public``；建 actor、交换对、运行时一律用公开序号。
    """
    t0 = time.perf_counter()
    guard = InnerSweepGuard(windows if guard_windows is None else guard_windows)
    pad_shapes = padded_bin_shapes(chs, scfg.plan_pad_m)
    attempt_reasons: list[dict[str, int]] = []

    def extra_reject(i, x, y, yaw, _placed):
        return guard.first_rejection(distractor_bin_state(i, x, y, yaw, chs, pad_shapes)) is not None

    def plan_for(layout, perm, seed):
        local = torch.Generator()
        local.manual_seed(int(seed))
        public = relabel_distractor_layout(layout, perm)
        plan = plan_distractor_swaps_balanced(public, windows, local, cfg=scfg, cube_half_size=chs,
                                              buttons_xy=buttons_xy, stats=stats)
        plan.perm = [int(v) for v in perm]
        return public, plan

    def accept(layout: DistractorLayout):
        perm = torch.randperm(layout.count, generator=generator).tolist()  # 追加在放置与 cube 抽样之后
        seed = int(torch.randint(0, 2 ** 62, (1,), generator=generator).item())
        _public, plan = plan_for(layout, perm, seed)
        attempt_reasons.append(dict(plan.reasons))
        return (True, (perm, seed, plan)) if plan.ok else (False, f"window_{plan.fail_window}")

    layout, payload = resample_distractor_layout(
        dcfg, obstacles=obstacles, generator=generator, cube_half_size=chs, recorder=recorder,
        accept=accept, max_attempts=scfg.layout_max_attempts, extra_reject=extra_reject,
    )
    perm, seed, plan = payload
    perm_v = [int(v) for v in recorder.value("objects.distractors.label_perm", list(perm),
                                             decision_key=f"{difficulty}.distractor_swap.partner.relabel")]
    seed_v = int(recorder.value("objects.distractors.swap_plan_seed", int(seed),
                                decision_key=f"{difficulty}.distractor_swap.initiator_rule"))
    public, replanned = plan_for(layout, perm_v, seed_v)
    if (perm_v, seed_v) != (list(perm), int(seed)):
        # 只在回放可能发生：冻结的重排 / 种子与重抽不同 ⇒ 用冻结值重新规划并复核（N17）
        plan = replanned
        if not plan.ok:
            raise SceneGenerationError(f"回放复核：冻结的外环重排 / 种子在第 {plan.fail_window} 窗不可行"
                                       f"（外环 {public.count} 个，拒绝原因 {dict(plan.reasons)}）")
    recorder.record("objects.distractors.public", public.to_spec())
    pairs = [(int(o), int(p)) for o, p in plan.pairs]
    recorder.record("actions.distractor_swap_pairs", [[o, p] for o, p in pairs])
    recorder.record("actions.distractor_swap_fallback", [int(v) for v in plan.fallbacks])
    cnt = _su.participation(pairs, public.count)
    balance = {"counts": cnt, "unvisited": int(sum(1 for c in cnt if c == 0)),
               "range": int(max(cnt) - min(cnt)), "undo": int(_su.count_undo(pairs))}
    recorder.record("actions.distractor_swap_balance", balance)
    timing["layout_and_plan_s"] = time.perf_counter() - t0
    stats["candidate_rejects"] = sum(sum(r.values()) for r in attempt_reasons)
    for reasons in attempt_reasons:
        for key, value in reasons.items():
            stats[f"reject_{key}"] = stats.get(f"reject_{key}", 0) + int(value)
    stats["outer_unvisited"] = balance["unvisited"]
    stats["outer_undo"] = balance["undo"]
    return public, pairs, timing, stats


# ── 运行时 ─────────────────────────────────────────────────────────────────────
def joint_sweep_from_actual(env, sweep_index: int, initiator, partner) -> tuple[float, CollisionRejection | None, dict]:
    """运行时窗口起点的两对联合复核（实际位姿、真实碰撞盒、认证预筛）：内环对（运行时解析，L22 a）+ 本窗规划的外环对，
    其余内环与干扰容器全部静止。返回 ``(最小判定值, 拒绝证据, 附带信息)``；附带信息里有与 reset 预演的内环对是否一致。"""
    inner = {
        index: object_state_from_actor(actor, f"bin_{index}")
        for index, actor in enumerate(env.spawned_bins)
        if actor is not None
    }
    a = env.spawned_bins.index(initiator)
    b = env.spawned_bins.index(partner)
    outer = [object_state_from_actor(actor, f"distractor_bin_{index}")
             for index, actor in enumerate(getattr(env, "distractor_bins", []))]
    planned = list(getattr(env, "distractor_swap_pairs", None) or [])
    moving_pairs = [(inner[a], inner[b])]
    moving_outer: tuple[int, ...] = ()
    if int(sweep_index) < len(planned):
        o, p = planned[int(sweep_index)]
        moving_pairs.append((outer[o], outer[p]))
        moving_outer = (int(o), int(p))
    bystanders = [state for index, state in sorted(inner.items()) if index not in (a, b)]
    bystanders += [state for index, state in enumerate(outer) if index not in moving_outer]
    gap, rejection = check_multi_swap_sweep(moving_pairs, bystanders, sweep_index=sweep_index, stage="sweep")
    predicted = list(getattr(env, "predicted_inner_swap_pairs", None) or [])
    expected = tuple(predicted[int(sweep_index)]) if int(sweep_index) < len(predicted) else None
    info = {
        "inner_pair": [a, b],
        "outer_pair": list(moving_outer) if moving_outer else None,
        "predicted_inner_pair": None if expected is None else list(expected),
        "inner_partner_mismatch": expected is not None and expected != (a, b),
    }
    return gap, rejection, info


def run_outer_swaps(env, timestep) -> None:
    """外环交换执行（在 ``step`` 里、AST 锁定的内环搭档循环之外调用）：第 k 窗与内环第 k 窗同一 ``[start, end]``，
    按 reset 规划的 ``(o, p)`` 调 ``swap_flat_two_lane``（lane 0.07、smoothstep、竖直），其余干扰容器钉住；
    窗口首步 ``record`` 一条运行时留痕。不增加任何控制步。"""
    from .statechange import swap_flat_two_lane  # 延迟导入：纯几何单测不必加载仿真依赖

    pairs = list(getattr(env, "distractor_swap_pairs", None) or [])
    schedule = list(getattr(env, "swap_schedule", None) or [])
    bins = list(getattr(env, "distractor_bins", None) or [])
    step = int(timestep)
    for k, (o, p) in enumerate(pairs):
        if k >= len(schedule):
            break
        start, end = int(schedule[k][2]), int(schedule[k][3])
        if step == start:
            env._spec.record(f"actions.distractor_swap_windows.{k}",
                             {"initiator": int(o), "partner": int(p), "start_step": start, "end_step": end})
        swap_flat_two_lane(
            env,
            cube_a=bins[o],
            cube_b=bins[p],
            start_step=start,
            end_step=end,
            cur_step=step,
            lane_offset=LANE_OFFSET,
            smooth=True,
            keep_upright=True,
            other_cube=[actor for index, actor in enumerate(bins) if index not in (o, p)],
        )
