"""V4 xhard：Unmask 族的外环干扰容器（NEWTASK_RELEASE_V4_PLAN 2.7① / B3 / B13 / H1）。

只被 xhard 分支调用；原三档从不进入本模块，所以不影响 V0/V1。

做法（全部是本模块自己的判据，**不改** ``object_generation.spawn_random_bin`` 的既有语义）：

* 位置：在 ``[-R, R]²``（``R = ring_max_abs_xy[1]``）上均匀抽 ``(x, y)``，拒绝
  ``max(|x|,|y|) < ring_max_abs_xy[0]`` 的点 ⇒ 中心严格落在方环 ``max(|x|,|y|) ∈ [r_in, r_out]``；
* 相机可见：把容器按「任意 yaw 的外接正方形」取底面与顶面 8 个角点投到前视相机
  （``eye / target / fov / 分辨率`` 与环境 ``_default_sensor_configs`` 写死的相同），全部落在画面内才接受；
* 避让：与 ``spawn_random_bin`` 同一判据——``avoid`` 里的 actor 按 OBB 外扩 ``min_gap``、预制 OBB 元组原样，
  候选中心到障碍 OBB 的点距须 ``≥ bin_half_size + min_gap``；已放的干扰容器随即进 ``avoid``；
* yaw：位置通过全部检查后才抽 ``u·90°``（与 ``spawn_random_bin`` 一致）；
* 含 cube：随机取 ``cube_count_range`` 闭区间内的个数、随机挑容器、从 ``DISTRACTOR_COLORS`` 不放回挑颜色。

随机调用顺序固定为：逐容器（拒绝循环的 2 次 rand/次 + 通过后的 1 次 yaw）→ cube 个数 → 挑容器 → 挑颜色。
调用方必须把本函数放在该环境**全部既有取值点之后**（红线 N5）。

每个取值点都经 ``recorder.value``（带 ``decision_key``）；请求数与实际数经 ``recorder.record`` 记下，
放不满直接抛 :class:`SceneGenerationError`（2.2④：不许静默截断）。
"""

from __future__ import annotations

import numpy as np
import torch

from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb

from .object_generation import _trimesh_box_to_obb2d, build_bin, spawn_fixed_cube
from .SceneGenerationError import SceneGenerationError
from .xhard import DISTRACTOR_COLORS

# 与 VideoUnmask / ButtonUnmask 的 _default_sensor_configs 逐字相同（那里是写死的字面量）
BASE_CAMERA_EYE = (0.3, 0.0, 0.4)
BASE_CAMERA_TARGET = (0.0, 0.0, -0.2)
BASE_CAMERA_FOV = np.pi / 2
BASE_CAMERA_RES = 256

# 干扰容器用的名字前缀：刻意不叫 bin_<i>，否则会被 step 里的揭示动画扫到（计划 2.7①）
DISTRACTOR_BIN_PREFIX = "distractor_bin"
DISTRACTOR_CUBE_PREFIX = "distractor_cube"


def bin_geometry(cube_half_size: float) -> tuple[float, float, float]:
    """返回 ``(采样判据半边, 任意 yaw 外接半边, 高度)``，与 ``build_bin`` / ``spawn_random_bin`` 同一套尺寸。"""
    inner_side = cube_half_size * 2.5
    wall_thickness = 0.005
    floor_thickness = 0.004
    half = (inner_side + wall_thickness) * 0.5          # spawn_random_bin 的 bin_half_size
    outer_half = inner_side * 0.5 + wall_thickness       # 外廓半边（0.03）
    height = floor_thickness + cube_half_size * 2.5      # 底板 + 墙高
    return half, outer_half * np.sqrt(2.0), height


def _camera_axes(eye, target):
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return eye, forward, right, up


def visible_in_camera(points, eye=BASE_CAMERA_EYE, target=BASE_CAMERA_TARGET,
                      fov=BASE_CAMERA_FOV, margin_px: float = 0.0, res: int = BASE_CAMERA_RES) -> bool:
    """针孔模型：所有点都在相机前方且投影落在 ``[margin, res-margin]`` 像素框内才算可见。"""
    eye, forward, right, up = _camera_axes(eye, target)
    tan_half = np.tan(fov / 2.0)
    limit = 1.0 - 2.0 * margin_px / res
    for point in points:
        d = np.asarray(point, dtype=np.float64) - eye
        depth = float(d @ forward)
        if depth <= 1e-6:
            return False
        if abs(float(d @ right) / depth) / tan_half > limit:
            return False
        if abs(float(d @ up) / depth) / tan_half > limit:
            return False
    return True


def bin_corners(x: float, y: float, reach: float, height: float):
    """容器按任意 yaw 的外接正方形取底面与顶面 8 个角点（保守：对任何 yaw 都成立）。"""
    return [(x + sx * reach, y + sy * reach, z)
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for z in (0.0, height)]


def in_ring(x: float, y: float, ring) -> bool:
    m = max(abs(x), abs(y))
    return float(ring[0]) <= m <= float(ring[1])


def _obstacle_obbs(avoid, min_gap):
    """与 spawn_random_bin 同一收集口径：actor 外扩 min_gap，预制 OBB 元组原样。"""
    out = []
    for item in avoid:
        if isinstance(item, tuple):
            if len(item) == 3 and isinstance(item[0], np.ndarray) and isinstance(item[1], np.ndarray):
                out.append(item)
                continue
            actor, pad = item
        else:
            actor, pad = item, min_gap
        try:
            out.append(_trimesh_box_to_obb2d(get_actor_obb(actor, to_world_frame=True, vis=False),
                                             extra_pad=float(pad)))
        except Exception:  # noqa: BLE001 与原实现一致：没有物理网格的对象忽略
            pass
    return out


def _hits(pos, obbs, reach):
    for c_obs, a_obs, h_obs in obbs:
        local = a_obs.T @ (pos - c_obs)
        closest = c_obs + a_obs @ np.clip(local, -h_obs, h_obs)
        if np.linalg.norm(pos - closest) < reach:
            return True
    return False


def spawn_ring_distractor_bins(env, *, cfg: dict, avoid: list, generator: torch.Generator,
                               recorder, hidden_half_size: float,
                               spec_prefix: str = "objects.distractors",
                               decision_prefix: str = "xhard.distractor"):
    """放 ``cfg['count']`` 个外环干扰容器，其中随机 ``cube_count_range`` 个内含干扰色 cube。

    ``cfg`` 即 decision 里 ``xhard.distractor`` 子树；返回 ``(bins, cubes)`` 两个列表，
    并把已放容器追加进 ``avoid``（调用方若还有后续避让可以继续用）。
    """
    count = int(cfg["count"])
    ring = [float(v) for v in cfg["ring_max_abs_xy"]]
    low, high = (int(v) for v in cfg["cube_count_range"])
    min_gap = env.cube_half_size * float(cfg["min_gap_factor"])
    max_trials = int(cfg["max_trials"])
    if not (0 < ring[0] < ring[1]):
        raise ValueError(f"ring_max_abs_xy 非法：{ring}")
    if not (0 <= low <= high <= count) or high > len(DISTRACTOR_COLORS):
        raise ValueError(f"cube_count_range 非法：{[low, high]}（容器 {count} 个、干扰色 {len(DISTRACTOR_COLORS)} 种）")

    pool = [c["name"] for c in DISTRACTOR_COLORS]
    if list(cfg["color_pool"]) != pool:
        # 色池是全局决策 B2（黄/青/品红），不许按局改；申报在 decision 里只为留档可见
        raise ValueError(f"color_pool 必须等于全局干扰色池 {pool}，收到 {cfg['color_pool']}")

    half, reach_any_yaw, height = bin_geometry(env.cube_half_size)
    reject_reach = half + min_gap
    span = ring[1]
    recorder.record(f"{spec_prefix}.requested", count)

    bins = []
    for i in range(count):
        placed = None
        # 障碍 OBB 每个容器只收集一次（trimesh 取 OBB 较慢）；本容器的拒绝循环里 avoid 不变
        obbs = _obstacle_obbs(avoid, min_gap)
        for _ in range(max_trials):
            x = float(torch.rand(1, generator=generator).item() * 2.0 * span - span)
            y = float(torch.rand(1, generator=generator).item() * 2.0 * span - span)
            if not in_ring(x, y, ring):
                continue
            if not visible_in_camera(bin_corners(x, y, reach_any_yaw, height)):
                continue
            if _hits(np.array([x, y], dtype=np.float64), obbs, reject_reach):
                continue
            yaw = float(torch.rand(1, generator=generator).item() * 90.0)
            placed = (x, y, yaw)
            break
        if placed is None:
            recorder.record(f"{spec_prefix}.placed", len(bins))
            raise SceneGenerationError(
                f"xhard 干扰容器放不满：请求 {count} 个，第 {i} 个在 {max_trials} 次尝试内无可行位置"
            )
        x, y, yaw = recorder.value(f"{spec_prefix}.bins.{i}", list(placed),
                                   decision_key=f"{decision_prefix}.ring_max_abs_xy")
        actor = build_bin(env, callsign=f"{DISTRACTOR_BIN_PREFIX}_{i}", position=[x, y, 0.002],
                          z_rotation_deg=yaw)
        bins.append(actor)
        avoid.append(actor)
    recorder.record(f"{spec_prefix}.placed", len(bins))

    n_cubes = int(recorder.value(
        f"{spec_prefix}.cube_count",
        torch.randint(low, high + 1, (1,), generator=generator).item(),
        decision_key=f"{decision_prefix}.cube_count_range",
    ))
    cube_bins = recorder.value(
        f"{spec_prefix}.cube_bins",
        torch.randperm(count, generator=generator)[:n_cubes].tolist(),
        decision_key=f"{decision_prefix}.count",
    )
    color_idx = recorder.value(
        f"{spec_prefix}.cube_colors",
        torch.randperm(len(DISTRACTOR_COLORS), generator=generator)[:n_cubes].tolist(),
        decision_key=f"{decision_prefix}.color_pool",
    )
    if len(cube_bins) != n_cubes or len(color_idx) != n_cubes:
        raise SceneGenerationError(f"干扰 cube 规格自相矛盾：个数 {n_cubes}、容器 {cube_bins}、颜色 {color_idx}")

    cubes = []
    for j, (b_idx, c_idx) in enumerate(zip(cube_bins, color_idx)):
        color = DISTRACTOR_COLORS[int(c_idx)]
        p = bins[int(b_idx)].pose.p
        if isinstance(p, torch.Tensor):
            p = p[0].detach().cpu().numpy()
        cube = spawn_fixed_cube(
            env,
            position=[float(p[0]), float(p[1])],
            half_size=hidden_half_size,
            color=color["rgba"],
            name_prefix=f"{DISTRACTOR_CUBE_PREFIX}_{j}_{color['name']}",
            yaw=0.0,
            dynamic=True,
        )
        cubes.append(cube)
    return bins, cubes


# ── 干扰容器的揭示与误抓判失败（用户 2026-09-22 决策「参与揭示+误抓即失败」）──────────────
# 四个 Unmask 环境（VideoUnmask / ButtonUnmask / VideoUnmaskSwap / ButtonUnmaskSwap）的 xhard 共用；
# 两个 Swap 环境的干扰容器由 unmask_swap_xhard.build_distractors 生成，同样挂在 env.distractor_bins 上。
# 只在新值族（V6 族判断）分支被调用，原三档从不进入。


def reveal_distractor_bins(env, *, start_step: int, end_step: int, cur_step: int) -> None:
    """让干扰容器与区域内容器走**同一揭示机制、同一时段**：逐个调用
    ``statechange.lift_and_drop_objects_back_to_original``（窗口前半段移到远处露出内容、半窗那一步放回原位）。

    空的干扰容器同样抬起（露出无物）。只做揭示，不进 ``spawned_bins``、不参与 swap / 最近邻。
    """
    from .statechange import lift_and_drop_objects_back_to_original

    for actor in list(getattr(env, "distractor_bins", None) or []):
        if actor is None:
            continue
        lift_and_drop_objects_back_to_original(
            env, obj=actor, start_step=start_step, end_step=end_step, cur_step=cur_step,
        )


def any_distractor_bin_lifted(env):
    """任一干扰容器被抬起即真；判据与区域内容器相同（``subgoal_evaluate_func.is_bin_pickup``：z > 0.15）。"""
    from .subgoal_evaluate_func import is_any_bin_pickup

    return is_any_bin_pickup(env, [a for a in (getattr(env, "distractor_bins", None) or []) if a is not None])


def add_distractor_misgrasp_failure(env, tasks) -> int:
    """给任务表里每个**已有** ``failure_func`` 的条目追加「任一干扰容器被抬起即失败」，返回改动条数。

    * 只包装 ``failure_func`` 非 None 的条目（即抓取 / 放下两类；static、按钮等原本不判失败的条目不动）；
    * 新的 ``failure_func`` 返回 ``[原结果, 干扰判据]`` 列表，由 ``_coerce_failure_result`` 取 any——
      原结果的形态（如 ButtonUnmask 首个抓取任务返回的单元素列表）原样保留在列表第一项里；
    * 原 ``failure_func`` 不是可调用对象时（预先算好的值）原样作为第一项；
    * 只改 ``failure_func`` 一个键，``solve``（``inject_fail_grasp`` 替换的对象）与其余键不动，调用先后均可。
    """
    changed = 0
    for task in tasks:
        if not isinstance(task, dict):
            continue
        original = task.get("failure_func")
        if original is None:
            continue

        def _combined(original=original):
            first = original() if callable(original) else original
            return [first, any_distractor_bin_lifted(env)]

        _combined.__name__ = "xhard_distractor_misgrasp_failure"
        task["failure_func"] = _combined
        changed += 1
    return changed
