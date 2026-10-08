"""V4 xhard 档的共用常量与采样工具（NEWTASK_RELEASE_V4_PLAN 2.0①）。

只在 xhard 分支被读；原三档从不 import 本模块里的东西参与取值，所以不影响 V0/V1。

* ``DISTRACTOR_COLORS``：「其他颜色」干扰物的全局色池（B2：黄／青／品红，六个环境共用，A5 固定 3 个）。
* ``corner_push``：边角偏置映射。把 ``[0,1]`` 上的均匀抽样值按 ``corner_bias`` 推向两端，
  ``corner_bias=0`` 时**原样返回同一个对象**（调用方据此保证与现有均匀采样逐字等价）。
* ``cube_obb2d_exact``（V5 2.0①）：按方块真实 yaw 给出预制 2D 障碍 ``(c, A, h)``，替代
  ``object_generation._trimesh_box_to_obb2d`` 在正方体上会退化成线段的 actor 路径。
"""

from __future__ import annotations

import colorsys
import math

import numpy as np

# 顺序固定：黄、青、品红。名字用于任务文本与规格记录，rgba 与 B2 决策逐字一致。
DISTRACTOR_COLORS = (
    {"name": "yellow", "rgba": (1, 1, 0, 1)},
    {"name": "cyan", "rgba": (0, 1, 1, 1)},
    {"name": "magenta", "rgba": (1, 0, 1, 1)},
)

# V7 定值表（0928 方案 §3.2.2、R11）：PickXtimes／SwingXtimes 的干扰块 1/2/3/4，需要第 4 种干扰色。
# 只给这两个环境用；上面的三色池同时是四个 Unmask／Swap 任务的色池（采样器要求逐字相等），不动。
# 第 4 色须与红／蓝／绿目标色及三色池可分，阶段 3 出图目视选定（方案 §1 第 9 条 C1）。
BLOCK_DISTRACTOR_COLORS = DISTRACTOR_COLORS + (
    {"name": "orange", "rgba": (1, 0.5, 0, 1)},
)

# 「方块颜色任意」的色域（用户 2026-09-22 定「设饱和度/亮度下限」）：色相任意，
# 饱和度 ≥0.5、亮度 ≥0.4，排除近白（会与白色高亮圆盘混淆）、近黑、近灰。
# 仍然只抽 3 个 [0,1) 均匀数（与原 RGB 均匀抽法的随机调用次数相同），再做确定性映射。
HSV_FLOOR_COLOR = {"h_range": [0.0, 1.0], "s_range": [0.5, 1.0], "v_range": [0.4, 1.0]}


def hsv_floor_rgb(u, cfg=None):
    """3 个 [0,1) 均匀数 → 限定色域内的 RGB（浮点三元组）。"""
    cfg = HSV_FLOOR_COLOR if cfg is None else cfg
    (h0, h1), (s0, s1), (v0, v1) = cfg["h_range"], cfg["s_range"], cfg["v_range"]
    h = h0 + float(u[0]) * (h1 - h0)
    s = s0 + float(u[1]) * (s1 - s0)
    v = v0 + float(u[2]) * (v1 - v0)
    return list(colorsys.hsv_to_rgb(h % 1.0, s, v))


# corner_bias=1 时的幂指数 1/(1+CORNER_GAIN)；4 ⇒ 指数 0.2，t=0.5 被推到约 0.87。
CORNER_GAIN = 4.0


def corner_push(u, corner_bias):
    """把单个 ``u ∈ [0,1]`` 按边角偏置推向 0 或 1 端。

    映射：``t = 2u-1``，``t' = sign(t)·|t|^p``，``p = 1/(1+CORNER_GAIN·b)``，返回 ``(t'+1)/2``。
    单调、保端点、关于 0.5 对称；两个坐标各自独立推移 ⇒ 联合分布偏向四角。
    ``b`` 必须在 ``[0,1]``；``b == 0`` 时不做任何浮点运算、直接返回 ``u``。
    """
    b = float(corner_bias)
    if b == 0.0:
        return u
    if not 0.0 <= b <= 1.0:
        raise ValueError(f"corner_bias 必须在 [0,1]，收到 {corner_bias}")
    t = 2.0 * float(u) - 1.0
    p = 1.0 / (1.0 + CORNER_GAIN * b)
    mag = abs(t) ** p
    return (1.0 + (mag if t >= 0 else -mag)) / 2.0


def _as_numpy(value):
    """torch 张量／sapien 数组／列表 → float64 的 numpy 数组（不 import torch，按鸭子类型取值）。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _xy_yaw_from_pose(pose):
    """从位姿（``.p`` 为 xyz、``.q`` 为 wxyz 四元数，允许带 batch 维 ``[1, ·]``）取中心 xy 与 yaw。

    yaw 取「三根体轴里最水平的那根」在桌面上的朝向：任意旋转下三根轴 z 分量平方和为 1，
    最水平那根的 |z| ≤ 1/√3，其 xy 投影长度 ≥ √(2/3)，因此**永不退化**；正方体四个侧面等价，
    取哪根水平轴都得到同一个正方形。直立方块（只绕 z 转）时第 0 列即体 x 轴，yaw 与建方块时的 yaw 一致（模 2π）。
    """
    p = _as_numpy(pose.p).reshape(-1)[:3]
    w, x, y, z = _as_numpy(pose.q).reshape(-1)[:4]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("cube_obb2d_exact: 位姿四元数为零")
    w, x, y, z = w / n, x / n, y / n, z / n
    # 旋转矩阵的三列（体 x/y/z 轴在世界系下的方向）
    cols = (
        (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
        (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
        (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)),
    )
    k = min(range(3), key=lambda i: abs(cols[i][2]))  # 并列时取编号小的，直立方块取第 0 列
    return float(p[0]), float(p[1]), math.atan2(cols[k][1], cols[k][0])


def cube_obb2d_exact(pose_or_xy_yaw, half, pad=0.0):
    """方块的精确 2D 障碍 ``(c, A, h)``，可直接放进 ``spawn_random_cube`` / ``spawn_random_target`` 的 ``avoid``。

    * ``pose_or_xy_yaw``：``(x, y, yaw)`` 三个数（yaw 为绕 z 的弧度），或带 ``.p`` / ``.q`` 的位姿
      （mani_skill ``Pose``、``sapien.Pose``），或带 ``.pose`` 的 actor（取其当前位姿）。
    * ``half``：方块半边长（米，标量）；``pad``：两个半轴各自外扩的余量，语义与 actor 路径
      ``avoid=[(actor, pad)]`` 的 ``extra_pad`` 相同。
    * 返回 ``(c, A, h)``：``c`` 形状 ``(2,)``、``A`` 形状 ``(2, 2)``（每列是一根单位轴，``[[cos, -sin], [sin, cos]]``）、
      ``h`` 形状 ``(2,)``，全部 float64 的 ``np.ndarray``——正是两个 spawn 函数识别「预制障碍」的格式
      （三元组且前两项是 ``np.ndarray``），也与 ``_build_new_cube_obb2d(x, y, half, yaw, pad)`` 逐位同构。

    纯函数：不抽随机数、不读写任何环境状态；两根轴恒为正交单位向量，不会像
    ``_trimesh_box_to_obb2d`` 那样在竖直轴落进前两列时退化成线段（计划 2.0①）。
    """
    if hasattr(pose_or_xy_yaw, "p") and hasattr(pose_or_xy_yaw, "q"):
        x, y, yaw = _xy_yaw_from_pose(pose_or_xy_yaw)
    elif hasattr(pose_or_xy_yaw, "pose"):
        x, y, yaw = _xy_yaw_from_pose(pose_or_xy_yaw.pose)
    else:
        values = _as_numpy(pose_or_xy_yaw).reshape(-1)
        if values.shape != (3,):
            raise ValueError(f"cube_obb2d_exact: 需要 (x, y, yaw) 三个数，收到形状 {values.shape}")
        x, y, yaw = (float(v) for v in values)
    half = float(half)
    pad = float(pad)
    if not half > 0.0:
        raise ValueError(f"cube_obb2d_exact: half 必须为正，收到 {half}")
    if pad < 0.0:
        raise ValueError(f"cube_obb2d_exact: pad 不能为负，收到 {pad}")
    # 与 object_generation._build_new_cube_obb2d 同一写法，保证同一 (x, y, yaw) 得到逐位相同的数组
    c = np.array([x, y], dtype=np.float64)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    A = np.array([[cos_y, -sin_y],
                  [sin_y, cos_y]], dtype=np.float64)
    h = np.array([half + pad, half + pad], dtype=np.float64)
    return c, A, h
