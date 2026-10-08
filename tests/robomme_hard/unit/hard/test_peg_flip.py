"""抓杆姿态归约（``utils/subgoal_planner_func.peg_grasp_needs_flip`` 与 ``flip_grasp_q``）：手算四元数算例。

判据：夹爪局部 x 轴的水平朝向相对「基座 → 抓取点」方位角超过 ±90° 才翻 Rz(π)；正好 90° 不翻（严格 ``>``）。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from robomme_hard.robomme_env.utils import subgoal_planner_func as SPF


def _yaw_q(theta):
    """绕世界 z 轴转 theta 的单位四元数 (w, x, y, z)：夹爪 x 轴水平朝向即 theta。"""
    return np.array([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)])


BASE = (0.0, 0.0)
GRASP_P = (1.0, 0.0, 0.1)  # 基座 → 抓取点的方位角 = 0


@pytest.mark.parametrize("heading,flip", [
    (0.0, False),  # 沿径向朝外
    (math.radians(80), False),
    (math.radians(-80), False),
    (math.radians(100), True),
    (math.radians(-100), True),
    (math.pi, True),  # 正好反向
])
def test_flip_iff_relative_heading_beyond_quarter_turn(heading, flip):
    assert SPF.peg_grasp_needs_flip(GRASP_P, _yaw_q(heading), BASE) is flip


def test_relative_angle_wraps_around_pi():
    # 方位角 170°、夹爪朝向 −170°：相对角 = −340° ≡ +20°，不翻
    p = (math.cos(math.radians(170)), math.sin(math.radians(170)), 0.0)
    assert SPF.peg_grasp_needs_flip(p, _yaw_q(math.radians(-170)), BASE) is False


def test_exactly_quarter_turn_does_not_flip():
    # 抓取点在 +y 方向（方位 90°），夹爪朝向 0：相对角恰为 −90°，严格 > 不成立
    assert SPF.peg_grasp_needs_flip((0.0, 1.0, 0.0), np.array([1.0, 0.0, 0.0, 0.0]), BASE) is False


def test_flip_rotates_heading_by_pi_and_normalises():
    q = _yaw_q(math.radians(30)) * 2.0  # 未归一化输入
    out = SPF.flip_grasp_q(q)
    assert out.dtype == np.float32
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)
    # 翻转后再判：原来朝向 30° 不翻，翻转后朝向 210° 相对角 −150° 应判翻
    assert SPF.peg_grasp_needs_flip(GRASP_P, q / np.linalg.norm(q), BASE) is False
    assert SPF.peg_grasp_needs_flip(GRASP_P, out, BASE) is True
    # 翻两次回到原姿态（四元数差一个整体符号视为同一姿态）
    twice = SPF.flip_grasp_q(out).astype(np.float64)
    ref = q / np.linalg.norm(q)
    assert min(np.linalg.norm(twice - ref), np.linalg.norm(twice + ref)) < 1e-6
