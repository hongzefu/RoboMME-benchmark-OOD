"""Peg grasp orientation reduction (``utils/subgoal_planner_func.peg_grasp_needs_flip`` and ``flip_grasp_q``): hand-computed quaternion examples.

Criterion: flip by Rz(π) only when the horizontal heading of the gripper's local x axis deviates more than ±90° from the "base → grasp point" azimuth; exactly 90° does not flip (strict ``>``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from robomme_ood.robomme_env.utils import subgoal_planner_func as SPF


def _yaw_q(theta):
    """Unit quaternion (w, x, y, z) for rotation by theta about world z: the gripper x-axis horizontal heading is theta."""
    return np.array([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)])


BASE = (0.0, 0.0)
GRASP_P = (1.0, 0.0, 0.1)  # azimuth of base → grasp point = 0


@pytest.mark.parametrize("heading,flip", [
    (0.0, False),  # radially outward
    (math.radians(80), False),
    (math.radians(-80), False),
    (math.radians(100), True),
    (math.radians(-100), True),
    (math.pi, True),  # exactly opposite
])
def test_flip_iff_relative_heading_beyond_quarter_turn(heading, flip):
    assert SPF.peg_grasp_needs_flip(GRASP_P, _yaw_q(heading), BASE) is flip


def test_relative_angle_wraps_around_pi():
    # azimuth 170°, gripper heading −170°: relative angle = −340° ≡ +20°, no flip
    p = (math.cos(math.radians(170)), math.sin(math.radians(170)), 0.0)
    assert SPF.peg_grasp_needs_flip(p, _yaw_q(math.radians(-170)), BASE) is False


def test_exactly_quarter_turn_does_not_flip():
    # grasp point in the +y direction (azimuth 90°), gripper heading 0: relative angle exactly −90°, strict > does not hold
    assert SPF.peg_grasp_needs_flip((0.0, 1.0, 0.0), np.array([1.0, 0.0, 0.0, 0.0]), BASE) is False


def test_flip_rotates_heading_by_pi_and_normalises():
    q = _yaw_q(math.radians(30)) * 2.0  # unnormalized input
    out = SPF.flip_grasp_q(q)
    assert out.dtype == np.float32
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)
    # judge again after flipping: heading 30° originally does not flip; after flipping heading 210° relative angle −150° should flip
    assert SPF.peg_grasp_needs_flip(GRASP_P, q / np.linalg.norm(q), BASE) is False
    assert SPF.peg_grasp_needs_flip(GRASP_P, out, BASE) is True
    # flipping twice returns to the original orientation (quaternions differing by an overall sign are the same orientation)
    twice = SPF.flip_grasp_q(out).astype(np.float64)
    ref = q / np.linalg.norm(q)
    assert min(np.linalg.norm(twice - ref), np.linalg.norm(twice + ref)) < 1e-6
