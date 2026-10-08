"""碰撞与障碍几何（``utils/bin_collision.py``、``object_generation._obb2d_intersect``、``xhard.cube_obb2d_exact``、
``InsertPeg`` 之外的二维 OBB 工具）：独立手算算例。

盒体尺寸取 2 的幂（0.25、0.5 m），中心距用精确可表示的值，使判定值的等号边界可逐位核对：
相切（g == 0）归数值边界带而不是放行；穿入记 contact；最小判定值与形状遍历顺序无关。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from robomme_hard.robomme_env.utils import bin_collision as BC
from robomme_hard.robomme_env.utils import object_generation as og
from robomme_hard.robomme_env.utils.xhard import cube_obb2d_exact

I3 = np.eye(3)


def _cube(name, x, y=0.0, half=0.25, yaw=0.0):
    p, q = BC.cube_actor_pose((x, y), yaw, half)
    return BC.ObjectState(name=name, p=p, q=q, shapes=BC.cube_shape_specs(half))


# ── sat_gap：三维分离轴判定值 ────────────────────────────────────────────────


@pytest.mark.parametrize("dx,expected", [(3.0, 1.0), (2.0, 0.0), (1.5, -0.5)])
def test_sat_gap_axis_aligned(dx, expected):
    h = np.ones(3)
    assert BC.sat_gap(np.zeros(3), I3, h, np.array([dx, 0.0, 0.0]), I3, h) == pytest.approx(expected, abs=1e-12)


def test_sat_gap_rotated_box():
    # 绕 z 转 45° 的单位半边盒在 x 轴上的投影半径为 √2
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    g = BC.sat_gap(np.zeros(3), I3, np.ones(3), np.array([4.0, 0.0, 0.0]), rot, np.ones(3))
    assert g == pytest.approx(4.0 - 1.0 - math.sqrt(2.0), abs=1e-12)


def test_sat_gap_nonfinite_is_nan():
    assert math.isnan(BC.sat_gap(np.zeros(3), I3, np.ones(3), np.array([np.inf, 0, 0]), I3, np.ones(3)))


# ── check_pair_static：分类与等号边界 ─────────────────────────────────────────


def test_tangent_is_numerical_boundary_not_clear():
    gap, rej = BC.check_pair_static(_cube("a", 0.0), _cube("b", 0.5))
    assert gap == 0.0 and rej.reason == "numerical_boundary"


def test_within_eps_band_and_clear_beyond():
    _, rej = BC.check_pair_static(_cube("a", 0.0), _cube("b", 0.5 + BC.EPS_M / 2))
    assert rej is not None and rej.reason == "numerical_boundary"
    gap, rej = BC.check_pair_static(_cube("a", 0.0), _cube("b", 0.5 + 2 * BC.EPS_M))
    assert rej is None and gap == pytest.approx(2 * BC.EPS_M, rel=1e-6)


def test_penetration_is_contact():
    gap, rej = BC.check_pair_static(_cube("a", 0.0), _cube("b", 0.4))
    assert rej.reason == "contact" and gap == pytest.approx(-0.1, abs=1e-12)


def test_worst_pair_independent_of_shape_order():
    """两个容器中心距 0.04：前后壁真穿入（g = −0.01），中央方块与前壁那一对恰好相切——必须报 contact。"""
    h = 0.02
    shapes = BC.bin_shape_specs(h)
    a = BC.ObjectState("a", *BC.bin_actor_pose((0.0, 0.0), 0.0, h), shapes=shapes)
    b = BC.ObjectState("b", *BC.bin_actor_pose((0.0, 0.04), 0.0, h), shapes=shapes)
    gap, rej = BC.check_pair_static(a, b)
    rev = BC.ObjectState("b", b.p, b.q, shapes=tuple(reversed(shapes)))
    gap2, rej2 = BC.check_pair_static(a, rev)
    assert rej.reason == rej2.reason == "contact"
    assert gap == pytest.approx(gap2) and gap < 0


def test_missing_shapes_uncertified():
    empty = BC.ObjectState("e", np.zeros(3), np.array([1.0, 0, 0, 0]), shapes=(), radii=(0.0,))
    _, rej = BC.check_pair_static(_cube("a", 0.0), empty)
    assert rej.reason == "uncertified"


# ── check_bin_layout：全场与粗筛 ──────────────────────────────────────────────


def test_layout_reports_the_offending_pair_and_can_raise():
    objs = [_cube("a", 0.0), _cube("b", 2.0), _cube("c", 2.4)]
    gap, rej = BC.check_bin_layout(objs, exhaustive=True)
    assert {rej.object_a, rej.object_b} == {"b", "c"} and rej.reason == "contact"
    with pytest.raises(BC.BinCollisionError):
        BC.check_bin_layout(objs, raise_on_reject=True)


def test_layout_coarse_and_exhaustive_agree_on_verdict():
    objs = [_cube("a", 0.0), _cube("b", 0.75), _cube("c", 3.0)]
    g_fast, r_fast = BC.check_bin_layout(objs)
    g_full, r_full = BC.check_bin_layout(objs, exhaustive=True)
    assert r_fast is None and r_full is None
    assert g_full == pytest.approx(0.25, abs=1e-12) and g_fast >= 0.0


# ── check_swap_sweep：连续交换路径 ────────────────────────────────────────────


def test_swap_clear_without_bystanders_on_lane():
    a, b = _cube("a", 0.0, half=0.02), _cube("b", 0.3, half=0.02)
    far = _cube("far", 0.15, y=1.0, half=0.02)
    _, rej = BC.check_swap_sweep(a, b, [far])
    assert rej is None


def test_swap_blocked_by_bystander_on_a_lane():
    a, b = _cube("a", 0.0, half=0.02), _cube("b", 0.3, half=0.02)
    # 弯道中点在连线中点沿 ±法向偏 LANE_OFFSET（A 走 +、B 走 −）；两侧各放一块必然挡住其中一条
    lanes = [_cube(f"s{k}", 0.15, y=k * BC.LANE_OFFSET, half=0.02) for k in (1, -1)]
    _, rej = BC.check_swap_sweep(a, b, lanes)
    assert rej is not None and rej.stage == "sweep"


# ── 二维 OBB 与静止障碍构造 ────────────────────────────────────────────────────


def test_obb2d_intersect_includes_contact():
    a = og._build_new_cube_obb2d(0.0, 0.0, 0.25, 0.0)
    assert og._obb2d_intersect(*a, *og._build_new_cube_obb2d(0.5, 0.0, 0.25, 0.0)) is True  # 相接算相交
    assert og._obb2d_intersect(*a, *og._build_new_cube_obb2d(0.5 + 1e-9, 0.0, 0.25, 0.0)) is False
    # 转 45° 后对角伸出：中心距 0.25 + 0.25√2 − 0.01 时相交
    rot = og._build_new_cube_obb2d(0.25 + 0.25 * math.sqrt(2) - 0.01, 0.0, 0.25, math.pi / 4)
    assert og._obb2d_intersect(*a, *rot) is True


def test_cube_obb2d_exact_forms_and_guards():
    c, A, h = cube_obb2d_exact((0.1, -0.2, math.pi / 6), 0.02, pad=0.01)
    assert np.allclose(c, [0.1, -0.2]) and np.allclose(h, [0.03, 0.03])
    assert np.allclose(A.T @ A, np.eye(2)) and math.isclose(math.atan2(A[1, 0], A[0, 0]), math.pi / 6)
    same = og._build_new_cube_obb2d(0.1, -0.2, 0.02, math.pi / 6, 0.01)
    assert all(np.array_equal(u, v) for u, v in zip((c, A, h), same)), "与放置逻辑逐位同构"
    for bad in [((0, 0), 0.02, 0.0), ((0, 0, 0), 0.0, 0.0), ((0, 0, 0), 0.02, -1.0)]:
        with pytest.raises(ValueError):
            cube_obb2d_exact(*bad)


def test_static_state_from_obb2d_rejects_skewed_axes_and_bad_z():
    good = (np.array([0.0, 0.0]), np.eye(2), np.array([0.1, 0.2]))
    s = BC.static_state_from_obb2d("o", good, z_range=(0.0, 1.0))
    assert np.allclose(s.p, [0, 0, 0.5]) and np.allclose(s.shapes[0].half, [0.1, 0.2, 0.5])
    with pytest.raises(ValueError):
        BC.static_state_from_obb2d("o", (good[0], np.array([[1.0, 0.5], [0.0, 1.0]]), good[2]), z_range=(0, 1))
    with pytest.raises(ValueError):
        BC.static_rect_state("o", (0, 0), (0.1, 0.1), z_range=(1.0, 1.0))


def test_nearest_partner_first_wins_on_tie():
    idx, table = BC.nearest_partner_index((0.0, 0.0), [(3, (1.0, 0.0)), (5, (0.0, 1.0)), (7, (2.0, 0.0))])
    assert idx == 3 and table == [(3, 1.0), (5, 1.0), (7, 2.0)]
