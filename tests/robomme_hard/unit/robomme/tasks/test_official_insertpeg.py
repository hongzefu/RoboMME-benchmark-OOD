"""InsertPeg 原生三档真值表（C05、C07 几何）。

几何判据（手算摆位）：
- 「按演示的那一端拿起」：抓取端高度 > T.PEG_PICKUP_Z，且 tcp 离抓取端比离插入端近（严格 <）；
- 「从演示的那一侧插入」：插入端到盒子 < T.INSERT_XY、比抓取端更近，且 (tcp_y − box_y)·direction < 0；
  |tcp_y − box_y| < 1e−3 时改用抓取端 y 判侧；
- 换 peg、换端、插错端、反向插入 → 失败。演示段（拿起 → 插入 → 复位 → 静止 100 步）也用这些摆位真实驱动，
  复位那一步用真实 step 里的 reset_in_proecess 分支把三根 peg 放回初始位姿。三档在官方实现里共用同一逻辑。
"""
from __future__ import annotations

import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_hard.unit.robomme import official_thresholds as T

TASK = "InsertPeg"
DIFFS = ("easy", "medium", "hard")
LIFT = T.PEG_LIFT_Z


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _lift_by(ep, grab, other):
    """拿起：两端都抬高，tcp 贴着 grab 端。"""
    gx, gy, _ = grab.xyz
    ox, oy, _ = other.xyz
    grab.move_to(z=LIFT)
    other.move_to(z=LIFT)
    ep.agent.held = grab
    ep.tcp_to(gx, gy, LIFT)


def _insert(ep, insert_end, grab_end, side_sign, gap=0.0):
    """插入端放到盒子中心（偏 gap），抓取端在 y 方向 side_sign 一侧 T.PEG_END_OFFSET，tcp 跟着抓取端。"""
    bx, by, bz = ep.env.box.xyz
    insert_end.move_to(bx + gap, by, bz)
    grab_end.move_to(bx, by + side_sign * T.PEG_END_OFFSET, bz)
    ep.tcp_to(bx, by + side_sign * T.PEG_END_OFFSET, bz + T.PEG_END_OFFSET)


def _correct_side(env):
    return -env.direction  # (tcp_y − box_y)·direction < 0


def _drive_demo(ep):
    env = ep.env
    _lift_by(ep, env.grasp_target, env.insert_target)
    ep.step()
    assert ep.task_index == 1
    _insert(ep, env.insert_target, env.grasp_target, _correct_side(env))
    ep.step()
    assert ep.task_index == 2
    ep.agent.held = None
    env.reset_in_proecess = True  # solve_strong_reset 期间的标志：真实 step 把 peg 放回初始位姿
    ep.step()
    env.reset_in_proecess = False
    guard = 0
    while ep.task_index < ep.first_online_index():
        ep.step()
        guard += 1
        assert guard < 300


@pytest.mark.parametrize("diff", DIFFS)
def test_demo_then_correct_end_and_side_succeeds(world, diff):
    ep = world.make(diff, seed=8)
    env = ep.env
    assert env.insert_way == ("left" if env.direction == -1 else "right")
    assert f"{env.insert_way} side" in env.task_list[-1]["name"]
    assert "same side of the box" in goal_text(env)[0]
    _drive_demo(ep)
    assert not ep.success
    _lift_by(ep, env.grasp_target, env.insert_target)
    ep.step()
    _insert(ep, env.insert_target, env.grasp_target, _correct_side(env))
    ep.step()
    assert ep.success and not ep.fail


def _online(world, diff="easy", seed=8):
    ep = world.make(diff, seed=seed)
    ep.skip_demo()
    return ep, ep.env


@pytest.mark.parametrize("diff", DIFFS)
def test_wrong_end_grasp_fails(world, diff):
    ep, env = _online(world, diff)
    _lift_by(ep, env.insert_target, env.grasp_target)
    ep.step()
    assert ep.fail and not ep.success


def test_other_peg_fails(world):
    ep, env = _online(world)
    other = next(p for p in env.pegs if p is not env.peg)
    _lift_by(ep, other.head, other.tail)
    ep.step()
    assert ep.fail and not ep.success


def test_wrong_side_insert_fails(world):
    ep, env = _online(world)
    _lift_by(ep, env.grasp_target, env.insert_target)
    ep.step()
    _insert(ep, env.insert_target, env.grasp_target, -_correct_side(env))
    ep.step()
    assert ep.fail and not ep.success


def test_wrong_end_insert_fails(world):
    ep, env = _online(world)
    _lift_by(ep, env.grasp_target, env.insert_target)
    ep.step()
    _insert(ep, env.grasp_target, env.insert_target, _correct_side(env))
    ep.step()
    assert ep.fail and not ep.success


def test_equidistant_tcp_does_not_count_as_grasp(world):
    ep, env = _online(world)
    a, b = env.grasp_target, env.insert_target
    a.move_to(z=LIFT)
    b.move_to(z=LIFT)
    mid = (a.xyz + b.xyz) / 2
    ep.tcp_to(*mid)
    ep.step()
    assert ep.task_index == ep.first_online_index() and not ep.fail


@pytest.mark.parametrize("gap, inserted", [(T.INSERT_XY - T.EPS, True), (T.INSERT_XY + T.EPS, False)])
def test_insert_distance_threshold(world, gap, inserted):
    ep, env = _online(world)
    _lift_by(ep, env.grasp_target, env.insert_target)
    ep.step()
    _insert(ep, env.insert_target, env.grasp_target, _correct_side(env), gap=gap)
    ep.step()
    assert ep.success is inserted


@pytest.mark.parametrize("grip_side_ok", [True, False])
def test_direction_near_zero_falls_back_to_grip_end(world, grip_side_ok):
    ep, env = _online(world)
    _lift_by(ep, env.grasp_target, env.insert_target)
    ep.step()
    side = _correct_side(env) if grip_side_ok else -_correct_side(env)
    _insert(ep, env.insert_target, env.grasp_target, side)
    bx, by, bz = env.box.xyz
    ep.tcp_to(bx + T.PEG_END_OFFSET, by + T.DIRECTION_NEAR_ZERO / 2, bz)  # |tcp_y − box_y| < 阈值
    ep.step()
    assert ep.success is grip_side_ok and ep.fail is (not grip_side_ok)


def test_reset_branch_restores_peg_poses(world):
    ep = world.make("easy", seed=8)
    env = ep.env
    init = [p.xyz.copy() for p in env.pegs]
    for p in env.pegs:
        p.set_pose(((T.CARRY_HIGH_Z,) * 3, (1, 0, 0, 0)))  # 任意远离初始位姿的位置
    env.reset_in_proecess = True
    ep.step()
    for p, x in zip(env.pegs, init):
        assert abs(p.xyz - x).max() < 1e-6
