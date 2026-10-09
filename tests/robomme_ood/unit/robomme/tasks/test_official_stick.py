"""PatternLock 与 RouteStick 原生三档真值表（C05、C07 路径与叉积）。

两任务都用 stick：tcp 落到某个目标上方（水平距离阈值内、高度低于阈值）即「触到」该目标。
演示段真实驱动：依次触到演示路径上的目标 → 复位（solve_strong_reset 会置 after_demo）→ 回到起点姿态（swing_qpos）。
- PatternLock：在线段按同一顺序重走；起点必须在在线段被触到（复位回 swing_qpos 时 tcp 就在起点上方），
  最后 len(路径) 个触到记录必须与路径逐项相同；触到路径外的按钮（非期望、非上一个）→ 失败；
  在同一按钮上停留只记一次。
- RouteStick：每一段除了落点正确，还要按演示方向绕行：在线轨迹各点相对「上一目标→本目标」有向线段的
  叉积均值 > 0 为 clockwise、< 0 为 counterclockwise、= 0 为失败；方向不符、触错目标 → 失败，失败锁存。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from _official_world import OfficialWorld, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

DIFFS = ("easy", "medium", "hard")
TOUCH_Z = T.STICK_TOUCH_Z
HIGH_Z = T.STICK_HIGH_Z


def _touch(ep, target):
    x, y, _ = target.xyz
    ep.tcp_to(x, y, TOUCH_Z)
    ep.step()


def _drive_demo(ep, home_on_start=True):
    env = ep.env
    path = env.selected_buttons
    _touch(ep, path[0])
    env.swing_qpos = env.agent.robot.qpos.clone()  # solve_swingonto(record_swing_qpos=True) 记下起点姿态
    for t in path[1:]:
        _touch(ep, t)
    ep.tcp_to(0.0, 0.0, HIGH_Z)
    env.after_demo = True  # solve_strong_reset 置位
    guard = 0
    while ep.task_index < ep.first_online_index() - 1:
        ep.step()
        guard += 1
        assert guard < 50
    if home_on_start:  # 复位回 swing_qpos：tcp 回到起点上方
        x, y, _ = path[0].xyz
        ep.tcp_to(x, y, TOUCH_Z)
    ep.step()
    assert ep.task_index == ep.first_online_index()


# --------------------------------------------------------------------------- PatternLock


@pytest.fixture
def pl():
    with OfficialWorld("PatternLock") as w:
        yield w


@pytest.mark.parametrize("diff", DIFFS)
def test_pl_retrace_succeeds(pl, diff):
    ep = pl.make(diff, seed=9)
    env = ep.env
    grid = len(env.buttons_grid)
    assert int(round(grid ** 0.5)) ** 2 == grid
    _drive_demo(ep)
    for t in env.selected_buttons[1:]:
        _touch(ep, t)
        ep.step()  # 停留一步：同一按钮不重复记录
    assert ep.success and not ep.fail
    assert [a.name for a in env.achieved_list] == [s.name for s in env.selected_buttons]


@pytest.mark.parametrize("diff", DIFFS)
def test_pl_wrong_button_fails(pl, diff):
    ep = pl.make(diff, seed=9)
    env = ep.env
    _drive_demo(ep)
    path = env.selected_buttons
    stray = next(b for b in env.buttons_grid if all(b is not p for p in path))
    _touch(ep, stray)
    assert ep.fail and not ep.success


def test_pl_skipping_a_node_fails(pl):
    for seed in range(40):
        ep = pl.make("hard", seed=seed)
        if len(ep.env.selected_buttons) >= 3:
            break
    env = ep.env
    _drive_demo(ep)
    _touch(ep, env.selected_buttons[2])  # 跳过 selected[1]
    assert ep.fail and not ep.success


def test_pl_suffix_only_is_not_a_match(pl):
    """在线段没触到起点（只重走后缀）：子任务都完成，但最近记录与路径不等 → 失败。"""
    ep = pl.make("easy", seed=9)
    env = ep.env
    _drive_demo(ep, home_on_start=False)
    for t in env.selected_buttons[1:]:
        _touch(ep, t)
    assert ep.fail and not ep.success


def test_pl_demo_touches_are_not_recorded(pl):
    ep = pl.make("medium", seed=9)
    env = ep.env
    path = env.selected_buttons
    for t in path:
        _touch(ep, t)
    assert env.achieved_list == []  # after_demo 之前不记录


# --------------------------------------------------------------------------- RouteStick


@pytest.fixture
def rs():
    with OfficialWorld("RouteStick") as w:
        yield w


def _detour(ep, prev, curr, sign, steps=3):
    """沿 prev→curr 的有向线段走，偏到左法向 sign 一侧（高处，不触碰任何目标），最后落到 curr。"""
    p, c = prev.xyz[:2], curr.xyz[:2]
    line = c - p
    normal = np.array([-line[1], line[0]]) / np.linalg.norm(line)
    for i in range(1, steps + 1):
        q = p + line * i / (steps + 1) + sign * T.DETOUR_OFFSET * normal
        ep.tcp_to(q[0], q[1], HIGH_Z)
        ep.step()
    _touch(ep, curr)


def _sign(direction):
    return 1.0 if direction == "clockwise" else -1.0  # 左法向一侧叉积为正 → clockwise


@pytest.mark.parametrize("diff", DIFFS)
def test_rs_follow_route_with_directions_succeeds(rs, diff):
    ep = rs.make(diff, seed=10)
    env = ep.env
    path = env.selected_buttons
    assert all(env.buttons_grid.index(b) in (0, 2, 4, 6, 8) for b in path)  # 只走凸起的 5 个目标
    assert len(env.swing_directions) == len(path) - 1
    for name, d in zip([t["name"] for t in env.task_list if not t["demonstration"]], env.swing_directions):
        assert name.endswith(d)
    _drive_demo(ep)
    for prev, curr, d in zip(path[:-1], path[1:], env.swing_directions):
        _detour(ep, prev, curr, _sign(d))
        assert not ep.fail
    assert ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_rs_wrong_direction_fails_and_latches(rs, diff):
    ep = rs.make(diff, seed=10)
    env = ep.env
    path = env.selected_buttons
    _drive_demo(ep)
    _detour(ep, path[0], path[1], -_sign(env.swing_directions[0]))
    ep.step()
    assert ep.fail and not ep.success
    ep.step(3)
    assert ep.fail  # 失败锁存


def test_rs_straight_line_is_on_the_line_and_fails(rs):
    ep = rs.make("easy", seed=10)
    env = ep.env
    path = env.selected_buttons
    _drive_demo(ep)
    _detour(ep, path[0], path[1], 0.0)
    ep.step()
    assert ep.fail and not ep.success


def test_rs_touching_unexpected_raised_target_fails(rs):
    ep = rs.make("easy", seed=10)
    env = ep.env
    path = env.selected_buttons
    _drive_demo(ep)
    stray = next(env.buttons_grid[i] for i in (0, 2, 4, 6, 8)
                 if env.buttons_grid[i] is not path[0] and env.buttons_grid[i] is not path[1])
    _touch(ep, stray)
    assert ep.fail and not ep.success


def test_rs_direction_fail_degenerate_inputs(rs):
    """direction_fail 的退化输入：无轨迹、零长线段 → 判失败（返回 False 并置 failureflag）。"""
    ep = rs.make("easy", seed=10)
    env = ep.env
    a, b = env.buttons_grid[0], env.buttons_grid[2]
    env._gripper_xy_trace = []
    env.failureflag = torch.tensor([False])
    assert env.direction_fail([b, a, "clockwise"]) is False and bool(env.failureflag.item())
    env._gripper_xy_trace = [(1, torch.tensor([0.0, 0.0]))]
    env.failureflag = torch.tensor([False])
    assert env.direction_fail([a, a, "clockwise"]) is False and bool(env.failureflag.item())
    assert env.direction_fail(None) is True
