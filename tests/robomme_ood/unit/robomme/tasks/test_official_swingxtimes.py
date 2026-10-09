"""SwingXtimes 原生三档真值表（C05）：拾起 → 右→左 往返 N 轮 → 放下 → 按钮。

判定要点（期望由任务定义与手写事件序列得出）：
- 「到达目标上方」= 水平距离 <= T.SWING_ENTER_XY 且高度 < T.SWING_ENTER_Z；
- 每次「进入」右／左目标各记一次摆动，停留不重复计（离开阈值 T.SWING_EXIT_XY 的迟滞）；摆动总数 > 2N 即失败；
- 左右反序不推进子任务；按钮前未放下、或拾干扰块 → 失败。
"""
from __future__ import annotations

import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

TASK = "SwingXtimes"
DIFFS = ("easy", "medium", "hard")
SWING_Z = T.LIFT_Z  # < T.SWING_ENTER_Z


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _over(ep, env, target, dx=0.0, z=SWING_Z):
    x, y, _ = target.xyz
    ep.carry(env.target_cube, x + dx, y, z)
    ep.step()


def _away(ep, env):
    ep.carry(env.target_cube, 0.0, 0.0, T.CARRY_HIGH_Z)
    ep.step()


def _full_rounds(ep, env, rounds):
    for _ in range(rounds):
        _over(ep, env, env.target_right)
        _over(ep, env, env.target_left)


def _finish(ep, env):
    _away(ep, env)
    ep.release(env.target_cube, 0.0, 0.0)
    ep.step()
    ep.press(env.button)
    ep.step()


@pytest.mark.parametrize("diff", DIFFS)
def test_n_round_trips_then_putdown_and_button_succeeds(world, diff):
    ep = world.make(diff, seed=5)
    env = ep.env
    assert env.target_right.xyz[1] < env.target_left.xyz[1]  # 「右」是 y 较小的那个
    n = env.num_repeats
    if n > 1:
        assert any(f"{['', 'one', 'two', 'three'][n]} times" in g for g in goal_text(env))
    ep.grasp(env.target_cube)
    ep.step()
    _full_rounds(ep, env, n)
    assert env.swing_count == 2 * n
    assert not ep.success and not ep.fail
    _finish(ep, env)
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", DIFFS)
def test_left_before_right_does_not_advance_and_overflows(world, diff):
    ep = world.make(diff, seed=5)
    env = ep.env
    ep.grasp(env.target_cube)
    ep.step()
    idx = ep.task_index
    _over(ep, env, env.target_left)  # 反序：先到左
    assert ep.task_index == idx and not ep.fail
    _away(ep, env)
    _full_rounds(ep, env, env.num_repeats)  # 之后再照常做满 N 轮：总摆动 2N+1
    assert env.swing_count == 2 * env.num_repeats + 1
    assert env.swing_over_limit
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_dwelling_counts_once(world, diff):
    ep = world.make(diff, seed=5)
    env = ep.env
    ep.grasp(env.target_cube)
    ep.step()
    for _ in range(5):
        _over(ep, env, env.target_right)
    assert env.swing_count == 1
    # 超出进入阈值、仍在离开阈值内晃动：仍算停留
    _over(ep, env, env.target_right, dx=T.SWING_EXIT_XY - T.EPS)
    assert env.swing_count == 1
    # 真正离开再回来：再计一次
    _away(ep, env)
    _over(ep, env, env.target_right)
    assert env.swing_count == 2


@pytest.mark.parametrize("dx, z, advances", [
    (T.SWING_ENTER_XY - T.EPS, SWING_Z, True), (T.SWING_ENTER_XY + T.EPS, SWING_Z, False),
    (0.0, T.SWING_ENTER_Z - T.EPS, True), (0.0, T.SWING_ENTER_Z + T.EPS, False),
])
def test_swing_thresholds(world, dx, z, advances):
    ep = world.make("easy", seed=5)
    env = ep.env
    ep.grasp(env.target_cube)
    ep.step()
    idx = ep.task_index
    _over(ep, env, env.target_right, dx=dx, z=z)
    assert (ep.task_index == idx + 1) is advances


@pytest.mark.parametrize("diff", DIFFS)
def test_button_without_putdown_fails(world, diff):
    ep = world.make(diff, seed=5)
    env = ep.env
    ep.grasp(env.target_cube)
    ep.step()
    _full_rounds(ep, env, env.num_repeats)
    ep.press(env.button)  # 仍在「放下」子任务，按钮是失败条件
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", ("medium", "hard"))
def test_pick_distractor_fails(world, diff):
    ep = world.make(diff, seed=5)
    env = ep.env
    ep.grasp(env.non_target_cubes[0])
    ep.step()
    assert ep.fail and not ep.success


def test_too_many_rounds_fails(world):
    ep = world.make("easy", seed=5)
    env = ep.env
    ep.grasp(env.target_cube)
    ep.step()
    _full_rounds(ep, env, env.num_repeats + 1)
    ep.step()
    assert ep.fail and not ep.success
