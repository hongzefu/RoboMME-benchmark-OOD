"""VideoRepick 原生三档真值表（C05、C06）：演示里拾放过的那一块，交换后仍按身份再拾放 N 次，最后按按钮。

演示段整段用真实 evaluate/step 驱动：演示拾起 → 放下 →（easy/medium）静止 20 步 → 交换（真实 swap_flat_two_lane）
→ 复位检查；hard 档无交换。期望：
- 在线段 N 次「拾起同一块 → 放下」后按按钮 → 成功；演示里那一次不计入 N；
- 拿起别的方块（包括交换后占着它原位置的那一块）→ 失败；
- 拾放子任务期间按钮：在计时窗口 T.REPICK_BUTTON_WINDOW 内按下 → 失败；窗口开始前按下不判失败（官方现状）。
"""
from __future__ import annotations

import numpy as np
import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_hard.unit.robomme import official_thresholds as T

TASK = "VideoRepick"
DIFFS = ("easy", "medium", "hard")
WORDS = {2: "two", 3: "three"}


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _drive_demo(ep):
    env = ep.env
    target = env.target_cube_1
    ep.grasp(target)
    ep.step()
    ep.release(target)
    ep.step()
    drop_xy = target.xyz[:2].copy()
    guard = 0
    while ep.task_index < ep.first_online_index():
        ep.step()
        guard += 1
        assert guard < 1000
    ep.step()  # 越过最后一次交换在下一步开头的定位
    return drop_xy


def _cycle(ep, cube):
    ep.grasp(cube)
    ep.step()
    ep.release(cube)
    ep.step()


@pytest.mark.parametrize("diff", DIFFS)
def test_repick_same_cube_n_times_then_button_succeeds(world, diff):
    ep = world.make(diff, seed=6)
    env = ep.env
    n = env.num_repeats
    goals = goal_text(env)
    if n > 1:
        assert any(f"{WORDS[n]} times" in g or (n == 2 and "twice" in g) for g in goals), goals
    else:
        assert all("again" in g for g in goals), goals
    _drive_demo(ep)
    assert not ep.success and not ep.fail
    for _ in range(n):
        _cycle(ep, env.target_cube_1)
    assert not ep.success
    ep.press(env.button_left)
    ep.step()
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", ("easy", "medium"))
def test_swap_happens_and_involves_target(world, diff):
    ep = world.make(diff, seed=6)
    env = ep.env
    assert env.swap_times >= 1
    _drive_demo(ep)
    a, b = env.swap_schedule[0][0], env.swap_schedule[0][1]
    assert a is env.target_cube_1 and b is not None and b is not a  # 第一对交换总含目标块
    # 多次交换后目标块可能被换回原处，所以「被换走」只在下一条用例里按布局挑选


@pytest.mark.parametrize("diff", ("easy", "medium"))
def test_cube_at_original_position_is_wrong(world, diff):
    for seed in range(40):
        ep = world.make(diff, seed=seed)
        env = ep.env
        drop_xy = _drive_demo(ep)
        impostor = min(env.spawned_cubes, key=lambda c: np.linalg.norm(c.xyz[:2] - drop_xy))
        if impostor is not env.target_cube_1:
            break
    else:
        pytest.fail("40 个 seed 内目标块都没被换走")
    ep.grasp(impostor)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_demo_pick_not_counted(world, diff):
    """只做 N−1 次在线拾放：计时窗口内按按钮 → 失败（演示那一次不算）。"""
    ep = world.make(diff, seed=6)
    env = ep.env
    _drive_demo(ep)
    start = int(env.elapsed_steps)
    for _ in range(env.num_repeats - 1):
        _cycle(ep, env.target_cube_1)
    ep.step(max(0, T.REPICK_BUTTON_WINDOW[0] + 10 - (int(env.elapsed_steps) - start)))
    ep.press(env.button_left)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("wait, fails", [(T.REPICK_BUTTON_WINDOW[0] - 10, False), (T.REPICK_BUTTON_WINDOW[0] + 10, True)])
def test_button_timewindow_during_repick(world, wait, fails):
    ep = world.make("hard", seed=6)
    env = ep.env
    _drive_demo(ep)
    ep.step(wait)
    ep.press(env.button_left)
    ep.step()
    assert ep.fail is fails and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_wrong_cube_fails(world, diff):
    ep = world.make(diff, seed=6)
    env = ep.env
    _drive_demo(ep)
    other = next(c for c in env.spawned_cubes if c is not env.target_cube_1)
    ep.grasp(other)
    ep.step()
    assert ep.fail and not ep.success


def test_hard_has_no_swap_and_15_cubes(world):
    ep = world.make("hard", seed=6)
    env = ep.env
    assert env.swap_times == 0
    assert len(env.spawned_cubes) == 15  # 5 轮 × 3 色
    assert [t.get("specialflag") for t in env.task_list].count("swap") == 0
