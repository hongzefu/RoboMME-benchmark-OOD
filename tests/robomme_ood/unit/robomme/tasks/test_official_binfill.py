"""BinFill 原生三档真值表（C05）：按目标语言逐色放入指定数量的方块，再按按钮。

期望：每种颜色放入数量恰好等于目标数（目标语言里写明）→ 按钮成功；少一块就按、多放一块、放错颜色、
提前按 → 失败。放入箱子的方块被移出场景（10, 10），之后不再被计数。
"""
from __future__ import annotations

import numpy as np
import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

TASK = "BinFill"
DIFFS = ("easy", "medium", "hard")
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
# 越过 dynamic 抬起动画窗口（最长 idx*100 步），让替身位置只由测试摆放
ONLINE_START = T.BINFILL_ONLINE_START


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _targets(env):
    return {"red": env.red_cubes_target_number, "blue": env.blue_cubes_target_number,
            "green": env.green_cubes_target_number}


def _cubes(env, color):
    return {"red": env.red_cubes, "blue": env.blue_cubes, "green": env.green_cubes}[color]


def _put_into_bin(ep, env, cube):
    ep.grasp(cube)
    ep.step()
    ep.place_on(cube, env.board_with_hole)
    ep.step()


def _fill_exact(ep, env):
    used = {c: 0 for c in ("red", "blue", "green")}
    for color, count in env.binfill_language_sequence:
        for _ in range(count):
            _put_into_bin(ep, env, _cubes(env, color)[used[color]])
            used[color] += 1
    return used


def _start(world, diff, seed=1):
    ep = world.make(diff, seed=seed)
    ep.skip_demo(elapsed=ONLINE_START)
    return ep


@pytest.mark.parametrize("diff", DIFFS)
def test_exact_counts_then_button_succeeds(world, diff):
    ep = _start(world, diff)
    env = ep.env
    targets = _targets(env)
    goals = goal_text(env)
    for color, n in targets.items():
        if n > 0:
            assert all(f"{WORDS[n]} {color} cube" in g for g in goals), (color, n, goals)
    _fill_exact(ep, env)
    assert [env.red_cubes_in_bin, env.blue_cubes_in_bin, env.green_cubes_in_bin] == [
        targets["red"], targets["blue"], targets["green"]]
    assert not ep.success and not ep.fail
    ep.press(env.button)
    ep.step()
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", DIFFS)
def test_cube_put_into_bin_is_removed_from_scene(world, diff):
    ep = _start(world, diff)
    env = ep.env
    color, _ = env.binfill_language_sequence[0]
    cube = _cubes(env, color)[0]
    _put_into_bin(ep, env, cube)
    np.testing.assert_allclose(cube.xyz, T.BINFILL_REMOVED_XYZ, atol=1e-6)
    ep.step(3)  # 已移出的方块不再重复计数
    assert sum([env.red_cubes_in_bin, env.blue_cubes_in_bin, env.green_cubes_in_bin]) == 1


@pytest.mark.parametrize("diff", DIFFS)
def test_button_one_short_fails(world, diff):
    ep = _start(world, diff)
    env = ep.env
    seq = env.binfill_language_sequence
    total = sum(n for _, n in seq)
    done = 0
    used = {c: 0 for c in ("red", "blue", "green")}
    for color, count in seq:
        for _ in range(count):
            if done == total - 1:
                break
            _put_into_bin(ep, env, _cubes(env, color)[used[color]])
            used[color] += 1
            done += 1
    ep.press(env.button)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_extra_cube_after_exact_fails(world, diff):
    ep = _start(world, diff)
    env = ep.env
    used = _fill_exact(ep, env)
    spare = next((c for c in env.all_cubes if c.xyz[0] < 5 and all(
        c is not x for col in used for x in _cubes(env, col)[:used[col]])), None)
    assert spare is not None, "spawn 数 >= 目标数，至少多一块可放"
    _put_into_bin(ep, env, spare)
    assert ep.fail and not ep.success


def test_wrong_color_fails_at_button(world):
    """medium/hard 才有多色；选一个目标色之外有块的颜色放入，计数不符 → 按钮失败。"""
    for seed in range(40):
        ep = _start(world, "hard", seed=seed)
        env = ep.env
        targets = _targets(env)
        wrong = [c for c in ("red", "blue", "green") if targets[c] == 0 and _cubes(env, c)]
        if wrong:
            break
    else:
        pytest.fail("找不到有非目标色方块的布局")
    seq = env.binfill_language_sequence
    # 按序列的总数放，但第一块换成错色
    total = sum(n for _, n in seq)
    wrong_cube = _cubes(env, wrong[0])[0]
    _put_into_bin(ep, env, wrong_cube)
    used = {c: 0 for c in ("red", "blue", "green")}
    placed = 1
    for color, count in seq:
        for _ in range(count):
            if placed >= total:
                break
            _put_into_bin(ep, env, _cubes(env, color)[used[color]])
            used[color] += 1
            placed += 1
    assert not ep.fail  # 放的过程中不判失败（拾取子任务接受任意方块）
    ep.press(env.button)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_early_button_fails(world, diff):
    ep = _start(world, diff)
    ep.press(ep.env.button)
    ep.step()
    assert ep.fail and not ep.success
    ep.unpress(ep.env.button)
    ep.step(2)
    assert ep.fail  # 失败锁存


def test_drop_outside_bin_does_not_count(world):
    ep = _start(world, "easy")
    env = ep.env
    color, _ = env.binfill_language_sequence[0]
    cube = _cubes(env, color)[0]
    ep.grasp(cube)
    ep.step()
    bx, by, _ = env.board_with_hole.xyz
    ep.release(cube, bx + T.DROP_ONTO_XY + T.EPS, by)  # 刚出放置阈值
    ep.step()
    assert sum([env.red_cubes_in_bin, env.blue_cubes_in_bin, env.green_cubes_in_bin]) == 0


def test_drop_with_closed_gripper_does_not_count(world):
    """check_block_away_gripper：夹爪未张开（两指 <= T.GRIPPER_OPEN）时不计入箱子。"""
    ep = _start(world, "easy")
    env = ep.env
    color, _ = env.binfill_language_sequence[0]
    cube = _cubes(env, color)[0]
    ep.grasp(cube)
    ep.step()
    ep.close_gripper()
    ep.place_on(cube, env.board_with_hole)
    ep.step()
    assert sum([env.red_cubes_in_bin, env.blue_cubes_in_bin, env.green_cubes_in_bin]) == 0
    ep.open_gripper()
    ep.step()
    assert sum([env.red_cubes_in_bin, env.blue_cubes_in_bin, env.green_cubes_in_bin]) == 1
