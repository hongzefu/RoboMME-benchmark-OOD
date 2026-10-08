"""PickXtimes 原生三档真值表（C05）：N 次完整拾放后按按钮才成功。

期望来自任务定义本身（目标语言「repeating this action N times, then press the button」）：
N 次拾放 + 按钮 → 成功；N−1 次就按、N+1 次多拾、抓错对象、持物按按钮 → 失败；做完 N 次不按 → 进行中。
N 读自本局 env.num_repeats，并与目标语言里的次数词交叉核对（同一局的语言与判定绑定）。
"""
from __future__ import annotations

import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_hard.unit.robomme import official_thresholds as T

TASK = "PickXtimes"
DIFFS = ("easy", "medium", "hard")
# 独立期望：英文基数词（只用于核对目标语言里的次数，来源：英语）
WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _cycle(ep, env):
    ep.grasp(env.target_cube)
    ep.step()
    ep.place_on(env.target_cube, env.target)
    ep.step()


@pytest.mark.parametrize("diff", DIFFS)
def test_n_cycles_then_button_succeeds(world, diff):
    ep = world.make(diff, seed=3)
    env = ep.env
    n = env.num_repeats
    goals = goal_text(env)
    if n > 1:
        assert all(WORDS[n] in g for g in goals), goals
    else:
        assert all("repeating" not in g for g in goals), goals
    for _ in range(n):
        _cycle(ep, env)
        assert not ep.success and not ep.fail
    ep.step()
    assert not ep.success and not ep.fail  # 做完 N 次不按按钮：未完成不成功
    ep.press(env.button)
    ep.step()
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", DIFFS)
def test_button_after_n_minus_1_cycles_fails(world, diff):
    ep = world.make(diff, seed=3)
    env = ep.env
    for _ in range(env.num_repeats - 1):
        _cycle(ep, env)
    ep.press(env.button)
    ep.step()
    assert ep.fail and not ep.success
    ep.unpress(env.button)
    ep.step(3)
    assert ep.fail and not ep.success  # 失败终态稳定


@pytest.mark.parametrize("diff", DIFFS)
def test_extra_pickup_after_n_cycles_fails(world, diff):
    ep = world.make(diff, seed=3)
    env = ep.env
    for _ in range(env.num_repeats):
        _cycle(ep, env)
    ep.grasp(env.target_cube)  # 第 N+1 次拾起：按钮子任务的失败条件
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", ("medium", "hard"))
def test_wrong_object_fails(world, diff):
    ep = world.make(diff, seed=3)
    env = ep.env
    assert env.non_target_cubes, "medium/hard 有干扰块"
    ep.grasp(env.non_target_cubes[0])
    ep.step()
    assert ep.fail and not ep.success


def test_easy_has_single_color_no_distractor(world):
    ep = world.make("easy", seed=3)
    assert ep.env.non_target_cubes == []
    assert len(ep.env.all_cubes) == 1


@pytest.mark.parametrize("diff", DIFFS)
def test_press_button_while_holding_fails(world, diff):
    ep = world.make(diff, seed=3)
    env = ep.env
    for _ in range(env.num_repeats - 1):
        _cycle(ep, env)
    ep.grasp(env.target_cube)
    ep.step()
    assert not ep.fail
    ep.press(env.button)  # 持物（处于「放到 target」子任务）时按钮
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_place_off_target_does_not_advance(world, diff):
    ep = world.make(diff, seed=3)
    env = ep.env
    ep.grasp(env.target_cube)
    ep.step()
    before = ep.task_index
    tx, ty, _ = env.target.xyz
    ep.release(env.target_cube, tx + 4 * T.DROP_ONTO_XY, ty)  # 远在放置阈值之外
    ep.step()
    assert ep.task_index == before and not ep.success and not ep.fail


def test_drop_distance_boundary(world):
    """is_obj_dropped_onto 的水平距离阈值 T.DROP_ONTO_XY（<= 判定）：阈值内侧推进、外侧不推进。"""
    for offset, advances in ((T.DROP_ONTO_XY - T.EPS, True), (T.DROP_ONTO_XY + T.EPS, False)):
        ep = world.make("easy", seed=3)
        env = ep.env
        ep.grasp(env.target_cube)
        ep.step()
        before = ep.task_index
        tx, ty, _ = env.target.xyz
        ep.release(env.target_cube, tx + offset, ty)
        ep.step()
        assert (ep.task_index == before + 1) is advances, offset


def test_rebuild_clears_episode_state(world):
    """重建（新一局）后子任务指针与失败标志从零开始。"""
    ep = world.make("easy", seed=3)
    ep.press(ep.env.button)
    ep.step()
    assert ep.fail
    ep2 = world.make("easy", seed=3)
    assert ep2.task_index == 0
    ep2.step()
    assert not ep2.fail and not ep2.success
