"""VideoUnmask 与 ButtonUnmask 原生三档真值表（C05）。

两任务共用判定：「拿起容器」= 容器高度 > T.BIN_PICKUP_Z（严格大于），「放下」= 高度 <= T.BIN_PUTDOWN_Z、未被抓、tcp 高于 T.PICKUP_Z。
- VideoUnmask：演示段是 T.VIDEO_UNMASK_STATIC_STEPS 步静止观看（期间容器被抬走再放回），之后拿起藏着目标色方块的 bin_0；
  hard 档要两次选择（bin_0 → 放下 → bin_1）。
- ButtonUnmask：先按按钮（深度 > T.BUTTON_DEPTH 才算按下），再同样选择。
目标语言里的颜色必须就是藏在 bin_0／bin_1 下的方块颜色（方块初始 xy 与容器重合）。
"""
from __future__ import annotations

import numpy as np
import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_hard.unit.robomme import official_thresholds as T

DIFFS = ("easy", "medium", "hard")
BIN_UP = T.BIN_UP_Z
REVEAL_END = T.REVEAL_END_STEP + 1  # 越过揭示动画窗口


@pytest.fixture(params=["VideoUnmask", "ButtonUnmask"])
def task(request):
    return request.param


@pytest.fixture
def world(task):
    with OfficialWorld(task) as w:
        yield w


def _to_online(ep, task):
    env = ep.env
    if task == "VideoUnmask":
        # 演示段：机器人静止 T.VIDEO_UNMASK_STATIC_STEPS 步（真实 static_check），子任务指针才进入在线段
        guard = 0
        while ep.task_index < ep.first_online_index():
            ep.step()
            guard += 1
            assert guard < 200
    else:
        # 等过揭示动画（容器被抬到场景外再放回）再按按钮，见 test_button_unmask_press_during_reveal_fails
        ep.step(REVEAL_END)
        ep.press(env.button_left)
        ep.step()
        ep.unpress(env.button_left)
    assert ep.task_index == ep.first_online_index() + (1 if task == "ButtonUnmask" else 0)


def _pick_count(env):
    return env.configs[env.difficulty]["pick"]


def test_goal_colour_is_the_cube_under_bin0(world, task):
    for diff in DIFFS:
        ep = world.make(diff, seed=2)
        env = ep.env
        cube0 = getattr(env, f"target_cube_{env.color_names[0]}")
        np.testing.assert_allclose(cube0.xyz[:2], env.bin_0.xyz[:2], atol=1e-6)
        goals = goal_text(env)
        assert all(f"hiding the {env.color_names[0]} cube" in g for g in goals)
        if _pick_count(env) > 1:
            assert all(f"hiding the {env.color_names[1]} cube" in g for g in goals)


@pytest.mark.parametrize("diff", DIFFS)
def test_correct_selection_succeeds(world, task, diff):
    ep = world.make(diff, seed=2)
    env = ep.env
    _to_online(ep, task)
    ep.grasp(env.bin_0, z=BIN_UP)
    ep.step()
    if _pick_count(env) == 1:
        assert ep.success and not ep.fail
        return
    assert not ep.success
    ep.release(env.bin_0)
    ep.step()
    ep.grasp(env.bin_1, z=BIN_UP)
    ep.step()
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", DIFFS)
def test_wrong_container_fails(world, task, diff):
    ep = world.make(diff, seed=2)
    env = ep.env
    _to_online(ep, task)
    wrong = env.spawned_bins[-1]
    assert wrong is not env.bin_0
    ep.grasp(wrong, z=BIN_UP)
    ep.step()
    assert ep.fail and not ep.success


def test_hard_reversed_order_fails(world, task):
    ep = world.make("hard", seed=2)
    env = ep.env
    _to_online(ep, task)
    ep.grasp(env.bin_1, z=BIN_UP)
    ep.step()
    assert ep.fail and not ep.success


def test_hard_second_pick_without_putdown_fails(world, task):
    ep = world.make("hard", seed=2)
    env = ep.env
    _to_online(ep, task)
    ep.grasp(env.bin_0, z=BIN_UP)
    ep.step()
    env.bin_1.move_to(z=BIN_UP)  # bin_0 还举着就拿第二个
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("z, picked", [(T.BIN_PICKUP_Z + T.EPS, True), (T.BIN_PICKUP_Z, False)])
def test_pickup_height_threshold_is_strict(world, task, z, picked):
    ep = world.make("easy", seed=2)
    env = ep.env
    _to_online(ep, task)
    env.bin_0.move_to(z=z)
    ep.step()
    assert ep.success is picked


def test_video_unmask_motion_restarts_static_window():
    """演示段的 static_check：机器人一动，静止计时重新开始（演示→在线切换的时点）。"""
    with OfficialWorld("VideoUnmask") as world:
        ep = world.make("easy", seed=2)
        ep.step(T.VIDEO_UNMASK_STATIC_STEPS // 2)
        ep.move_robot()
        ep.step()
        ep.hold_robot()
        moved_at = int(ep.env.elapsed_steps)
        while ep.task_index == 0:
            ep.step()
        assert int(ep.env.elapsed_steps) - moved_at >= T.VIDEO_UNMASK_STATIC_STEPS


def test_video_unmask_bins_return_to_origin_after_reveal():
    """揭示窗口里容器被抬走、在窗口中点放回原位；在线段开始时容器都在初始位置。"""
    with OfficialWorld("VideoUnmask") as world:
        ep = world.make("hard", seed=2)
        origin = [b.xyz.copy() for b in ep.env.spawned_bins]
        ep.step(10)
        assert all(b.xyz[2] == T.REVEAL_AWAY_Z for b in ep.env.spawned_bins)  # 揭示中：被抬到场景外
        while ep.task_index == 0:
            ep.step()
        for b, o in zip(ep.env.spawned_bins, origin):
            np.testing.assert_allclose(b.xyz, o, atol=1e-6)


# 按钮深度「严格大于」的契约只在 test_sequential_check.py::test_button_depth_strict 断言（贴近生产函数）。


@pytest.mark.parametrize("press_at, fails", [
    (1, True), (T.REVEAL_DROP_STEP - 1, True), (T.REVEAL_DROP_STEP, False), (T.REVEAL_END_STEP, False),
])
def test_button_unmask_press_during_reveal_fails(press_at, fails):
    """官方现状（契约增量登记为 conditional）：揭示窗口前半段（elapsed < T.REVEAL_DROP_STEP）容器被临时移到场景外高处，
    此时若已按下按钮，下一步「拿起其他容器」的失败条件成立 → fail（随即 terminated）。"""
    with OfficialWorld("ButtonUnmask") as world:
        ep = world.make("easy", seed=2)
        ep.step(press_at - 1)
        ep.press(ep.env.button_left)
        ep.step(2)
        assert any(f for _, f, _ in ep.history) is fails


def test_button_unmask_failure_not_latched_by_env():
    """ButtonUnmask.evaluate 每步重置 failureflag（不锁存）；失败的终局靠 terminated 截断，而不是 env 记忆。"""
    with OfficialWorld("ButtonUnmask") as world:
        ep = world.make("easy", seed=2)
        _to_online(ep, "ButtonUnmask")
        wrong = ep.env.spawned_bins[-1]
        ep.grasp(wrong, z=BIN_UP)
        ep.step()
        assert ep.fail
        ep.release(wrong)
        ep.step()
        assert not ep.fail


def test_button_unmask_pick_before_button_is_not_success():
    """跳过按钮先拿容器：不推进也不判失败（按钮子任务无失败条件）；补按按钮后才成功——官方现状。"""
    with OfficialWorld("ButtonUnmask") as world:
        ep = world.make("easy", seed=2)
        env = ep.env
        ep.step(REVEAL_END)
        ep.grasp(env.bin_0, z=BIN_UP)
        ep.step(3)
        assert not ep.success and not ep.fail and ep.task_index == 0
        ep.press(env.button_left)
        ep.step()
        ep.step()
        assert ep.success
