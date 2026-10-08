"""VideoUnmaskSwap 与 ButtonUnmaskSwap 原生三档真值表（C05、C06 场景运动）。

- 交换动画用真实 ``swap_flat_two_lane``（替身 actor 只提供 pose/set_pose）：每次交换结束时两只容器恰好互换位置，
  其余容器原地不动；所有交换结束后容器位置是初始位置的一个排列。
- 判定按「容器身份」而不是「原来的位置」：跟着被遮挡的那只容器走才算对；拿起现在占着原位置的另一只 → 失败。
- ButtonUnmaskSwap：两个按钮任意顺序各按一次（按过的从列表里移除）；同一按钮按两次不推进；
  同一步两个按钮同时按下 → 第一个子任务一次性吃掉两个按钮，第二个子任务永远无法完成（官方现状，登记 conditional）。
"""
from __future__ import annotations

import numpy as np
import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_hard.unit.robomme import official_thresholds as T

DIFFS = ("easy", "medium", "hard")
BIN_UP = T.BIN_UP_Z


def _xy(a):
    return a.xyz[:2].copy()


def _run_until(ep, pre_elapsed):
    """单步推进直到「下一步开始时的 elapsed」== pre_elapsed。"""
    while int(ep.env.elapsed_steps) < pre_elapsed:
        ep.step()


def _demo_done_vus(ep):
    """跑完演示段（静止观看到最后一次交换结束）；再多走到 elapsed > 交换结束步，
    让最后一次交换在「下一步开头」的定位落地后，在线动作才不被它覆盖。"""
    guard = 0
    while ep.task_index < 1:
        ep.step()
        guard += 1
        assert guard < 600
    assert int(ep.env.elapsed_steps) >= _swap_end(ep.env)  # 切换时点精确值见 test_vus_still_demo_one_step_before_swap_end
    _run_until(ep, _swap_end(ep.env) + 1)


def _swap_end(env):
    return int(env.swap_schedule[-1][3])


def _pick_sequence(ep, env):
    ep.grasp(env.selected_bins[0], z=BIN_UP)
    ep.step()
    if env.pick_times == 2:
        ep.release(env.selected_bins[0])
        ep.step()
        ep.grasp(env.selected_bins[1], z=BIN_UP)
        ep.step()


# --------------------------------------------------------------------------- VideoUnmaskSwap


@pytest.fixture
def vus():
    with OfficialWorld("VideoUnmaskSwap") as w:
        yield w


@pytest.mark.parametrize("diff", DIFFS)
def test_vus_swap_mechanics(vus, diff):
    ep = vus.make(diff, seed=4)
    env = ep.env
    n_swaps = len(env.swap_schedule)
    assert n_swaps == env.swap_times
    origin = {id(b): _xy(b) for b in env.spawned_bins}
    start1, end1 = int(env.swap_schedule[0][2]), int(env.swap_schedule[0][3])
    _run_until(ep, start1)
    snap = {id(b): _xy(b) for b in env.spawned_bins}
    for b in env.spawned_bins:  # 揭示动画结束后都回到原位
        np.testing.assert_allclose(snap[id(b)], origin[id(b)], atol=1e-6)
    _run_until(ep, end1 + 1)
    a, b = env.swap_schedule[0][0], env.swap_schedule[0][1]
    assert a is not None and b is not None and a is not b
    np.testing.assert_allclose(_xy(a), snap[id(b)], atol=1e-5)
    np.testing.assert_allclose(_xy(b), snap[id(a)], atol=1e-5)
    if n_swaps == 1:
        for o in env.spawned_bins:
            if o is not a and o is not b:
                np.testing.assert_allclose(_xy(o), snap[id(o)], atol=1e-6)
    _demo_done_vus(ep)
    final = sorted(map(tuple, np.round([_xy(o) for o in env.spawned_bins], 5)))
    initial = sorted(map(tuple, np.round(list(origin.values()), 5)))
    # 交换只是排列（连续交换的端点捕获有亚毫米级漂移，容器间距 >= 0.1 m，1 mm 容差足以区分）
    np.testing.assert_allclose(final, initial, atol=1e-3)


@pytest.mark.parametrize("diff", DIFFS)
def test_vus_follow_identity_succeeds(vus, diff):
    ep = vus.make(diff, seed=4)
    env = ep.env
    goals = goal_text(env)
    assert all(f"hiding the {env.color_names[0]} cube" in g for g in goals)
    _demo_done_vus(ep)
    _pick_sequence(ep, env)
    assert ep.success and not ep.fail


def test_vus_original_position_is_wrong():
    """找一个目标容器被交换走的布局：拿起现在占着它原位置的那只 → 失败。"""
    with OfficialWorld("VideoUnmaskSwap") as world:
        for seed in range(40):
            ep = world.make("hard", seed=seed)
            env = ep.env
            target = env.selected_bins[0]
            start_xy = _xy(target)
            _demo_done_vus(ep)
            impostor = min(env.spawned_bins, key=lambda o: np.linalg.norm(_xy(o) - start_xy))
            if impostor is not target:
                break
        else:
            pytest.fail("40 个 seed 内目标容器都没被交换走")
        ep.grasp(impostor, z=BIN_UP)
        ep.step()
        assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", ("medium", "hard"))
def test_vus_empty_distractor_container_fails(vus, diff):
    ep = vus.make(diff, seed=4)
    env = ep.env
    assert len(env.spawned_bins) == 4
    empty = [b for b in env.spawned_bins if all(b is not s for s in env.selected_bins)]
    assert len(empty) == 1  # 4 只容器只藏 3 个方块
    _demo_done_vus(ep)
    ep.grasp(empty[0], z=BIN_UP)
    ep.step()
    assert ep.fail and not ep.success


def test_vus_two_picks_reversed_fails(vus):
    ep = vus.make("hard", seed=4)
    env = ep.env
    assert env.pick_times == 2
    _demo_done_vus(ep)
    ep.grasp(env.selected_bins[1], z=BIN_UP)
    ep.step()
    assert ep.fail and not ep.success


def test_vus_still_demo_one_step_before_swap_end(vus):
    ep = vus.make("easy", seed=4)
    env = ep.env
    _run_until(ep, _swap_end(env) - 1)
    assert ep.task_index == 0  # 最后一次交换结束前仍是演示段
    ep.step()
    assert ep.task_index == 1


# --------------------------------------------------------------------------- ButtonUnmaskSwap


@pytest.fixture
def bus():
    with OfficialWorld("ButtonUnmaskSwap") as w:
        yield w


REVEAL_END = T.REVEAL_END_STEP + 1  # 越过揭示动画窗口


def _press_both(ep, env, order=("button_left", "button_right")):
    for name in order:
        btn = getattr(env, name)
        ep.press(btn)
        ep.step()
        ep.unpress(btn)


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("order", [("button_left", "button_right"), ("button_right", "button_left")])
def test_bus_two_buttons_then_follow_identity_succeeds(bus, diff, order):
    ep = bus.make(diff, seed=4)
    env = ep.env
    ep.step(REVEAL_END)
    _press_both(ep, env, order)
    assert ep.task_index == 2
    _run_until(ep, _swap_end(env) + 2)
    _pick_sequence(ep, env)
    assert ep.success and not ep.fail


def test_bus_same_button_twice_does_not_advance(bus):
    ep = bus.make("easy", seed=4)
    env = ep.env
    ep.step(REVEAL_END)
    _press_both(ep, env, ("button_left", "button_left"))
    assert ep.task_index == 1
    assert env.button_list == [env.button_right]


def test_bus_both_buttons_same_step_blocks_second_subgoal(bus):
    """官方现状：同一步两个按钮同时按下，第一个子任务把两个按钮都从列表移除，第二个子任务永远不完成。"""
    ep = bus.make("easy", seed=4)
    env = ep.env
    ep.step(REVEAL_END)
    ep.press(env.button_left)
    ep.press(env.button_right)
    ep.step()
    ep.unpress(env.button_left)
    ep.unpress(env.button_right)
    assert ep.task_index == 1 and env.button_list == []
    _run_until(ep, _swap_end(env) + 2)
    _press_both(ep, env)
    _pick_sequence(ep, env)
    assert not ep.success and ep.task_index == 1


def test_bus_one_button_then_pick_is_not_success(bus):
    ep = bus.make("medium", seed=4)
    env = ep.env
    ep.step(REVEAL_END)
    _press_both(ep, env, ("button_left",))
    _run_until(ep, _swap_end(env) + 2)
    ep.grasp(env.selected_bins[0], z=BIN_UP)
    ep.step(2)
    assert not ep.success and not ep.fail and ep.task_index == 1


def test_bus_button_list_only_restored_by_rebuild(bus):
    """按钮列表在 _load_scene 里建、_initialize_episode 不重置：只重跑 _initialize_episode 时列表残留（官方现状）；
    基准每局重新 gym.make（重跑 _load_scene），新一局两个按钮都在。"""
    ep = bus.make("easy", seed=4)
    env = ep.env
    ep.step(REVEAL_END)
    _press_both(ep, env)
    assert env.button_list == []
    import torch

    env._initialize_episode(torch.arange(1), {})
    assert env.button_list == []
    ep2 = bus.make("easy", seed=4)
    assert ep2.env.button_list == [ep2.env.button_left, ep2.env.button_right]
