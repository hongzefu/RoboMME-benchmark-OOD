"""PickHighlight 原生三档真值表（C05）：先按按钮，再把每个高亮方块都拾起一次（中间放下）。

期望：按钮 → 依次拾起每个高亮块 → 成功；漏一个高亮块 → 不成功；拾非高亮块（在拾取子任务中）→ 失败；
同一高亮块拾两次不算第二个。

官方现状（契约增量登记为 conditional，计划 Q13「PickHighlight 失败回调与语言冲突」）：
- 按钮子任务的 failure_func 不是 lambda，而是在 _load_scene 时求值一次的常量 False，所以按按钮之前拾起任何方块都不判失败；
- 成功只看「每个高亮块被拾起过 >= 1 次」的计数，不要求按过按钮——不按按钮、直接拾完所有高亮块也判成功；
- 第二条目标语言说「finally press the button again to stop」，但任务表里没有末尾按钮子任务。
"""
from __future__ import annotations

import pytest

from _official_world import OfficialWorld, goal_text

TASK = "PickHighlight"
DIFFS = ("easy", "medium", "hard")


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _press(ep):
    ep.press(ep.env.button)
    ep.step()
    ep.unpress(ep.env.button)


def _pick_targets(ep, targets):
    for i, cube in enumerate(targets):
        ep.grasp(cube)
        ep.step()
        if i != len(targets) - 1:
            ep.release(cube)
            ep.step()


@pytest.mark.parametrize("diff", DIFFS)
def test_button_then_all_highlighted_succeeds(world, diff):
    ep = world.make(diff, seed=7)
    env = ep.env
    assert len(env.target_cubes) == env.configs[diff]["pickup"]
    _press(ep)
    _pick_targets(ep, env.target_cubes[:-1])
    if len(env.target_cubes) > 1:
        ep.release(env.target_cubes[-2])
        ep.step()
        assert not ep.success  # 漏最后一个：不成功
    _pick_targets(ep, env.target_cubes[-1:])
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", ("medium", "hard"))
def test_repick_same_highlight_is_not_second(world, diff):
    ep = world.make(diff, seed=7)
    env = ep.env
    _press(ep)
    first = env.target_cubes[0]
    for _ in range(3):
        ep.grasp(first)
        ep.step()
        ep.release(first)
        ep.step()
    assert not ep.success and not ep.fail
    assert env.target_cube_pickup_counts[env.target_cube_names[0]] == 3
    assert all(v == 0 for k, v in env.target_cube_pickup_counts.items() if k != env.target_cube_names[0])


@pytest.mark.parametrize("diff", DIFFS)
def test_non_highlighted_pick_fails_in_pick_phase(world, diff):
    ep = world.make(diff, seed=7)
    env = ep.env
    others = [c for c in env.all_cubes if all(c is not t for t in env.target_cubes)]
    assert others
    _press(ep)
    ep.grasp(others[0])
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_status_quo_button_failure_is_constant(world, diff):
    """按钮子任务的失败条件是加载时求出的常量：按按钮前拾非高亮块不判失败。"""
    ep = world.make(diff, seed=7)
    env = ep.env
    assert env.task_list[0]["failure_func"] is False
    others = [c for c in env.all_cubes if all(c is not t for t in env.target_cubes)]
    ep.grasp(others[0])
    ep.step()
    assert not ep.fail


@pytest.mark.parametrize("diff", DIFFS)
def test_status_quo_success_without_button(world, diff):
    """不按按钮、直接把所有高亮块各拾一次 → 判成功（与「first press the button」语言冲突）。"""
    ep = world.make(diff, seed=7)
    env = ep.env
    assert all("first press the button" in g for g in goal_text(env))
    assert any("press the button again" in g for g in goal_text(env))
    assert all("button" not in (t.get("name") or "") for t in env.task_list[1:])  # 末尾没有按钮子任务
    _pick_targets(ep, env.target_cubes)
    assert ep.success and ep.task_index == 0
