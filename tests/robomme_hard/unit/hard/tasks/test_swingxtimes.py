"""SwingXtimes 新值档真值表（xhard1～xhard5，N 取自本局 ``num_repeats``）：抓起 → 右→左摆 N 轮 → 放下 → 按钮为成功。

错误与边界：左右反序不推进；在同一侧停留多步只计一次摆动（step 的进出滞回）；摆动总数超过上限即失败；
抓非目标方块、提前按按钮即失败；距离阈值含等号、高度阈值不含等号（阈值改成 float32 精确可表示的值后核对）。
全部经任务类真实 ``step``（摆动计数在 step 里）与 ``evaluate``。
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world

TASK = "SwingXtimes"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}
SWING_Z = 0.08  # 摆到目标上方的高度：低于进入阈值
AWAY_Z = 0.25


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _over(w, target, z=SWING_Z):
    x, y = w.xyz(target)[:2]
    w.move(w.env.target_cube, (x, y, z))
    w.tcp_to((x, y, z))


def _between(w):
    """离开两个目标：移到两目标连线中点上方高处（水平距两目标都远超退出阈值）。"""
    a, b = w.xyz(w.env.target_right)[:2], w.xyz(w.env.target_left)[:2]
    mid = (a + b) / 2
    w.move(w.env.target_cube, (mid[0], mid[1], AWAY_Z))
    w.tcp_to((mid[0], mid[1], AWAY_Z))


def _swing_rounds(w, n):
    for _ in range(n):
        for side in (w.env.target_right, w.env.target_left):
            _over(w, side)
            assert w.step() == OK
            _between(w)
            assert w.step() == OK


def _pick(w):
    w.grasp(w.env.target_cube)
    assert w.step() == OK
    assert w.stage == 1


@pytest.mark.parametrize("tier", TIERS)
def test_n_rounds_then_putdown_then_button_succeeds(world, tier):
    w = world(tier)
    n = w.env.num_repeats
    _pick(w)
    _swing_rounds(w, n)
    assert w.env.swing_count == 2 * n
    w.release_onto(w.env.target_cube, w.xyz(w.env.target_cube)[:2])
    assert w.step() == OK
    w.press(w.env.button)
    assert w.step() == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS)
def test_one_round_short_then_button_fails(world, tier):
    w = world(tier)
    _pick(w)
    _swing_rounds(w, w.env.num_repeats - 1)
    w.press(w.env.button)
    assert w.step()["fail"] is True


@pytest.mark.parametrize("tier", TIERS[:1])
def test_left_before_right_does_not_advance(world, tier):
    w = world(tier)
    _pick(w)
    _over(w, w.env.target_left)
    assert w.step() == OK
    assert w.stage == 1, "先摆左侧不推进（要求右→左）"
    _between(w)
    _over(w, w.env.target_right)
    assert w.step() == OK and w.stage == 2


@pytest.mark.parametrize("tier", TIERS[:1])
def test_staying_on_one_side_counts_once(world, tier):
    w = world(tier)
    _pick(w)
    _over(w, w.env.target_right)
    for _ in range(4):
        w.step()
    assert w.env.swing_count == 1


@pytest.mark.parametrize("tier", TIERS)
def test_exceeding_max_swings_fails(world, tier):
    """摆动次数超过生产上限 ``max_swings`` 即失败（在多摆的那一轮之后）。"""
    w = world(tier)
    _pick(w)
    limit = w.env.max_swings
    out = OK
    sides = [w.env.target_right, w.env.target_left]
    for i in range(limit + 2):
        _over(w, sides[i % 2])
        out = w.step()
        _between(w)
        out = w.step()
        if out["fail"]:
            break
    assert w.env.swing_count > limit
    assert out == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_non_target_pickup_and_early_button_fail(world, tier):
    w = world(tier)
    _pick(w)
    w.agent.held = None
    w.grasp(w.env.distractor_cubes[0])
    assert w.step()["fail"] is True
    w = World.build(TASK, tier)
    _pick(w)
    w.press(w.env.button)
    assert w.step()["fail"] is True


@pytest.mark.parametrize("tier", TIERS[:1])
def test_swing_distance_inclusive_and_height_exclusive(world, tier):
    """阈值等号：任务表的摆动判定读 sampling 参数 ``swing_thresholds``；把它换成 float32 精确可表示的
    1/32 m 与 1/8 m，目标移到原点：水平距离恰为阈值 → 算到达（``<=``）；高度恰为阈值 → 不算（``<``）。"""
    w = world(tier)
    _pick(w)
    thr = w.env._sampling["parameters"]["swing_thresholds"]
    thr["distance"], thr["z"] = 2.0 ** -5, 2.0 ** -3
    right = w.env.target_right
    w.move(right, (0.0, 0.0, 0.0))
    w.move(w.env.target_cube, (2.0 ** -5 + 1e-4, 0.0, 0.1))
    assert w.tick() == OK and w.stage == 1
    w.move(w.env.target_cube, (0.0, 0.0, 2.0 ** -3))
    assert w.tick() == OK and w.stage == 1
    w.move(w.env.target_cube, (2.0 ** -5, 0.0, 0.1))
    assert w.tick() == OK and w.stage == 2
