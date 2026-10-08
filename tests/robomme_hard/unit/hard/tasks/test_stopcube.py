"""StopCube 新值档真值表（xhard1～xhard5）：方块沿直线往返，第 N 次经过目标时按按钮停住为成功。

经任务类真实 ``step``（方块运动 ``move_straight_line`` 与「按下即停」都在 step 里）推进时钟：
在第 N 次经过的窗口中点按下 → 成功；在第 N−1 次经过时按下 → 失败；一直不按、超过第 N 段 → 失败；
成功后继续推进终态不变。N、节拍取自本局（包内规格回放）。
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world

TASK = "StopCube"
TIERS = O.tiers_of(TASK)


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _hover(w):
    """第一项「移到按钮上方」：TCP 停在按钮正上方低处。"""
    x, y = w.xyz(w.env.button)[:2]
    w.tcp_to((x, y, 0.1))


def _run_until(w, step_index):
    out = None
    while int(w.env.elapsed_steps) < step_index:
        out = w.step()
        if out["fail"]:
            return out
    return out


def _press_at(w, step_index):
    _hover(w)
    out = _run_until(w, step_index)
    if out is not None and out["fail"]:
        return out
    w.press(w.env.button)
    return w.step()


@pytest.mark.parametrize("tier", TIERS)
def test_press_on_nth_pass_succeeds(world, tier):
    w = world(tier)
    t, n = w.env.move_interval, w.env.stop_time
    out = _press_at(w, (n - 1) * t + t // 2)  # 第 n 段中点：方块正经过目标
    assert out == {"success": True, "fail": False}
    lo, hi = w.env.stop_time_range
    assert lo <= w.env.stop_timestep <= hi
    for _ in range(3):
        assert w.step()["success"] is True, "成功后继续推进终态不变"


@pytest.mark.parametrize("tier", TIERS)
def test_press_on_previous_pass_fails(world, tier):
    w = world(tier)
    t, n = w.env.move_interval, w.env.stop_time
    out = _press_at(w, (n - 2) * t + t // 2)  # 第 n−1 次经过
    w.unpress(w.env.button)
    out = w.step() if not out["fail"] else out
    # 停在了目标上但不是第 n 次：要么当场判错（按下时不在目标上），要么走完任务表后计时不符
    for _ in range(t * 2):
        if out["fail"]:
            break
        out = w.step()
    assert out == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:2])
def test_never_pressing_fails_after_window(world, tier):
    w = world(tier)
    t, n = w.env.move_interval, w.env.stop_time
    _hover(w)
    out = _run_until(w, n * t + 2)
    assert out == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_press_off_target_fails_immediately(world, tier):
    """方块不在目标上时按下（段起点，方块在路线端点）→ 当场失败。"""
    w = world(tier)
    t, n = w.env.move_interval, w.env.stop_time
    out = _press_at(w, (n - 1) * t + 1)
    assert out == {"success": False, "fail": True}
