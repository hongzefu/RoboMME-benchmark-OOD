"""StopCube 原生三档真值表（C05、C06 计时与场景运动）。

方块在两端点之间往返（真实 move_straight_line）。到达步、停止窗口与截止步不在测试里按公式推，而是按具体
(move_interval, stop_time) 手算后钉在 ``official_thresholds.STOPCUBE_CASES``；测试按这两个取值挑 seed。期望
（目标语言：「stop the cube … for the k-th time」）：
- 先把 tcp 悬到按钮上方（准备子任务），在第 stop_time 次到达时按下 → 成功，且之后继续步进保持成功；
- 第 stop_time−1 次到达时按 → 失败（停止步不在窗口内）；方块不在目标上时按 → 立即失败；
- 一直不按 → 过了截止步即失败。三档在官方实现里共用同一逻辑（难度不参与取值）。
"""
from __future__ import annotations

import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_hard.unit.robomme import official_thresholds as T

TASK = "StopCube"
DIFFS = ("easy", "medium", "hard")
ORDINALS = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}
CASES = T.STOPCUBE_CASES


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _case_episode(world, diff, case):
    """挑一个 (move_interval, stop_time) 与钉值算例一致的 seed（离线 _initialize_episode，不是仿真 reset）。"""
    for seed in range(64):
        ep = world.make(diff, seed=seed)
        if (ep.env.move_interval, ep.env.stop_time) == (case["move_interval"], case["stop_time"]):
            return ep
    pytest.fail(f"64 个 seed 内找不到 {case}")


def _hover(ep):
    bx, by, _ = ep.env.button.xyz
    ep.tcp_to(bx, by, T.LIFT_Z)


def _run_to(ep, elapsed):
    while int(ep.env.elapsed_steps) < elapsed:
        ep.step()


def _press_at(ep, step):
    """推进到 elapsed == step 再按下按钮（下一步 evaluate 看到的方块位置正是 move_straight_line(cur_step=step) 的位置）。"""
    _run_to(ep, step)
    ep.press(ep.env.button)
    ep.step()
    ep.unpress(ep.env.button)


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"mi{c['move_interval']}-st{c['stop_time']}")
def test_stop_on_the_nth_visit_succeeds_and_persists(world, diff, case):
    ep = _case_episode(world, diff, case)
    env = ep.env
    st = case["stop_time"]
    goals = goal_text(env)
    assert f"for the {ORDINALS[st]} time" in goals[0] and f"on its {ORDINALS[st]} visit" in goals[1]
    _hover(ep)
    ep.step()
    assert ep.task_index == 1  # 准备子任务完成
    _press_at(ep, case["visits"][st - 1])
    ep.step(3)
    assert ep.success and not ep.fail
    locked = env.stop_timestep
    lo, hi = case["window"]
    assert lo <= locked <= hi
    _run_to(ep, case["deadline"] + case["move_interval"])  # 成功后继续：越过截止步仍成功，停止步不被改写
    assert ep.success and not ep.fail
    assert env.stop_timestep == locked


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"mi{c['move_interval']}-st{c['stop_time']}")
def test_stop_on_previous_visit_fails(world, diff, case):
    ep = _case_episode(world, diff, case)
    _hover(ep)
    ep.step()
    _press_at(ep, case["visits"][case["stop_time"] - 2])
    _run_to(ep, case["deadline"])
    assert any(f for _, f, _ in ep.history)
    assert not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_press_off_target_fails_immediately(world, diff):
    case = CASES[0]
    ep = _case_episode(world, diff, case)
    _hover(ep)
    ep.step()
    _press_at(ep, case["off_target_step"])
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"mi{c['move_interval']}-st{c['stop_time']}")
def test_never_pressing_times_out(world, diff, case):
    ep = _case_episode(world, diff, case)
    _hover(ep)
    _run_to(ep, case["deadline"])
    assert not ep.fail
    ep.step()
    assert ep.fail and not ep.success


def test_without_hover_the_final_subgoal_is_never_reached(world):
    case = CASES[0]
    ep = _case_episode(world, "easy", case)
    _press_at(ep, case["visits"][case["stop_time"] - 1])  # 没先悬停按钮：指针停在准备子任务，正确时刻按也不算
    _run_to(ep, case["deadline"] + 1)
    assert ep.fail and not ep.success


def test_stop_freezes_cube(world):
    case = CASES[0]
    ep = _case_episode(world, "easy", case)
    env = ep.env
    _hover(ep)
    ep.step()
    _press_at(ep, case["visits"][case["stop_time"] - 1])
    frozen = env.cube.xyz.copy()
    ep.step(10)
    assert (env.cube.xyz == frozen).all()
