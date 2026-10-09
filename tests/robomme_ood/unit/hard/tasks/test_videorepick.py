"""VideoRepick 新值档真值表（xhard1、xhard2）：演示拾放目标 → 多次交换 → 执行段把同一个方块（身份，不是位置）拾放 N 次后按按钮。

经任务类真实 ``step``（交换动画在 step 里）走完演示与交换；N 取自本局 ``num_repeats``。
错误与边界：执行段抓「目标交换前所在位置」上现在的方块即失败；少拾放一次就按按钮即失败（时间窗内）；
演示里那一次拾放不计入执行段次数。
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from .swap_driver import bin_at

TASK = "VideoRepick"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier, k=0: World.build(TASK, tier, k)


def _demo_and_swaps(w):
    """演示段：抓起目标 → 放下 → 静止 → 等全部交换结束 → 复位。返回目标在交换前的位置。"""
    env = w.env
    cube = env.target_cube_1
    w.grasp(cube)
    assert w.step() == OK
    w.release_onto(cube, w.xyz(cube)[:2])
    assert w.step() == OK
    origin = w.xyz(cube)[:2].copy()
    w.still()
    last_end = env.swap_schedule[-1][3]
    while int(env.elapsed_steps) < last_end + 30:
        out = w.step()
        assert out == OK, int(env.elapsed_steps)
    first_exec = next(i for i, t in enumerate(env.task_list) if not t["demonstration"])
    assert w.stage == first_exec, "演示与交换项全部完成"
    return origin


def _cycles(w, n):
    cube = w.env.target_cube_1
    for _ in range(n):
        w.grasp(cube)
        assert w.step() == OK
        w.release_onto(cube, w.xyz(cube)[:2])
        assert w.step() == OK


@pytest.mark.parametrize("tier", TIERS)
def test_repick_same_identity_n_times_succeeds(world, tier):
    w = world(tier)
    _demo_and_swaps(w)
    _cycles(w, w.env.num_repeats)
    w.press(w.env.button_left)
    assert w.step() == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_one_short_then_button_fails(world, tier):
    w = world(tier)
    _demo_and_swaps(w)
    _cycles(w, w.env.num_repeats - 1)
    w.press(w.env.button_left)
    assert w.step()["fail"] is True


#: 选格（T13 离线探针，xhard1／xhard2 各前 8 个正式局）：「目标交换前位置上现在是另一个方块」
#: 在 xhard1 第 0、2、5 局成立，xhard2 第 0、1、2、5、6 局成立；第 0 局两档都成立（原 skip 实际未触发），
#: 仍按档显式钉格并把触发条件写成前置断言，包内规格若变动使条件不再成立则响亮失败而不是静默 skip。
OLD_POS_K = {"xhard1": 0, "xhard2": 0}


@pytest.mark.parametrize("tier", TIERS)
def test_original_position_cube_is_a_trap(world, tier):
    w = world(tier, OLD_POS_K[tier])
    origin = _demo_and_swaps(w)
    impostor = bin_at(w, origin, w.env.spawned_cubes)
    assert impostor is not None and impostor is not w.env.target_cube_1, \
        "选格失效：目标交换前的位置上现在没有别的方块，须重选 OLD_POS_K"
    w.grasp(impostor)
    assert w.step() == {"success": False, "fail": True}
