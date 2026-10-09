"""VideoPlaceButton 新值档真值表（xhard1、xhard2）：执行段把被问方块放到「按钮前最后一次／按钮后第一次」放过的那个目标台。

答案由演示段实际发生的放置事件按手写规则推出（before → 按钮前该方块最后落到的目标台；after → 按钮后第一次），
目标台在演示收尾时会交换位置，答案跟随台的身份。错误与边界：before／after 反转、放到答案台交换前所在位置上
现在的台、执行段抓别的方块均为失败。
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import demo_driver as DD
from .swap_driver import bin_at

TASK = "VideoPlaceButton"
TIERS = O.tiers_of(TASK)
OK = {"success": False, "fail": False}


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier, k=0: World.build(TASK, tier, k)


def _answer(w, log):
    seq = DD.target_placements(log, w.env.target_cube, w.env.targets)
    if w.env.target_target_language == "before":
        return [t for t, after in seq if not after][-1], [t for t, after in seq if after][0]
    return [t for t, after in seq if after][0], [t for t, after in seq if not after][-1]


def _place(w, target):
    cube = w.env.target_cube
    w.grasp(cube)
    assert w.step() == OK
    w.release_onto(cube, w.xyz(target)[:2])
    return w.step()


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("k", range(2))
def test_place_onto_answer_target_succeeds(world, tier, k):
    w = world(tier, k)
    log = DD.run_demo(w)
    answer, _ = _answer(w, log)
    assert answer is w.env.target_target
    assert _place(w, answer) == {"success": True, "fail": False}


#: 选格（T12 实测，xhard1／xhard2 各前 8 个正式局）：before／after 反转在第 0 局即可构造（8/8 局都可）；
#: 「答案台交换前位置上现在的台」只在部分局存在（xhard1 第 1、4、6 局，xhard2 第 2、4、5 局），
#: 原先固定取第 0 局时该分支两档都不成立、断言空转。下面按档钉确定能触发的局，并把触发条件写成前置断言：
#: 包内规格若变动使条件不再成立，用例响亮失败而不是静默空转。
SWAP_K = {"xhard1": 0, "xhard2": 0}
OLD_POS_K = {"xhard1": 1, "xhard2": 2}


@pytest.mark.parametrize("tier", TIERS)
def test_before_after_swapped_fails(world, tier):
    w = world(tier, SWAP_K[tier])
    log = DD.run_demo(w)
    answer, mirror = _answer(w, log)
    assert mirror is not answer, "选格失效：本局按钮前后放的是同一个台，before／after 反转无从区分，须重选 SWAP_K"
    assert _place(w, mirror) == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS)
def test_answer_old_position_and_wrong_cube_fail(world, tier):
    w = world(tier, OLD_POS_K[tier])
    pre = {t.name: w.xyz(t)[:2].copy() for t in w.env.targets}
    log = DD.run_demo(w)
    answer, _ = _answer(w, log)
    impostor = bin_at(w, pre[answer.name], w.env.targets)
    assert impostor is not None and impostor is not answer, \
        "选格失效：答案台交换前的位置上现在没有别的台，须重选 OLD_POS_K"
    assert _place(w, impostor) == {"success": False, "fail": True}
    w = World.build(TASK, tier)
    DD.run_demo(w)
    w.grasp(w.env.non_target_cubes[0])
    assert w.step() == {"success": False, "fail": True}
