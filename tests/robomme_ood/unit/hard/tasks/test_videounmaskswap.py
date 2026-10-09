"""VideoUnmaskSwap 新值档真值表（xhard1、xhard2）：容器经多次交换后，抓起仍藏着目标颜色方块的那个容器（跟随身份）。

经真实 ``step`` 走完揭示与全部交换窗口（方块随容器移动由生产的停放／落回逻辑完成），再按交换后的几何藏物关系抓放：
正例成功；抓「目标方块交换前所在位置」上现在的容器（若已换成别的容器）即失败；抬干扰容器即失败；
交换没走完（静止项未完成）之前抓对的容器不推进。
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import swap_driver as S
from . import unmask_driver as D

TASK = "VideoUnmaskSwap"
TIERS = O.tiers_of(TASK)


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier, k=0: World.build(TASK, tier, k)


@pytest.mark.parametrize("tier", TIERS)
def test_follow_identity_through_swaps_succeeds(world, tier):
    w = world(tier)
    S.run_through_swaps(w)
    assert w.stage == 1, "交换全部走完后静止项完成"
    bins = S.bins_in_order(w)
    assert [b.name for b in bins] == [w.env.selected_bins[i].name for i in range(w.env.pick_times)]
    assert S.pick_sequence(w, bins) == {"success": True, "fail": False}


#: 选格（T13 离线探针，xhard1／xhard2 各前 8 个正式局）：「目标方块交换前位置上现在是另一个容器」
#: 在 xhard1 第 0、1、2、4、7 局成立，xhard2 第 1、3、4、5、7 局成立；原先只参数化 xhard1 第 0 局，
#: xhard2 从未覆盖（其第 0 局不成立）。按档钉能触发的局，触发条件写成前置断言：包内规格若变动使条件
#: 不再成立，用例响亮失败而不是静默 skip。
OLD_POS_K = {"xhard1": 0, "xhard2": 1}


@pytest.mark.parametrize("tier", TIERS)
def test_original_position_is_a_trap(world, tier):
    w = world(tier, OLD_POS_K[tier])
    origin = S.run_through_swaps(w)
    target = D.colour_cubes(w.env)[0]
    right = S.bins_in_order(w)[0]
    impostor = S.bin_at(w, origin[target.name], w.env.spawned_bins)
    assert impostor is not None and impostor is not right, \
        "选格失效：目标方块交换前的位置上现在没有别的容器，须重选 OLD_POS_K"
    D.lift(w, impostor)
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_distractor_container_after_swaps_fails(world, tier):
    w = world(tier)
    S.run_through_swaps(w)
    D.lift(w, w.env.distractor_bins[0])
    assert w.tick() == {"success": False, "fail": True}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_picking_before_swaps_finish_does_not_advance(world, tier):
    w = world(tier)
    right = w.env.selected_bins[0]
    D.lift(w, right)
    for _ in range(5):
        assert w.tick() == S.OK
    assert w.stage == 0
