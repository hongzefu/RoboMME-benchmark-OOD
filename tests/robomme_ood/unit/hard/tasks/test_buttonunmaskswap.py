"""ButtonUnmaskSwap 新值档真值表（xhard1、xhard2）：先后按下两个不同按钮 → 等交换结束 → 抓起仍藏着目标颜色方块的容器。

错误与边界：重复按同一个按钮不推进；同一帧按下两个按钮只推进一项、第二项再也完成不了；
缺第二个按钮不推进；交换结束前抓对的容器不推进；交换后抓干扰容器即失败；重建环境后按钮列表恢复为两个。
"""
from __future__ import annotations

import pytest

from .. import offline_scene as O
from ..world import World, cpu_world
from . import swap_driver as S
from . import unmask_driver as D

TASK = "ButtonUnmaskSwap"
TIERS = O.tiers_of(TASK)
OK = S.OK


@pytest.fixture
def world():
    with cpu_world():
        yield lambda tier: World.build(TASK, tier)


def _tap(w, button):
    w.press(button)
    out = w.step()
    w.unpress(button)
    return out


@pytest.mark.parametrize("tier", TIERS)
def test_two_buttons_then_follow_identity_succeeds(world, tier):
    w = world(tier)
    assert len(w.env.button_list) == 2, "新建环境按钮列表恢复为两个（无跨局残留）"
    assert _tap(w, w.env.button_left) == OK and w.stage == 1
    assert _tap(w, w.env.button_right) == OK and w.stage == 2
    S.run_through_swaps(w)
    assert w.stage == 3, "等待交换结束项完成"
    bins = S.bins_in_order(w)
    assert [b.name for b in bins] == [w.env.selected_bins[i].name for i in range(w.env.pick_times)]
    assert S.pick_sequence(w, bins) == {"success": True, "fail": False}


@pytest.mark.parametrize("tier", TIERS[:1])
def test_both_buttons_in_one_frame_strands_second_item(world, tier):
    w = world(tier)
    w.press(w.env.button_left)
    w.press(w.env.button_right)
    w.step()
    w.unpress(w.env.button_left)
    w.unpress(w.env.button_right)
    assert w.stage == 1 and w.env.button_list == []
    for _ in range(3):
        _tap(w, w.env.button_right)
    assert w.stage == 1, "两个按钮已在同一帧被移出列表，第二项无法再完成"


@pytest.mark.parametrize("tier", TIERS[:1])
def test_repeat_button_missing_button_early_pick_then_distractor(world, tier):
    """一局里依次核：重复按同一个按钮不推进；缺第二个按钮时抓对的容器不推进；按齐两个按钮、
    交换走完后抬干扰容器即失败（Swap 建场较重，几条负例共用一局）。"""
    w = world(tier)
    _tap(w, w.env.button_left)
    assert _tap(w, w.env.button_left) == OK and w.stage == 1
    right = w.env.selected_bins[0]
    D.lift(w, right)
    for _ in range(2):
        assert w.step() == OK
    assert w.stage == 1
    D.put_down(w, right)
    assert _tap(w, w.env.button_right) == OK and w.stage == 2
    S.run_through_swaps(w)
    D.lift(w, w.env.distractor_bins[0])
    assert w.tick() == {"success": False, "fail": True}
