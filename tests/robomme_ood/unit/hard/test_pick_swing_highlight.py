"""PickXtimes／SwingXtimes／PickHighlight 新值档：离线真实 ``_load_scene`` 的布局、包内规格回放与自导出。

参数化维度 = 各任务 V9 实际交付的档（``V9_CELLS``）。期望值来自包内 header 内嵌的 ``sampling_config``（该档的
decision 子树）与手算几何（两两中心距、区域边界），不复刻生产的取值逻辑。
"""
from __future__ import annotations

import colorsys
import itertools

import numpy as np
import pytest

from . import cells as C
from . import offline_scene as O
from .world import World, cpu_world

TASKS = ("PickXtimes", "SwingXtimes", "PickHighlight")


def _decision(task, tier):
    header, _ = O.delivered_rows(task, tier, 0)
    return header["sampling_config"][task]["decision"]


def _xy(actor):
    return actor.pose.p[0, :2].numpy().astype(np.float64)


def _min_pair(actors):
    return min(float(np.linalg.norm(_xy(a) - _xy(b))) for a, b in itertools.combinations(actors, 2))


def _inside(actor, center, half):
    half = half if isinstance(half, (list, tuple)) else (half, half)
    x, y = _xy(actor)
    return abs(x - center[0]) <= half[0] + 1e-9 and abs(y - center[1]) <= half[1] + 1e-9


# ── 共有：回放、自导出、篡改负例 ─────────────────────────────────────────────


@pytest.mark.parametrize("task,tier,k", C.replay_cases(*TASKS))
def test_packaged_spec_replays_with_zero_mismatch(task, tier, k):
    C.check_packaged_replay(task, tier, k)


@pytest.mark.parametrize("task,tier,k", C.replay_cases(*TASKS))
def test_offline_export_equals_package_and_replays(task, tier, k):
    C.check_self_export(task, tier, k)


@pytest.mark.parametrize("task,tier", O.cells_of(*TASKS))
def test_tampered_spec_is_detected(task, tier):
    C.check_tamper_detected(task, tier)


# ── PickXtimes／SwingXtimes：方块数、干扰块、间距、区域 ─────────────────────────


@pytest.mark.parametrize("task,tier", O.cells_of("PickXtimes", "SwingXtimes"))
def test_cube_counts_spacing_and_regions(task, tier):
    _, env = C.replayed(task, tier, 0)
    dec = _decision(task, tier)
    sub = dec[tier]
    colors = sub["distractor"]["colors"]
    # 干扰块：颜色与个数等于本档 decision；三块有色候选数等于 decision.color[tier]
    assert sorted(a.name for a in env.distractor_cubes) == sorted(f"cube_{c}_0" for c in colors)
    assert len(env.target_candidates) == dec["color"][tier]
    assert set(env.all_cubes) == set(env.target_candidates) | set(env.distractor_cubes)
    assert len(env.all_cubes) == len(env.target_candidates) + len(colors)
    # 目标唯一且是有色候选之一；判失败集合 = 除目标外的全部方块（含全部干扰块）
    assert env.target_cube in env.target_candidates
    assert set(env.non_target_cubes) == set(env.all_cubes) - {env.target_cube}
    assert set(env.distractor_cubes) <= set(env.non_target_cubes)
    lo, hi = dec["number_range"][tier]
    assert lo <= env.num_repeats <= hi
    # 手算：两两中心距不小于本档 min_center_dist_m
    assert _min_pair(env.all_cubes) >= sub["min_center_dist_m"] - 1e-6
    reg = sub["distractor"]
    assert all(_inside(a, reg["region_center"], reg["region_half_size"]) for a in env.distractor_cubes)
    if task == "PickXtimes":
        tc = sub["target_cube_position_policy"]
        assert all(_inside(a, tc["region_center"], tc["region_half_size"]) for a in env.target_candidates)
        goal = sub["goal_position_policy"]
        assert _inside(env.target, goal["region_center"], goal["region_half_size"])
    else:
        assert {env.target_left.name, env.target_right.name} == {"temp_target_0", "temp_target_1"}


@pytest.mark.parametrize("task,tier", O.cells_of("PickXtimes", "SwingXtimes"))
def test_grasping_any_non_target_cube_fails(task, tier):
    """判失败集合：执行段第一步抓起任一非目标方块（含干扰块）即 fail；抓目标不 fail。"""
    with cpu_world():
        names = [a.name for a in World.build(task, tier).env.non_target_cubes]
        for name in names:
            w = World.build(task, tier)
            w.grasp(next(a for a in w.env.all_cubes if a.name == name))
            out = w.evaluate()
            assert out == {"success": False, "fail": True}, name
        w = World.build(task, tier)
        w.grasp(w.env.target_cube)
        assert w.evaluate()["fail"] is False


# ── PickHighlight：方块数、高亮数、颜色下限、区域 ─────────────────────────────


@pytest.mark.parametrize("tier", O.tiers_of("PickHighlight"))
def test_pickhighlight_counts_colors_and_region(tier):
    row, env = C.replayed("PickHighlight", tier, 0)
    dec = _decision("PickHighlight", tier)
    lo, hi = dec["spawn_count"][tier]
    assert lo <= len(env.all_cubes) <= hi
    hlo, hhi = dec["highlight_count"][tier]
    assert hlo <= len(env.target_cubes) <= hhi
    assert len(set(env.target_cubes)) == len(env.target_cubes)
    assert set(env.target_cubes) <= set(env.all_cubes)
    reg = dec["cube_region"]
    assert all(_inside(a, reg["region_center"], reg["region_half_size"]) for a in env.all_cubes)
    # 方块不相交的必要条件：中心距 ≥ 2 × 半边长
    assert _min_pair(env.all_cubes) >= 2 * env.cube_half_size - 1e-6
    # 颜色：本档 HSV 下限（饱和度、明度）对每块颜色成立——用标准库 colorsys 独立换算
    hsv = dec[tier]["block_color_hsv"]
    for rgba in row["spec"]["objects"]["color_rgba"].values():
        _, s, v = colorsys.rgb_to_hsv(*rgba[:3])
        assert s >= hsv["s_range"][0] - 1e-6 and v >= hsv["v_range"][0] - 1e-6


@pytest.mark.parametrize("tier", O.tiers_of("PickHighlight"))
def test_pickhighlight_grasping_unhighlighted_cube_fails(tier):
    """判失败集合：按下按钮之后，抓起任一未高亮方块即 fail（任务表第 2 项起的 failure_func）。"""
    with cpu_world():
        w = World.build("PickHighlight", tier)
        others = [a for a in w.env.all_cubes if a not in w.env.target_cubes]
        assert others, "本档必须有未高亮方块"
        for victim in others:
            w = World.build("PickHighlight", tier)
            w.press(w.env.button)
            assert w.evaluate() == {"success": False, "fail": False}
            w.unpress(w.env.button)
            w.grasp(next(a for a in w.env.all_cubes if a.name == victim.name))
            assert w.evaluate()["fail"] is True, victim.name
