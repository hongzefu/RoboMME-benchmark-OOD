"""VideoUnmask／ButtonUnmask／两个 Swap 任务共用的布局断言（手算几何，不调生产判据）。"""
from __future__ import annotations

import itertools

import numpy as np

from . import cells as C
from . import offline_scene as O


def xy(actor) -> np.ndarray:
    return actor.pose.p[0, :2].numpy().astype(np.float64)


def floor_half(actor) -> float:
    """容器底板的半边长：取自实际碰撞盒（build_bin 建出的形状），即容器俯视轮廓的内切半径。"""
    return max(float(s.half_size[0]) for s in actor._fake_shapes if s.kind == "box")


def assert_containers_disjoint(bins) -> None:
    """俯视两两不相交的必要条件：中心距 ≥ 两内切圆半径之和。"""
    for a, b in itertools.combinations(bins, 2):
        assert np.linalg.norm(xy(a) - xy(b)) >= floor_half(a) + floor_half(b) - 1e-6, (a.name, b.name)


def hidden_under(cube, bins, tol=1e-4):
    """方块藏在哪个容器下（俯视中心重合）；返回容器列表。"""
    return [b for b in bins if np.linalg.norm(xy(cube) - xy(b)) <= tol]


def assert_distractors_match_decision(env, dist_cfg) -> None:
    assert len(env.distractor_bins) == dist_cfg["count"]
    lo, hi = dist_cfg["cube_count_range"]
    assert lo <= len(env.distractor_cubes) <= hi
    pool = set(dist_cfg["color_pool"])
    for cube in env.distractor_cubes:
        assert cube.name.rsplit("_", 1)[-1] in pool, cube.name
        # 每个干扰方块恰藏在一个干扰容器下
        assert len(hidden_under(cube, env.distractor_bins)) == 1, cube.name


def check_unmask_layout(task, tier, k):
    """容器数、区域、藏物、干扰物、抓取次数（期望取自包内 header 的 decision）。"""
    _, env = C.replayed(task, tier, k)
    dec = O.delivered_rows(task, tier, 0)[0]["sampling_config"][task]["decision"]
    pol = dec["bin_layout_policy"]
    assert len(env.spawned_bins) == pol["count"][tier]
    c, h = pol["region_center"], pol["region_half_size"]
    for b in env.spawned_bins:
        x, y = xy(b)
        assert abs(x - c[0]) <= h + 1e-9 and abs(y - c[1]) <= h + 1e-9, b.name
    assert env.xhard_pick_count == dec["pick_count"][tier]
    # 三个有色方块各藏在不同的区域内容器下
    cubes = [env.target_cube_0, env.target_cube_1, env.target_cube_2]
    homes = [hidden_under(cube, env.spawned_bins) for cube in cubes]
    assert all(len(hm) == 1 for hm in homes)
    assert len({hm[0].name for hm in homes}) == 3
    assert_distractors_match_decision(env, dec[tier]["distractor"])
    assert_containers_disjoint(list(env.spawned_bins) + list(env.distractor_bins))
