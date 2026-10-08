"""VideoRepick 新值档（xhard1、xhard2）：同色方块数与间距、交换次数与均衡、重复次数，包内规格回放与自导出。"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pytest

from . import cells as C
from . import offline_scene as O

TASK = "VideoRepick"


def _decision(tier):
    header, _ = O.delivered_rows(TASK, tier, 0)
    return header["sampling_config"][TASK]["decision"]


def _xy(actor):
    return actor.pose.p[0, :2].numpy().astype(np.float64)


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK))
def test_packaged_spec_replays_with_zero_mismatch(task, tier, k):
    C.check_packaged_replay(task, tier, k)


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK))
def test_offline_export_equals_package_and_replays(task, tier, k):
    C.check_self_export(task, tier, k)


@pytest.mark.parametrize("tier", O.tiers_of(TASK))
def test_tampered_spec_is_detected(tier):
    C.check_tamper_detected(TASK, tier)


@pytest.mark.parametrize("tier", O.tiers_of(TASK))
@pytest.mark.parametrize("k", range(C.REPLAY_ROWS))
def test_cubes_swaps_and_repeats(tier, k):
    row, env = C.replayed(TASK, tier, k)
    dec = _decision(tier)
    lay = dec[tier]["layout"]
    cubes = env.spawned_cubes
    assert len(cubes) == lay["cube_count"]
    # 手算：两两中心距 ≥ 本档下限；全部落在区域内
    assert min(np.linalg.norm(_xy(a) - _xy(b)) for a, b in itertools.combinations(cubes, 2)) \
        >= lay["min_center_dist_m"] - 1e-6
    c, h = lay["region_center"], lay["region_half_size"]
    for a in cubes:
        x, y = _xy(a)
        assert abs(x - c[0]) <= h[0] + 1e-9 and abs(y - c[1]) <= h[1] + 1e-9
    rep = dec["num_repeats_range"][tier]
    assert rep["low"] <= env.num_repeats < rep["high_exclusive"]
    sw = dec["swap"][tier]
    assert sw["swap_min"] <= env.swap_times <= sw["swap_max"]
    # 交换计划：每次是两个不同容器；不立刻撤销上一次；参与次数按交换对重数后与记录一致，且极差 ≤ 记录的 range
    pairs = [(p["initiator"], p["partner"]) for _, p in sorted(row["spec"]["actions"]["swap_pairs"].items(),
                                                             key=lambda kv: int(kv[0]))]
    assert len(pairs) == env.swap_times
    assert all(a != b for a, b in pairs)
    s5 = dec[tier]["swap_plan"]["s5"]
    if s5["forbid_immediate_undo"]:
        assert all({a, b} != {c2, d} for (a, b), (c2, d) in zip(pairs, pairs[1:]))
    counts = Counter(x for pair in pairs for x in pair)
    plan = row["spec"]["objects"]["swap_plan"]
    assert [counts.get(f"bin_{i}", 0) for i in range(len(cubes))] == plan["counts"]
    assert max(plan["counts"]) - min(plan["counts"]) == plan["range"] <= s5["accept_range"]
    # 时间窗：首尾相接、等长
    sched = env.swap_schedule
    assert len(sched) == env.swap_times
    assert all(s[3] == t[2] for s, t in zip(sched, sched[1:]))
    assert len({s[3] - s[2] for s in sched}) == 1
