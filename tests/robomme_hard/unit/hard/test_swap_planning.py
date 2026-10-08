"""V9 实际走的交换规划：内环 S5 均衡（``utils/swap_uniform.py``）、内环 reset 预规划的回放复核
（``unmask_swap_xhard.plan_inner_swaps_v6``）、外环 O4 均衡（``balanced_pair_groups`` 与
``plan_distractor_swaps_balanced``）、以及交换时间窗与路径的小工具。

纯函数用手写小图；外环规划用包内正式局离线建场得到的真实内环窗口与干扰布局。
"""
from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import pytest
import torch

from robomme_hard.robomme_env.utils import swap_uniform as SU
from robomme_hard.robomme_env.utils import unmask_swap_xhard as UX
from robomme_hard.robomme_env.utils.bin_collision import LANE_OFFSET
from robomme_hard.robomme_env.utils.episode_spec import EpisodeSpecError
from robomme_hard.robomme_env.utils.unmask_distractor_sampler import DistractorLayout

from . import cells as C
from . import offline_scene as O
from .world import World, cpu_world


def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _complete(n):
    return [[i != j for j in range(n)] for i in range(n)]


def _count(pairs, n):
    c = Counter(x for p in pairs for x in p)
    return [c.get(i, 0) for i in range(n)]


# ── 可行槽对图 ─────────────────────────────────────────────────────────────────


def test_graph_helpers_on_hand_graph():
    adj = SU.slot_pair_graph(4, lambda a, b: (a, b) in {(0, 1), (1, 2)})
    assert SU.graph_edges(adj) == [(0, 1), (1, 2)]
    assert SU.isolated_slots(adj) == [3]
    assert SU.graph_connected(adj) is False
    assert SU.graph_connected(SU.slot_pair_graph(3, lambda a, b: b == a + 1)) is True
    assert SU.graph_connected([[False]]) is True


# ── S5 ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("score", SU.SCORES)
def test_s5_on_complete_graph_is_balanced_and_valid(seed, score):
    adj = _complete(4)
    plan = SU.plan_balanced_swaps(adj, 6, _gen(seed), score=score)
    assert len(plan.pairs) == 6
    counts = _count(plan.pairs, 4)
    assert counts == plan.counts and max(counts) - min(counts) == plan.spread <= 1
    problems, stats = SU.verify_swap_sequence(adj, plan.pairs)
    assert problems == [] and stats.undo == 0


def test_s5_same_seed_same_plan():
    a = SU.plan_balanced_swaps(_complete(5), 7, _gen(9))
    b = SU.plan_balanced_swaps(_complete(5), 7, _gen(9))
    assert a.pairs == b.pairs and a.slot_pairs == b.slot_pairs


def test_s5_single_edge_cannot_avoid_undo():
    adj = SU.slot_pair_graph(3, lambda a, b: (a, b) == (0, 1))
    assert SU.plan_balanced_swaps(adj, 1, _gen(0)).pairs in ([(0, 1)], [(1, 0)])
    assert SU.plan_balanced_swaps(adj, 2, _gen(0)) is None, "禁止立即撤销时第二次无候选"
    assert len(SU.plan_balanced_swaps(adj, 2, _gen(0), forbid_undo=False).pairs) == 2


def test_s5_degenerate_inputs():
    assert SU.plan_balanced_swaps(_complete(3), 0, _gen(0)).pairs == []
    assert SU.plan_balanced_swaps([[False] * 3 for _ in range(3)], 2, _gen(0)) is None
    with pytest.raises(ValueError):
        SU.plan_balanced_swaps(_complete(3), 2, _gen(0), score="nope")


def test_verify_tracks_slots_through_swaps():
    """槽 0-1、1-2 可行，0-2 不可行。对象 0、1 先交换后：对象 0 在槽 1、对象 1 在槽 0。
    此时交换对象 (0,2) 占槽 (1,2) 可行；交换对象 (1,2) 占槽 (0,2) 不可行。"""
    adj = SU.slot_pair_graph(3, lambda a, b: (a, b) in {(0, 1), (1, 2)})
    ok, _ = SU.verify_swap_sequence(adj, [(0, 1), (0, 2)])
    assert ok == []
    bad, _ = SU.verify_swap_sequence(adj, [(0, 1), (1, 2)])
    assert len(bad) == 1 and "不可行" in bad[0]
    undo, stats = SU.verify_swap_sequence(adj, [(0, 1), (1, 0)])
    assert stats.undo == 1 and any("撤销" in p for p in undo)
    oob, _ = SU.verify_swap_sequence(adj, [(0, 3), (1, 1)])
    assert len(oob) == 2


def test_participation_and_object_undo_count():
    pairs = [(0, 1), (1, 0), (2, 3), (3, 2), (0, 2)]
    assert SU.participation(pairs, 4) == [3, 2, 3, 2]
    assert SU.count_undo(pairs) == 2


def test_parse_inner_cfg_accepts_packaged_and_rejects_variants():
    header, _ = O.delivered_rows("ButtonUnmaskSwap", "xhard1", 0)
    cfg = header["sampling_config"]["ButtonUnmaskSwap"]["decision"]["xhard1"]["swap_plan_v6"]
    assert SU.parse_inner_swap_plan_cfg(copy.deepcopy(cfg)) == cfg
    for key, bad in [("rule", "x"), ("score", "x"), ("forbid_immediate_undo", False), ("range_retry_budget", 0),
                     ("range_retry_budget", True), ("accept_range", -1)]:
        with pytest.raises(ValueError):
            SU.parse_inner_swap_plan_cfg({**cfg, key: bad})


# ── 内环 reset 预规划：回放冻结序列时的复核 ─────────────────────────────────────


@pytest.mark.parametrize("task", ["VideoUnmaskSwap", "ButtonUnmaskSwap"])
def test_inner_plan_replay_rejects_immediate_undo(task):
    row, env = C.replayed(task, "xhard1", 0)
    pairs = env._newvalue_inner_plan["pairs"]
    assert len(pairs) == env.swap_times
    problems, _ = SU.verify_swap_sequence(env._newvalue_inner_plan["graph"], pairs)
    assert problems == []
    bad = copy.deepcopy(row["spec"])
    first = bad["actions"]["swap_pairs"]["0"]
    bad["actions"]["swap_pairs"]["1"] = {"initiator": first["partner"], "partner": first["initiator"]}
    with pytest.raises(EpisodeSpecError):
        with cpu_world():
            World.build(task, "xhard1", 0, spec=bad)


# ── 外环 O4 ──────────────────────────────────────────────────────────────────


def test_balanced_pair_groups_orders_by_max_then_sum_and_defers_last():
    groups = SU.balanced_pair_groups(3, [2, 0, 0], last=(1, 2))
    assert groups == [[(0, 1), (0, 2)], [(1, 2)]]
    groups = SU.balanced_pair_groups(4, [1, 1, 0, 0], last=None)
    assert groups[0] == [(2, 3)] and groups[1] == [(0, 2), (0, 3), (1, 2), (1, 3)] and groups[2] == [(0, 1)]


@pytest.mark.parametrize("task", ["VideoUnmaskSwap", "ButtonUnmaskSwap"])
def test_outer_plan_on_real_windows(task):
    row, env = C.replayed(task, "xhard1", 0)
    dec = O.delivered_rows(task, "xhard1", 0)[0]["sampling_config"][task]["decision"]["xhard1"]
    scfg = UX.parse_distractor_swap_cfg(dec["distractor_swap"])
    windows = UX.planned_inner_windows(env)
    plan = UX.plan_distractor_swaps_balanced(env.distractor_layout, windows, _gen(0), cfg=scfg,
                                             cube_half_size=env.cube_half_size)
    assert plan.ok and len(plan.pairs) == len(windows) == env.swap_times
    n = env.distractor_layout.count
    for k, (o, p) in enumerate(plan.pairs):
        assert 0 <= o < p < n
        if k and plan.fallbacks[k] == 0 and n > 2:
            assert (o, p) != plan.pairs[k - 1], "评分最小组里不会出现上一窗的同一对"
    assert len(plan.final_xy) == n


def test_parse_outer_cfg_lane_must_match_collision_model():
    header, _ = O.delivered_rows("VideoUnmaskSwap", "xhard1", 0)
    cfg = copy.deepcopy(header["sampling_config"]["VideoUnmaskSwap"]["decision"]["xhard1"]["distractor_swap"])
    assert UX.parse_distractor_swap_cfg(cfg).lane_offset == LANE_OFFSET
    with pytest.raises(ValueError):
        UX.parse_distractor_swap_cfg({**cfg, "lane_offset": LANE_OFFSET + 0.01})
    with pytest.raises(ValueError):
        UX.parse_distractor_swap_cfg({**cfg, "smooth": False})


# ── 时间窗与路径小工具 ─────────────────────────────────────────────────────────


def test_scaled_window_steps():
    assert UX.scaled_window_steps(50, 1) == 50
    assert UX.scaled_window_steps(50, 1.5) == 33
    assert UX.scaled_window_steps(50, 0.4) == 125
    for bad in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            UX.scaled_window_steps(50, bad)
    with pytest.raises(ValueError):
        UX.scaled_window_steps(1, 3.0)


@pytest.mark.parametrize("sel,ok", [([2, 0, 1], True), ([0, 1, 3], False), ([0, 0, 1], False),
                                    ([True, 0, 1], False), ([0.0, 1, 2], False), ([0, 1], False)])
def test_hidden_bin_selection(sel, ok):
    if ok:
        assert UX.validate_hidden_bin_selection(sel) == sel
    else:
        with pytest.raises(EpisodeSpecError):
            UX.validate_hidden_bin_selection(sel)


def test_relabel_distractor_layout():
    lay = DistractorLayout(bins=[(0.0, 0.0, 0.0), (1.0, 0.0, 10.0), (2.0, 0.0, 20.0)], cube_count=1,
                           cube_bins=[2], color_order=[1, 0, 2], trials=[1, 2, 3])
    new = UX.relabel_distractor_layout(lay, [2, 0, 1])
    assert new.bins == [(2.0, 0.0, 20.0), (0.0, 0.0, 0.0), (1.0, 0.0, 10.0)]
    assert new.cube_bins == [0] and new.trials == [3, 1, 2]
    with pytest.raises(ValueError):
        UX.relabel_distractor_layout(lay, [0, 0, 1])


def test_nearest_distractor_excludes_self_and_breaks_ties_low():
    pos = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (3.0, 0.0)]
    assert UX.nearest_distractor(pos, 0) == 1
    assert UX.nearest_distractor([(0.0, 0.0)], 0) is None


def test_lane_center_paths_endpoints_and_midpoint_offset():
    s = UX.path_samples(3)
    pa, pb = UX.lane_center_paths((0.0, 0.0), (0.3, 0.0), s)
    assert np.allclose(pa[0], [0, 0]) and np.allclose(pa[-1], [0.3, 0]) and np.allclose(pb[-1], [0, 0])
    assert np.allclose(pa[1], [0.15, LANE_OFFSET]) and np.allclose(pb[1], [0.15, -LANE_OFFSET])
