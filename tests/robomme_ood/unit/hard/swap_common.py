"""VideoUnmaskSwap／ButtonUnmaskSwap 共用的布局断言：内环交换次数与时间窗、抓取次数、干扰容器与外环交换。"""
from __future__ import annotations

from collections import Counter

from . import cells as C
from . import offline_scene as O
from . import unmask_common as U


def check_swap_layout(task, tier, k):
    row, env = C.replayed(task, tier, k)
    dec = O.delivered_rows(task, tier, 0)[0]["sampling_config"][task]["decision"]
    sub = dec[tier]
    lo, hi = dec["swap_count_range"][tier]
    assert lo <= env.swap_times <= hi
    plo, phi = dec["pick_count_range"][tier]
    assert plo <= env.pick_times <= phi
    # 三个有色方块各藏在一个选中的容器下，选中的容器互不相同
    assert len(env.cube_bin_pairs) == 3
    for cube, b in env.cube_bin_pairs:
        assert U.hidden_under(cube, [b]) == [b], cube.name
    assert len({b.name for _, b in env.cube_bin_pairs}) == 3
    assert env.target_bin in env.spawned_bins and env.target_cube in env.spawned_dynamic_cubes
    U.assert_distractors_match_decision(env, sub["distractor"])
    U.assert_containers_disjoint(list(env.spawned_bins) + list(env.distractor_bins))
    # 内环：每次交换两个不同容器，不立刻撤销上一次；时间窗首尾相接、等长、从 swap_window_start 起
    pairs = [tuple(p) for p in row["spec"]["actions"]["predicted_inner_swap_pairs"]]
    assert len(pairs) == env.swap_times
    assert all(a != b for a, b in pairs)
    if sub["swap_plan_v6"]["forbid_immediate_undo"]:
        assert all({a, b} != {c, d} for (a, b), (c, d) in zip(pairs, pairs[1:]))
    sched = env.swap_schedule
    assert len(sched) == env.swap_times and sched[0][2] == env.swap_window_start
    assert all(s[3] == t[2] for s, t in zip(sched, sched[1:]))
    assert {s[3] - s[2] for s in sched} == {env.swap_window_steps}
    # 外环：每次交换两个不同干扰容器，次数与内环相同；参与次数按交换对重数后与记录一致
    outer = [tuple(p) for p in row["spec"]["actions"]["distractor_swap_pairs"]]
    assert len(outer) == env.swap_times
    assert all(a != b and 0 <= a < len(env.distractor_bins) and 0 <= b < len(env.distractor_bins) for a, b in outer)
    bal = row["spec"]["actions"]["distractor_swap_balance"]
    counts = Counter(x for pair in outer for x in pair)
    assert [counts.get(i, 0) for i in range(len(env.distractor_bins))] == bal["counts"]
    assert max(bal["counts"]) - min(bal["counts"]) == bal["range"]
    assert bal["unvisited"] == sum(1 for c in bal["counts"] if c == 0)
