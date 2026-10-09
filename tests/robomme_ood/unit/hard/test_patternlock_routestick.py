"""PatternLock／RouteStick 新值档：离线真实 ``_load_scene`` 的路径长度与合法性、包内规格回放与自导出。

期望：路径长度区间取自包内 header 的 decision；路径合法性用手写判据（5×5 网格的 8 邻接、不重访；
RouteStick 相邻落点同排且隔一根杆）独立核对，不调用生产的路径搜索。
"""
from __future__ import annotations

import pytest

from . import cells as C
from . import offline_scene as O

TASKS = ("PatternLock", "RouteStick")


def _decision(task, tier):
    header, _ = O.delivered_rows(task, tier, 0)
    return header["sampling_config"][task]["decision"]


@pytest.mark.parametrize("task,tier,k", C.replay_cases(*TASKS))
def test_packaged_spec_replays_with_zero_mismatch(task, tier, k):
    C.check_packaged_replay(task, tier, k)


@pytest.mark.parametrize("task,tier,k", C.replay_cases(*TASKS))
def test_offline_export_equals_package_and_replays(task, tier, k):
    C.check_self_export(task, tier, k)


@pytest.mark.parametrize("task,tier", O.cells_of(*TASKS))
def test_tampered_spec_is_detected(task, tier):
    C.check_tamper_detected(task, tier)


@pytest.mark.parametrize("tier", O.tiers_of("PatternLock"))
@pytest.mark.parametrize("k", range(C.REPLAY_ROWS))
def test_patternlock_path_is_king_walk_of_declared_length(tier, k):
    row, env = C.replayed("PatternLock", tier, k)
    dec = _decision("PatternLock", tier)
    n = dec["grid"][tier]
    assert len(env.targets_grid) == n * n
    nodes = row["spec"]["actions"]["path_nodes"]
    lo, hi = dec["path_length_range"][tier]
    assert lo <= len(nodes) <= hi
    assert len(set(nodes)) == len(nodes), "路径重访节点"
    assert all(0 <= v < n * n for v in nodes)
    for a, b in zip(nodes, nodes[1:]):
        # 手写 8 邻接：行列差都 ≤ 1 且不同点
        assert max(abs(a // n - b // n), abs(a % n - b % n)) == 1, (a, b)
    # 环境真正要走的按钮序列就是这条路径
    assert [t.name for t in env.selected_buttons] == [f"target_{v}" for v in nodes]


@pytest.mark.parametrize("tier", O.tiers_of("RouteStick"))
@pytest.mark.parametrize("k", range(C.REPLAY_ROWS))
def test_routestick_walk_alternates_around_sticks(tier, k):
    row, env = C.replayed("RouteStick", tier, k)
    dec = _decision("RouteStick", tier)
    lo, hi = dec[tier]["segment_count_range"]
    seg = row["spec"]["objects"]["L"]
    assert lo <= seg <= hi
    nodes = row["spec"]["actions"]["nodes"]
    assert len(nodes) == seg + 1
    # 9 个落点排成一排，奇数位是杆（4 根），偶数位是可停的落点；每段恰好绕过一根杆到相邻落点
    assert len(env.buttons_grid) == 9 and sorted(env.target_cubes) == [1, 3, 5, 7]
    assert all(v % 2 == 0 for v in nodes)
    assert all(abs(a - b) == 2 for a, b in zip(nodes, nodes[1:]))
    dirs = row["spec"]["actions"]["directions"]
    assert len(dirs) == seg and set(dirs.values()) <= {"clockwise", "counterclockwise"}
    assert list(env.swing_directions) == [dirs[str(i)] for i in range(seg)]
    assert [t.name for t in env.selected_buttons] == [f"target_{v}" for v in nodes]
