"""PatternLock/RouteStick new-value tiers: path length and validity of the real offline ``_load_scene``, packaged spec replay and self-export.

Expectations: the path length range comes from the decision in the packaged header; path validity is checked independently with hand-written criteria (8-neighborhood on a 5×5 grid, no revisits;
for RouteStick, adjacent stops are in the same row with exactly one peg between them), without calling production's path search.
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
    assert len(set(nodes)) == len(nodes), "path revisits a node"
    assert all(0 <= v < n * n for v in nodes)
    for a, b in zip(nodes, nodes[1:]):
        # hand-written 8-neighborhood: row and column differences both ≤ 1 and distinct points
        assert max(abs(a // n - b // n), abs(a % n - b % n)) == 1, (a, b)
    # the button sequence the env actually walks is exactly this path
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
    # 9 stops in a row: odd positions are pegs (4), even positions are stoppable stops; each segment goes around exactly one peg to the adjacent stop
    assert len(env.buttons_grid) == 9 and sorted(env.target_cubes) == [1, 3, 5, 7]
    assert all(v % 2 == 0 for v in nodes)
    assert all(abs(a - b) == 2 for a, b in zip(nodes, nodes[1:]))
    dirs = row["spec"]["actions"]["directions"]
    assert len(dirs) == seg and set(dirs.values()) <= {"clockwise", "counterclockwise"}
    assert list(env.swing_directions) == [dirs[str(i)] for i in range(seg)]
    assert [t.name for t in env.selected_buttons] == [f"target_{v}" for v in nodes]
