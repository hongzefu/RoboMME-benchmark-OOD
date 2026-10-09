"""RouteStick stop random walk (``utils/route.generate_dynamic_walk``): topology, backtracking rule, config guards and determinism.

Expectations are expressed as properties (each step only moves to an adjacent node; with backtracking forbidden, no immediate reversal except at endpoints; same seed same sequence), without replicating the sampling process.
"""
from __future__ import annotations

import pytest
import torch

from robomme_ood.robomme_env.utils.route import generate_dynamic_walk

NODES = [0, 2, 4, 6, 8]
WALK = {"node_indices": NODES, "start_selection": "randint", "neighbor_order": [-1, 1],
        "force_reverse_at_endpoint": True}


def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("backtrack", [True, False])
def test_each_step_moves_to_an_adjacent_node(seed, backtrack):
    path = generate_dynamic_walk(NODES, steps=30, allow_backtracking=backtrack, generator=_gen(seed), walk_config=WALK)
    assert len(path) == 31 and set(path) <= set(NODES)
    idx = [NODES.index(v) for v in path]
    assert all(abs(a - b) == 1 for a, b in zip(idx, idx[1:]))


@pytest.mark.parametrize("seed", range(6))
def test_no_backtracking_reverses_only_at_endpoints(seed):
    path = generate_dynamic_walk(NODES, steps=40, allow_backtracking=False, generator=_gen(seed), walk_config=WALK)
    idx = [NODES.index(v) for v in path]
    for a, b, c in zip(idx, idx[1:], idx[2:]):
        if a == c:  # immediate reversal
            assert b in (0, len(NODES) - 1), (a, b, c)


def test_same_seed_same_walk_and_explicit_start():
    a = generate_dynamic_walk(NODES, steps=20, generator=_gen(3), walk_config=WALK)
    b = generate_dynamic_walk(NODES, steps=20, generator=_gen(3), walk_config=WALK)
    assert a == b
    c = generate_dynamic_walk(NODES, steps=5, start_idx=4, generator=_gen(3), walk_config=WALK)
    assert c[0] == NODES[4] and c[1] == NODES[3], "starting from the end, the first step can only go back"


@pytest.mark.parametrize("bad", [
    {**WALK, "node_indices": [0, 2, 4]},
    {**WALK, "start_selection": "fixed"},
    {**WALK, "neighbor_order": [1, -1]},
    {**WALK, "force_reverse_at_endpoint": False},
])
def test_walk_config_guard(bad):
    with pytest.raises(ValueError):
        generate_dynamic_walk(NODES, steps=3, generator=_gen(0), walk_config=bad)
