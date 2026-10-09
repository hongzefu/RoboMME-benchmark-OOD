"""Pure functions for V6 swap-object uniformization (NEWTASK_RELEASE_V6_PLAN 2.2 / 2.4 / 2.5, M6(a) S5; outer loop O4).

Only called from the xhard branches of VideoUnmaskSwap / ButtonUnmaskSwap / VideoRepick; the original three tiers never get here.

Terminology:
* **slot**: at the end of a swap the two objects exactly exchange poses, so the set of poses occupied by objects in an episode (the slots) is fixed; only "who occupies which slot" is permuted;
  the sweep geometry of a two-object swap depends only on the **unordered slot pair** they occupy (the path is the same whichever is the initiator).
* **feasible slot-pair graph G**: the adjacency matrix obtained by running the repository's exact sweep check once per unordered slot pair (the check is supplied by the caller).
* **S5 (count-balancing greedy + whole-sequence resampling)**: candidates for each swap = edges of G minus the slot pair used last time (immediate undo is forbidden: swapping the same slot pair again
  just swaps back the two objects that were just exchanged); among candidates take the one with the minimum score of "participation counts of the two occupants", break ties uniformly, then use one more random draw to decide
  the initiator; if the participation-count range of the whole sequence > ``accept_range``, resample with new random numbers, at most ``budget`` times; if still unsatisfied, take the sequence with the smallest range.
* All randomness goes through the caller-supplied **local** ``torch.Generator`` (seeded by one planning seed drawn additionally from the main stream); a variable number of draws does not affect the main stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

#: S5 rule name (declared value written into decision.xhard; the env dispatches by name).
S5_RULE = "s5_balanced_greedy"
#: Scoring convention: ``max_sum`` = compare the larger of the two participation counts first, then the sum (Unmask probe p2c/p2d convention);
#: ``sum_max`` = compare the sum first, then the larger one (VideoRepick probe schemes.s3_balanced convention).
SCORES = ("max_sum", "sum_max")


def inner_swap_plan_cfg(*, require_connected: bool, score: str) -> dict:
    """S5 declared value of ``decision.<tier>.swap_plan_v6``."""
    cfg = {
        "rule": S5_RULE,
        "score": score,
        "forbid_immediate_undo": True,
        "range_retry_budget": 20,
        "accept_range": 1,
        "require_connected_graph": bool(require_connected),
        "plan_seed_high_exclusive": 2 ** 62,
    }
    return cfg


def parse_inner_swap_plan_cfg(cfg: dict) -> dict:
    """Validate the S5 declared value; only the rule names and scoring conventions defined above are accepted."""
    if cfg.get("rule") != S5_RULE:
        raise ValueError(f"swap_plan_v6.rule only supports {S5_RULE!r}, got {cfg.get('rule')!r}")
    if cfg.get("score") not in SCORES:
        raise ValueError(f"swap_plan_v6.score only supports {SCORES}, got {cfg.get('score')!r}")
    if cfg.get("forbid_immediate_undo") is not True:
        raise ValueError("swap_plan_v6.forbid_immediate_undo must be True")
    budget = cfg.get("range_retry_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f"swap_plan_v6.range_retry_budget must be a positive integer, got {budget!r}")
    accept = cfg.get("accept_range")
    if isinstance(accept, bool) or not isinstance(accept, int) or accept < 0:
        raise ValueError(f"swap_plan_v6.accept_range must be a non-negative integer, got {accept!r}")
    return cfg


# -- feasible slot-pair graph -----------------------------------------------------------------
def slot_pair_graph(n: int, feasible: Callable[[int, int], bool]) -> list[list[bool]]:
    """Call ``feasible(a, b)`` (a<b) once for each of the C(n,2) unordered slot pairs; return a symmetric adjacency matrix (diagonal False)."""
    adj = [[False] * n for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            ok = bool(feasible(a, b))
            adj[a][b] = adj[b][a] = ok
    return adj


def graph_edges(adj: Sequence[Sequence[bool]]) -> list[tuple[int, int]]:
    n = len(adj)
    return [(a, b) for a in range(n) for b in range(a + 1, n) if adj[a][b]]


def graph_connected(adj: Sequence[Sequence[bool]]) -> bool:
    """Whether G is connected (all slots mutually reachable via feasible swaps); n <= 1 counts as connected."""
    n = len(adj)
    if n <= 1:
        return True
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in range(n):
            if adj[u][v] and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def isolated_slots(adj: Sequence[Sequence[bool]]) -> list[int]:
    return [i for i, row in enumerate(adj) if not any(row)]


# ── S5 ────────────────────────────────────────────────────────────────────────
@dataclass
class BalancedSwapPlan:
    """S5 planning result. ``pairs[k] = (initiator object, partner object)``, ``slot_pairs[k]`` is the (slot, slot) the two occupied before that swap."""

    pairs: list[tuple[int, int]] = field(default_factory=list)
    slot_pairs: list[tuple[int, int]] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    spread: int = 0
    tries: int = 0
    undo: int = 0

    def summary(self) -> dict:
        return {"counts": list(self.counts), "range": int(self.spread), "tries": int(self.tries), "undo": int(self.undo)}


def _score(ca: int, cb: int, score: str) -> tuple[int, int]:
    return (max(ca, cb), ca + cb) if score == "max_sum" else (ca + cb, max(ca, cb))


def _randint(high: int, generator: torch.Generator) -> int:
    return int(torch.randint(0, int(high), (1,), generator=generator).item())


def _greedy_once(edges, n, n_swaps, generator, score, forbid_undo) -> BalancedSwapPlan | None:
    occ = list(range(n))       # occ[slot] = object
    slot_of = list(range(n))   # slot_of[object] = slot
    cnt = [0] * n
    last = None
    plan = BalancedSwapPlan()
    for _k in range(n_swaps):
        cands = [e for e in edges if not (forbid_undo and e == last)]
        if not cands:
            return None
        keys = [_score(cnt[occ[a]], cnt[occ[b]], score) for a, b in cands]
        best = min(keys)
        pool = [e for e, key in zip(cands, keys) if key == best]
        sa, sb = pool[_randint(len(pool), generator)]
        if _randint(2, generator):
            sa, sb = sb, sa
        ia, ib = occ[sa], occ[sb]
        plan.pairs.append((ia, ib))
        plan.slot_pairs.append((sa, sb))
        cnt[ia] += 1
        cnt[ib] += 1
        occ[sa], occ[sb] = ib, ia
        slot_of[ia], slot_of[ib] = sb, sa
        last = (min(sa, sb), max(sa, sb))
    plan.counts = cnt
    plan.spread = max(cnt) - min(cnt) if cnt else 0
    return plan


def plan_balanced_swaps(
    adj: Sequence[Sequence[bool]],
    n_swaps: int,
    generator: torch.Generator,
    *,
    score: str = "max_sum",
    budget: int = 20,
    accept_range: int = 1,
    forbid_undo: bool = True,
) -> BalancedSwapPlan | None:
    """S5: plan ``n_swaps`` swaps on the feasible slot-pair graph ``adj``; return ``None`` if planning is impossible (no edges, or no candidate once undo is forbidden).

    Accept as soon as the whole-sequence range <= ``accept_range``; otherwise resample with new random numbers, at most ``budget`` sequences; if none qualifies, take the first with the smallest range.
    """
    if score not in SCORES:
        raise ValueError(f"score only supports {SCORES}, got {score!r}")
    n = len(adj)
    edges = graph_edges(adj)
    if n_swaps <= 0:
        return BalancedSwapPlan(counts=[0] * n, spread=0, tries=0)
    if not edges:
        return None
    best = None
    for t in range(1, int(budget) + 1):
        plan = _greedy_once(edges, n, int(n_swaps), generator, score, forbid_undo)
        if plan is None:
            return None
        plan.tries = t
        if plan.spread <= accept_range:
            return plan
        if best is None or plan.spread < best.spread:
            best = plan
    best.tries = int(budget)
    return best


def verify_swap_sequence(adj: Sequence[Sequence[bool]], pairs: Sequence[Sequence[int]],
                         *, forbid_undo: bool = True) -> tuple[list[str], BalancedSwapPlan]:
    """Independently re-check a (possibly frozen) swap sequence: each time the slot pair occupied by the two objects is feasible in G, and there is no immediate undo.

    Returns ``(violations, stats)``; stats contain participation counts, range and undo count (computed regardless of violations).
    """
    n = len(adj)
    occ = list(range(n))
    slot_of = list(range(n))
    cnt = [0] * n
    last = None
    problems: list[str] = []
    stats = BalancedSwapPlan()
    for k, pair in enumerate(pairs):
        a, b = (int(v) for v in pair)
        if not (0 <= a < n and 0 <= b < n) or a == b:
            problems.append(f"swap {k} ({a},{b}) out of range or duplicated")
            continue
        sa, sb = slot_of[a], slot_of[b]
        if not adj[sa][sb]:
            problems.append(f"swap {k} ({a},{b}) occupies slot pair ({sa},{sb}), which is infeasible in the feasibility graph")
        key = (min(sa, sb), max(sa, sb))
        if key == last:
            stats.undo += 1
            if forbid_undo:
                problems.append(f"swap {k} ({a},{b}) is an immediate undo of the previous one")
        stats.pairs.append((a, b))
        stats.slot_pairs.append((sa, sb))
        cnt[a] += 1
        cnt[b] += 1
        occ[sa], occ[sb] = b, a
        slot_of[a], slot_of[b] = sb, sa
        last = key
    stats.counts = cnt
    stats.spread = max(cnt) - min(cnt) if cnt else 0
    return problems, stats


def participation(pairs: Sequence[Sequence[int]], n: int) -> list[int]:
    cnt = [0] * n
    for a, b in pairs:
        cnt[int(a)] += 1
        cnt[int(b)] += 1
    return cnt


def count_undo(pairs: Sequence[Sequence[int]]) -> int:
    """Object-level immediate undo count: two consecutive swaps involve the same (unordered) object pair."""
    undo = 0
    last = None
    for a, b in pairs:
        key = (min(int(a), int(b)), max(int(a), int(b)))
        undo += int(key == last)
        last = key
    return undo


# -- outer loop O4: per window, among feasible pairs take the minimum by "max, then sum of the two participation counts"; immediate undo forbidden (unless no other choice); ties broken uniformly ----------
def balanced_pair_groups(count: int, cnt: Sequence[int], last: tuple[int, int] | None) -> list[list[tuple[int, int]]]:
    """Group all C(count,2) object pairs in ascending O4 score; ``last`` goes alone into the final group (used only when there is no other choice)."""
    pairs = [(a, b) for a in range(count) for b in range(a + 1, count) if (a, b) != last]
    keyed: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in pairs:
        keyed.setdefault(_score(int(cnt[a]), int(cnt[b]), "max_sum"), []).append((a, b))
    groups = [keyed[k] for k in sorted(keyed)]
    if last is not None:
        groups.append([last])
    return groups
