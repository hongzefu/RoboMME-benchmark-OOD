"""V6 交换对象均匀化（NEWTASK_RELEASE_V6_PLAN 2.2 / 2.4 / 2.5，M6(a) S5；外环 O4）的纯函数。

只在 VideoUnmaskSwap / ButtonUnmaskSwap / VideoRepick 的 xhard 分支被调用，原三档从不经过这里。

术语：
* **槽**：交换结束时两个对象精确对换位姿，所以一局里对象占据的位姿集合（槽）不变，只有「谁占哪个槽」在置换；
  两对象交换的扫掠几何只取决于二者所占的**无序槽对**（谁当发起者路径相同）。
* **可行槽对图 G**：对每个无序槽对跑一次仓库精确扫掠判定得到的邻接矩阵（调用方提供判定）。
* **S5（计数均衡贪心 + 整条重排）**：每次交换的候选 = G 的边 − 上一次用过的槽对（禁止立即撤销：同一槽对再换一次
  就是把刚换过去的两个对象换回来）；在候选里取「两个占用者已参与次数」的评分最小者，平局均匀抽，再用一次随机数决定
  谁当发起者；整条序列参与次数极差 > ``accept_range`` 时换随机数重排，最多 ``budget`` 次，仍不满足取极差最小的一条。
* 随机性全部走调用方给的**局部** ``torch.Generator``（由主流追加抽取的一个规划种子播种），抽取次数不固定也不影响主流。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

#: S5 规则名（写进 decision.xhard 的申报值，环境按名字分派）。
S5_RULE = "s5_balanced_greedy"
#: 评分口径：``max_sum`` = 先比二者参与次数的较大者、再比和（Unmask 探针 p2c/p2d 口径）；
#: ``sum_max`` = 先比和、再比较大者（VideoRepick 探针 schemes.s3_balanced 口径）。
SCORES = ("max_sum", "sum_max")


def inner_swap_plan_cfg(*, require_connected: bool, score: str) -> dict:
    """``decision.<tier>.swap_plan_v6`` 的 S5 申报值。"""
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
    """校验 S5 申报值；规则名与评分口径只认上面定下的几种。"""
    if cfg.get("rule") != S5_RULE:
        raise ValueError(f"swap_plan_v6.rule 只支持 {S5_RULE!r}，收到 {cfg.get('rule')!r}")
    if cfg.get("score") not in SCORES:
        raise ValueError(f"swap_plan_v6.score 只支持 {SCORES}，收到 {cfg.get('score')!r}")
    if cfg.get("forbid_immediate_undo") is not True:
        raise ValueError("swap_plan_v6.forbid_immediate_undo 必须为 True")
    budget = cfg.get("range_retry_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f"swap_plan_v6.range_retry_budget 必须是正整数，收到 {budget!r}")
    accept = cfg.get("accept_range")
    if isinstance(accept, bool) or not isinstance(accept, int) or accept < 0:
        raise ValueError(f"swap_plan_v6.accept_range 必须是非负整数，收到 {accept!r}")
    return cfg


# ── 可行槽对图 ─────────────────────────────────────────────────────────────────
def slot_pair_graph(n: int, feasible: Callable[[int, int], bool]) -> list[list[bool]]:
    """对全部 C(n,2) 个无序槽对各调一次 ``feasible(a, b)``（a<b），返回对称邻接矩阵（对角为 False）。"""
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
    """G 是否连通（全部槽经可行交换互达）；n ≤ 1 视为连通。"""
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
    """S5 规划结果。``pairs[k] = (发起者对象, 搭档对象)``，``slot_pairs[k]`` 为该次交换前二者所占的 (槽, 槽)。"""

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
    occ = list(range(n))       # occ[槽] = 对象
    slot_of = list(range(n))   # slot_of[对象] = 槽
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
    """S5：在可行槽对图 ``adj`` 上规划 ``n_swaps`` 次交换；无法规划（没有边、或禁止撤销后无候选）返回 ``None``。

    整条极差 ≤ ``accept_range`` 即采用；否则换随机数重排，最多 ``budget`` 条，全不满足取极差最小的第一条。
    """
    if score not in SCORES:
        raise ValueError(f"score 只支持 {SCORES}，收到 {score!r}")
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
    """独立复核一条（可能是冻结的）交换序列：每次两个对象所占槽对在 G 里可行、没有立即撤销。

    返回 ``(违反项, 统计)``；统计里有参与次数、极差与撤销数（不管是否违反都算）。
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
            problems.append(f"第 {k} 次交换 ({a},{b}) 越界或重复")
            continue
        sa, sb = slot_of[a], slot_of[b]
        if not adj[sa][sb]:
            problems.append(f"第 {k} 次交换 ({a},{b}) 所占槽对 ({sa},{sb}) 在可行图里不可行")
        key = (min(sa, sb), max(sa, sb))
        if key == last:
            stats.undo += 1
            if forbid_undo:
                problems.append(f"第 {k} 次交换 ({a},{b}) 立即撤销了上一次")
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
    """对象口径的立即撤销数：相邻两次交换是同一对对象（无序）。"""
    undo = 0
    last = None
    for a, b in pairs:
        key = (min(int(a), int(b)), max(int(a), int(b)))
        undo += int(key == last)
        last = key
    return undo


# ── 外环 O4：每窗在可行对里取「两者参与次数 max、再 sum」最小者，禁止立即撤销（除非别无选择），平局均匀 ──────────
def balanced_pair_groups(count: int, cnt: Sequence[int], last: tuple[int, int] | None) -> list[list[tuple[int, int]]]:
    """把全部 C(count,2) 个对象对按 O4 评分升序分组；``last`` 放到最末单独一组（只在别无选择时才用）。"""
    pairs = [(a, b) for a in range(count) for b in range(a + 1, count) if (a, b) != last]
    keyed: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in pairs:
        keyed.setdefault(_score(int(cnt[a]), int(cnt[b]), "max_sum"), []).append((a, b))
    groups = [keyed[k] for k in sorted(keyed)]
    if last is not None:
        groups.append([last])
    return groups
