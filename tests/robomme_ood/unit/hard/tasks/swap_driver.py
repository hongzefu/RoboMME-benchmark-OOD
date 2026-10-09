"""两个 Swap 任务真值表的公共流程：经真实 ``step`` 走完揭示与全部交换窗口，再按藏物关系抓放。"""
from __future__ import annotations

import numpy as np

from . import unmask_driver as D

OK = {"success": False, "fail": False}


def run_through_swaps(w, before_swaps=None):
    """经任务类真实 ``step`` 推进到最后一个交换窗口结束之后；``before_swaps(w)`` 在第一个交换窗口前调用一次。

    返回：每个被藏方块在交换前的俯视位置（交换前最后一步读取）。
    """
    env = w.env
    w.still()
    start = env.swap_window_start
    last_end = env.swap_schedule[-1][3]
    origin = {}
    while int(env.elapsed_steps) < last_end + 2:
        if int(env.elapsed_steps) == start - 1:
            origin = {c.name: w.xyz(c)[:2].copy() for c, _ in env.cube_bin_pairs}
            if before_swaps is not None:
                before_swaps(w)
        out = w.step()
        assert out["fail"] is False, f"第 {int(env.elapsed_steps)} 步提前失败"
    return origin


def bins_in_order(w):
    """第 k 次抓取的容器：藏着第 k 种颜色方块的那个（交换之后按几何判定）。"""
    cubes = D.colour_cubes(w.env)[: w.env.pick_times]
    return [D.bin_hiding(w, c, w.env.spawned_bins) for c in cubes]


def pick_sequence(w, bins):
    out = None
    for k, b in enumerate(bins):
        D.lift(w, b)
        out = w.tick()
        if k < len(bins) - 1:
            assert out == OK
            D.put_down(w, b)
            assert w.tick() == OK
    return out


def bin_at(w, xy, bins, tol=0.02):
    hits = [b for b in bins if np.linalg.norm(w.xyz(b)[:2] - np.asarray(xy)) <= tol]
    return hits[0] if len(hits) == 1 else None
