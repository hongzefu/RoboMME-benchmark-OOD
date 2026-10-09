"""Shared flow for the two Swap task truth tables: go through the reveal and all swap windows via the real ``step``, then pick and place by the hiding relation."""
from __future__ import annotations

import numpy as np

from . import unmask_driver as D

OK = {"success": False, "fail": False}


def run_through_swaps(w, before_swaps=None):
    """Advance through the task class's real ``step`` until after the last swap window ends; ``before_swaps(w)`` is called once before the first swap window.

    Returns: the top-view position of each hidden cube before the swaps (read at the last step before swapping).
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
        assert out["fail"] is False, f"step {int(env.elapsed_steps)} failed early"
    return origin


def bins_in_order(w):
    """Container for the k-th pick: the one hiding the cube of the k-th color (judged geometrically after the swaps)."""
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
