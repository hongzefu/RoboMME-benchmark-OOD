"""Shared primitives for the VideoUnmask/ButtonUnmask/two Swap task truth tables: pick and place by "the container hiding the cube of the k-th color".

The hiding relation is judged independently by top-view geometry (cube center coincides with container center), without reading production's ``bin_k`` naming convention.
"""
from __future__ import annotations

import numpy as np

LIFT_BIN_Z = 0.2  # container lift height: above the lift criterion of is_bin_pickup


def bin_hiding(w, cube, bins):
    c = w.xyz(cube)[:2]
    hits = [b for b in bins if np.linalg.norm(w.xyz(b)[:2] - c) <= 1e-4]
    assert len(hits) == 1, cube.name
    return hits[0]


def lift(w, b):
    x, y, _ = w.xyz(b)
    w._bin_home = getattr(w, "_bin_home", {})
    w._bin_home.setdefault(b.name, w.xyz(b).copy())
    w.move(b, (x, y, LIFT_BIN_Z))
    w.agent.held = b
    w.tcp_to((x, y, LIFT_BIN_Z))


def put_down(w, b):
    home = getattr(w, "_bin_home", {}).get(b.name, w.xyz(b))
    w.move(b, (home[0], home[1], home[2]))
    if w.agent.held is b:
        w.agent.held = None
    w.tcp_to((home[0], home[1], 0.25))


def colour_cubes(env):
    """Gives the hidden cube of each color in production ``color_names`` order (the k-th pick in the task table corresponds to the k-th color)."""
    return [getattr(env, f"target_cube_{name}") for name in env.color_names]
