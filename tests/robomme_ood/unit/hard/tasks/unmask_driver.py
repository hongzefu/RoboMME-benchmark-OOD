"""VideoUnmask／ButtonUnmask／两个 Swap 任务真值表的公共原语：按「藏着第 k 种颜色方块的容器」抓放。

藏物关系用俯视几何独立判定（方块中心与容器中心重合），不读生产的 ``bin_k`` 命名约定。
"""
from __future__ import annotations

import numpy as np

LIFT_BIN_Z = 0.2  # 容器抬起高度：高于 is_bin_pickup 的抬起判据


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
    """按生产 ``color_names`` 顺序给出每种颜色的被藏方块（任务表的第 k 次抓取对应第 k 种颜色）。"""
    return [getattr(env, f"target_cube_{name}") for name in env.color_names]
