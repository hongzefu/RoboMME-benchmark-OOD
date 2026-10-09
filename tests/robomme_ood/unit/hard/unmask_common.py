"""Layout assertions shared by VideoUnmask/ButtonUnmask/the two Swap tasks (hand-computed geometry, production criteria not called)."""
from __future__ import annotations

import itertools

import numpy as np

from . import cells as C
from . import offline_scene as O


def xy(actor) -> np.ndarray:
    return actor.pose.p[0, :2].numpy().astype(np.float64)


def floor_half(actor) -> float:
    """Half side length of the container base: taken from the actual collision box (shape built by build_bin), i.e. the inscribed radius of the container's top-view outline."""
    return max(float(s.half_size[0]) for s in actor._fake_shapes if s.kind == "box")


def assert_containers_disjoint(bins) -> None:
    """Necessary condition for pairwise disjointness in top view: center distance ≥ sum of the two inscribed radii."""
    for a, b in itertools.combinations(bins, 2):
        assert np.linalg.norm(xy(a) - xy(b)) >= floor_half(a) + floor_half(b) - 1e-6, (a.name, b.name)


def hidden_under(cube, bins, tol=1e-4):
    """Which container a cube is hidden under (top-view centers coincide); returns a list of containers."""
    return [b for b in bins if np.linalg.norm(xy(cube) - xy(b)) <= tol]


def assert_distractors_match_decision(env, dist_cfg) -> None:
    assert len(env.distractor_bins) == dist_cfg["count"]
    lo, hi = dist_cfg["cube_count_range"]
    assert lo <= len(env.distractor_cubes) <= hi
    pool = set(dist_cfg["color_pool"])
    for cube in env.distractor_cubes:
        assert cube.name.rsplit("_", 1)[-1] in pool, cube.name
        # each distractor cube is hidden under exactly one distractor container
        assert len(hidden_under(cube, env.distractor_bins)) == 1, cube.name


def check_unmask_layout(task, tier, k):
    """Container count, region, hidden objects, distractors, pick count (expectations taken from the decision in the packaged header)."""
    _, env = C.replayed(task, tier, k)
    dec = O.delivered_rows(task, tier, 0)[0]["sampling_config"][task]["decision"]
    pol = dec["bin_layout_policy"]
    assert len(env.spawned_bins) == pol["count"][tier]
    c, h = pol["region_center"], pol["region_half_size"]
    for b in env.spawned_bins:
        x, y = xy(b)
        assert abs(x - c[0]) <= h + 1e-9 and abs(y - c[1]) <= h + 1e-9, b.name
    assert env.xhard_pick_count == dec["pick_count"][tier]
    # the three colored cubes are each hidden under a different in-region container
    cubes = [env.target_cube_0, env.target_cube_1, env.target_cube_2]
    homes = [hidden_under(cube, env.spawned_bins) for cube in cubes]
    assert all(len(hm) == 1 for hm in homes)
    assert len({hm[0].name for hm in homes}) == 3
    assert_distractors_match_decision(env, dec[tier]["distractor"])
    assert_containers_disjoint(list(env.spawned_bins) + list(env.distractor_bins))
