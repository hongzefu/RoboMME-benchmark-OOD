"""V5 xhard: unified distractor-container sampler and independent park points shared by the four Unmask envs (NEWTASK_RELEASE_V5_PLAN 2.2 / L13 / L14).

Only called from xhard branches; the original three tiers never enter this module, so V0/V1 are unaffected. This module **does not change** any existing shared function
(``object_generation.spawn_random_bin``, ``statechange.*``, ``bin_collision.check_swap_sweep`` are only referenced read-only);
V4's ``unmask_distractors.spawn_ring_distractor_bins`` and ``unmask_swap_xhard.sample_distractors`` are kept as-is,
and each env switches to calling this module when it adopts it.

1. Unified sampler (L13: one implementation, one schema, random stream injected by the caller)
-----------------------------------------------------------------
Per-container rejection sampling (``max_trials`` attempts per container, unified at 1024 in V5):

1. draw ``x``, ``y`` uniformly on ``[-R, R]²`` (``R = ring_max_abs_xy[1]``) (2 ``rand`` calls);
2. reject points whose ``max(|x|,|y|)`` is not in ``[r_in, r_out]`` (max-norm annulus);
3. exact 8-corner visibility: project the 8 corner points of the bottom and top faces of the container's "circumscribed square for any yaw" into the front camera; accept only if all fall inside the image
   (same criterion as V4 VU/BU's ``visible_in_camera(bin_corners(...))``, stricter than the linear approximation of the two Swap envs in V4);
4. OBB spacing: the point distance from the candidate center to each obstacle 2D OBB must be ``>= sampling half extent 0.0275 + min_gap``, ``min_gap = cube_half_size ×
   min_gap_factor`` (unified 0.75 in V5, i.e. 0.015); placed distractor containers enter the obstacles as **exact** rectangles (computed from x, y, yaw, inflated by min_gap);
5. after passing, draw yaw (1 ``rand``, ``u·90°``);
6. if the caller provides ``extra_reject`` (the sweep / path rules of the two Swap envs), it is called after yaw; returning true rejects and moves to the next attempt.
   The callback **must not** draw random numbers.

After all containers are placed, draw in order: number of cubes contained ``randint(lo, hi+1)`` -> ``cube_bins = randperm(N)[:n]`` ->
``color_order = randperm(3)``; color rule ``balanced_cycle``: the j-th cube uses ``DISTRACTOR_COLORS[color_order[j % 3]]``,
named ``distractor_cube_<j>_<colour>``. This is item-by-item identical to V4 VU/BU's sampling order and counts (V4 colors are ``randperm(3)[:n]``,
the same randperm call); without a callback the whole random call sequence is exactly V4 VU/BU's sequence.

2. Recording discipline (N18)
-------------------
The sampling function :func:`sample_distractor_layout` is **pure geometry**: it only draws random numbers, does not touch the recorder and builds no actors,
so Swap's reset planning can first use it to judge feasibility and resample the whole layout; once accepted, :func:`commit_distractor_layout` routes everything through
``recorder.value`` (each sampling point called only once), while attempt counts and failure reasons go through ``recorder.record``.
:func:`resample_distractor_layout` chains "resample whole layout -> accept -> record" into one driver. When replaying a frozen spec,
``commit`` re-checks the frozen layout by the same rules (spirit of N17); violations raise :class:`SceneGenerationError`.

Unified schema (``spec_prefix`` defaults to ``objects.distractors``, same path as V4 VU/BU)::

    objects.distractors.requested      record  requested count (written at commit)
    objects.distractors.bins.<i>       value   [x, y, yaw_deg]      decision_key …ring_max_abs_xy
    objects.distractors.placed         record  actual count
    objects.distractors.trials         record  attempts used per container
    objects.distractors.cube_count     value   number of cubes contained   decision_key …cube_count_range
    objects.distractors.cube_bins      value   container of the j-th cube  decision_key …count
    objects.distractors.color_order    value   randperm(3)          decision_key …color_rule
    objects.distractors.cube_colors    record  color name of the j-th cube
    objects.distractors.cube_names     record  actor name of the j-th cube
    layout.distractor_layout_attempts  record  (whole-layout resampling mode) index of the accepted attempt
    layout.distractor_layout_failures  record  (whole-layout resampling mode) reason of each earlier failure

3. Independent park points (L14)
---------------------
In reveal / swap windows, V4 teleported all containers and hidden cubes to the same point (10,10,10); 22-24 objects interpenetrated,
and contact solving raised per-step time from 38 ms to 108-417 ms (plan 2.2 pitfall 1). Here each object gets its own off-screen park point:
one row per group (distractor containers / distractor cubes / inner-ring containers / inner-ring hidden cubes), evenly spaced by index within a group, with adjacent points
:data:`XHARD_PARK_PITCH_M` = 0.5 m apart (max object extent about 0.085 m), also separated from (10,10,10), so they never touch.
The timelines of :func:`lift_and_park_back_to_original` / :func:`lift_and_park_onto` are step-for-step identical to the corresponding functions in ``statechange``
(window start, half-window drop step and end put-back rule unchanged); the only difference is that "far away" becomes the caller-supplied park point; the caches use separate attribute names
and do not interfere with ``statechange``'s caches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .bin_collision import bin_actor_pose, object_state_from_actor, quat_to_matrix
from .SceneGenerationError import SceneGenerationError
from .unmask_distractors import (
    DISTRACTOR_BIN_PREFIX,
    DISTRACTOR_CUBE_PREFIX,
    bin_corners,
    bin_geometry,
    in_ring,
    visible_in_camera,
)
from .xhard import DISTRACTOR_COLORS

# -- Config ------------------------------------------------------------------------
#: The only supported color rule (L12 (a)): balanced three-color rotation.
COLOR_RULE_BALANCED_CYCLE = "balanced_cycle"
#: Keys of the unified config (the ``decision.xhard.distractor`` of all four envs use this key set).
DISTRACTOR_CFG_KEYS = (
    "count",
    "ring_max_abs_xy",
    "cube_count_range",
    "color_pool",
    "color_rule",
    "min_gap_factor",
    "max_trials",
)

# Density-derivation constants of plan 2.2 (L6 / L7 / L10): VU/BU have 8 inner containers / 0.4×0.4 = 50 per m²;
# annulus width W = 1/√ρ; a gap g = 0.015 (xhard min_gap) between the annulus and the inner range; 0.0275 is the container sampling half extent.
V5_INNER_DENSITY_PER_M2 = 50.0
V5_RING_GAP_M = 0.015
V5_RING_BAND_WIDTH_M = 1.0 / math.sqrt(V5_INNER_DENSITY_PER_M2)

#: V5 xhard distractor configs of the four envs (unified keys). The env files' ``XHARD_DISTRACTOR`` should deep-copy these values directly;
#: counts and annuli are determined by the formulas in V5 plan 2.2.
V5_DISTRACTOR_PRESETS: dict[str, dict[str, Any]] = {
    # 2.3: tight annulus [0.2 + g + 0.0275, 0.2 + g + W − 0.0275], N = floor(50·0.3027 + 0.5) = 15, cubes [7,8]
    "VideoUnmask": {
        "count": 15,
        "ring_max_abs_xy": [0.2425, 0.3289],
        "cube_count_range": [7, 8],
        "color_pool": [c["name"] for c in DISTRACTOR_COLORS],
        "color_rule": COLOR_RULE_BALANCED_CYCLE,
        "min_gap_factor": 0.75,
        "max_trials": 1024,
    },
    # 2.4: same annulus, the button blocks 5.7% ⇒ A_usable 0.2853, N = 14, cubes [7,7]
    "ButtonUnmask": {
        "count": 14,
        "ring_max_abs_xy": [0.2425, 0.3289],
        "cube_count_range": [7, 7],
        "color_pool": [c["name"] for c in DISTRACTOR_COLORS],
        "color_rule": COLOR_RULE_BALANCED_CYCLE,
        "min_gap_factor": 0.75,
        "max_trials": 1024,
    },
    # 2.6 / L16 (b): reuse the V4 annulus [0.2675, 0.45], 10 containers, cubes [5,5]
    "VideoUnmaskSwap": {
        "count": 10,
        "ring_max_abs_xy": [0.2675, 0.45],
        "cube_count_range": [5, 5],
        "color_pool": [c["name"] for c in DISTRACTOR_COLORS],
        "color_rule": COLOR_RULE_BALANCED_CYCLE,
        "min_gap_factor": 0.75,
        "max_trials": 1024,
    },
    # 2.7: same as VUS
    "ButtonUnmaskSwap": {
        "count": 10,
        "ring_max_abs_xy": [0.2675, 0.45],
        "cube_count_range": [5, 5],
        "color_pool": [c["name"] for c in DISTRACTOR_COLORS],
        "color_rule": COLOR_RULE_BALANCED_CYCLE,
        "min_gap_factor": 0.75,
        "max_trials": 1024,
    },
}


@dataclass(frozen=True)
class DistractorConfig:
    """Validated unified config."""

    count: int
    ring: tuple[float, float]
    cube_count_range: tuple[int, int]
    color_rule: str
    min_gap_factor: float
    max_trials: int


def parse_distractor_cfg(cfg: dict | DistractorConfig) -> DistractorConfig:
    """Validate the ``decision.xhard.distractor`` subtree into :class:`DistractorConfig`; unknown or missing keys raise directly."""
    if isinstance(cfg, DistractorConfig):
        return cfg
    keys = set(cfg)
    missing = [k for k in DISTRACTOR_CFG_KEYS if k not in keys]
    extra = sorted(keys - set(DISTRACTOR_CFG_KEYS))
    if missing or extra:
        raise ValueError(f"distractor config keys mismatch: missing {missing}, extra {extra} (unified keys are {list(DISTRACTOR_CFG_KEYS)})")
    count = int(cfg["count"])
    # V7 fixed-value table: the tight-annulus distractors of VideoUnmask / ButtonUnmask xhard1 are 0 (0928 proposal §3.2.2, part 2 §1.7)
    if count < 0:
        raise ValueError(f"count must be >= 0, got {cfg['count']}")
    ring = tuple(float(v) for v in cfg["ring_max_abs_xy"])
    if len(ring) != 2 or not (0.0 < ring[0] < ring[1]):
        raise ValueError(f"invalid ring_max_abs_xy: {cfg['ring_max_abs_xy']}")
    lo, hi = (int(v) for v in cfg["cube_count_range"])
    # V5: with balanced color rotation the cube count is no longer limited by the number of colors; only 0 <= lo <= hi <= count is required (plan 2.3)
    if not 0 <= lo <= hi <= count:
        raise ValueError(f"cube_count_range {cfg['cube_count_range']} must satisfy 0 <= lo <= hi <= count={count}")
    pool = [c["name"] for c in DISTRACTOR_COLORS]
    if list(cfg["color_pool"]) != pool:
        # the color pool is global decision B2 (yellow/cyan/magenta), also shared by PickXtimes/SwingXtimes; must not change per env
        raise ValueError(f"color_pool must equal the global distractor color pool {pool}, got {cfg['color_pool']}")
    if cfg["color_rule"] != COLOR_RULE_BALANCED_CYCLE:
        raise ValueError(f"color_rule only supports {COLOR_RULE_BALANCED_CYCLE!r}, got {cfg['color_rule']!r}")
    gap_factor = float(cfg["min_gap_factor"])
    if not (math.isfinite(gap_factor) and gap_factor >= 0.0):
        raise ValueError(f"min_gap_factor must be a non-negative finite number, got {cfg['min_gap_factor']}")
    max_trials = int(cfg["max_trials"])
    if max_trials < 1:
        raise ValueError(f"max_trials must be >= 1, got {cfg['max_trials']}")
    return DistractorConfig(count, (ring[0], ring[1]), (lo, hi), COLOR_RULE_BALANCED_CYCLE, gap_factor, max_trials)


# -- Geometry ------------------------------------------------------------------------
Obb2d = tuple[np.ndarray, np.ndarray, np.ndarray]  # (center (2,), 2×2 with axes as columns, half extents (2,))


def bin_outer_half(cube_half_size: float) -> float:
    """Half extent of the container's outer (square) outline: inner opening half extent + wall thickness (same source as ``build_bin``; 0.03 when cube_half_size=0.02)."""
    return cube_half_size * 2.5 * 0.5 + 0.005


def _yaw_axes_from_quat(q) -> np.ndarray:
    """2D axes of an upright (including flipped 180°) object: take the two columns of the rotation matrix with the longest horizontal projection and normalize them."""
    rot = quat_to_matrix(q)
    cols = [rot[:2, k] for k in range(3)]
    norms = [float(np.linalg.norm(c)) for c in cols]
    order = sorted(range(3), key=lambda k: -norms[k])[:2]
    return np.stack([cols[k] / norms[k] for k in order], axis=1)


def bin_obb2d(x: float, y: float, yaw_deg: float, cube_half_size: float, pad: float = 0.0) -> Obb2d:
    """Compute the exact container 2D OBB from ``(x, y, yaw)`` (``build_bin`` flipped 180° then rotated by yaw about z), inflated by ``pad``."""
    _p, q = bin_actor_pose([x, y], yaw_deg, cube_half_size)
    half = bin_outer_half(cube_half_size) + float(pad)
    return (np.array([float(x), float(y)], dtype=np.float64), _yaw_axes_from_quat(q), np.array([half, half]))


def actor_obb2d_exact(actor, pad: float = 0.0) -> Obb2d:
    """Compute the 2D bounding rectangle from the actor's real collision box (inflated by ``pad``).

    Does not go through ``get_actor_obb`` -> ``_trimesh_box_to_obb2d``: for near-cubes that path puts the vertical axis into the first two columns,
    degenerating the 2D box into a line segment (plan 2.0 (1)). Here the actor's pose and ``PhysxCollisionShapeBox`` are used directly:
    upright objects take the two axes with the longest horizontal projection; otherwise fall back to a world-axis-aligned box (a conservative bound for any pose).
    """
    state = object_state_from_actor(actor)
    rot = quat_to_matrix(state.q)
    # upright (including flipped 180°): one local axis is parallel to world z, and the horizontal projections of the other two are the exact 2D axes
    upright = float(np.max(np.abs(rot[2, :]))) > 1.0 - 1e-6
    axes = _yaw_axes_from_quat(state.q) if upright else np.eye(2)
    corners = []
    for shape in state.shapes:
        center = state.p + rot @ shape.local_p
        box_axes = rot @ quat_to_matrix(shape.local_q)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append((center + box_axes @ (shape.half * np.array([sx, sy, sz])))[:2])
    local = (np.asarray(corners) - state.p[:2]) @ axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    center = state.p[:2] + axes @ ((lo + hi) / 2.0)
    return center, axes, (hi - lo) / 2.0 + float(pad)


def obstacle_obbs(avoid: Sequence[Any], min_gap: float) -> list[Obb2d]:
    """Convert the avoidance list into a list of 2D OBBs; collection convention same as V4:

    * prefab triples ``(c, A, h)`` (e.g. the button's ``create_button_obb``, safety zone included) are used as-is;
    * ``(actor, pad)`` is inflated by ``pad``; a bare actor is inflated by ``min_gap``;
    * actors get an exact box via :func:`actor_obb2d_exact`; if the box shape cannot be read, fall back to V4's trimesh path, and if that fails too, ignore it
      (same as V4 ``_obstacle_obbs``: objects without a physical mesh take no part in avoidance).
    """
    out: list[Obb2d] = []
    for item in avoid:
        if isinstance(item, tuple):
            if len(item) == 3 and isinstance(item[0], np.ndarray) and isinstance(item[1], np.ndarray):
                out.append((np.asarray(item[0], np.float64)[:2], np.asarray(item[1], np.float64),
                            np.asarray(item[2], np.float64)[:2]))
                continue
            actor, pad = item
        else:
            actor, pad = item, min_gap
        try:
            out.append(actor_obb2d_exact(actor, float(pad)))
            continue
        except Exception:  # noqa: BLE001 non-box shapes etc.: fall back to the V4 path
            pass
        try:
            from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb

            from .object_generation import _trimesh_box_to_obb2d

            out.append(_trimesh_box_to_obb2d(get_actor_obb(actor, to_world_frame=True, vis=False), extra_pad=float(pad)))
        except Exception:  # noqa: BLE001 consistent with V4: objects without a physical mesh are ignored
            pass
    return out


def point_hits_obbs(pos: np.ndarray, obbs: Sequence[Obb2d], reach: float) -> bool:
    """True if the point's distance to any OBB is < ``reach`` (same criterion as V4 ``unmask_distractors._hits``)."""
    for c_obs, a_obs, h_obs in obbs:
        local = a_obs.T @ (pos - c_obs)
        closest = c_obs + a_obs @ np.clip(local, -h_obs, h_obs)
        if np.linalg.norm(pos - closest) < reach:
            return True
    return False


def bin_visible(x: float, y: float, cube_half_size: float) -> bool:
    """Exact 8-corner visibility criterion (same as V4 VU/BU)."""
    _half, reach_any_yaw, height = bin_geometry(cube_half_size)
    return visible_in_camera(bin_corners(float(x), float(y), reach_any_yaw, height))


# -- Placement result ----------------------------------------------------------------
@dataclass
class DistractorLayout:
    """A complete distractor layout (pure geometry, no actors)."""

    bins: list[tuple[float, float, float]]  # per container (x, y, yaw_deg)
    cube_count: int
    cube_bins: list[int]  # index of the container holding the j-th cube
    color_order: list[int]  # randperm(3), indices into DISTRACTOR_COLORS
    trials: list[int] = field(default_factory=list)  # attempts used per container (trace only)

    @property
    def count(self) -> int:
        return len(self.bins)

    @property
    def cube_color_indices(self) -> list[int]:
        """balanced_cycle: the j-th cube uses ``color_order[j % 3]``."""
        n = len(self.color_order)
        return [int(self.color_order[j % n]) for j in range(len(self.cube_bins))]

    @property
    def cube_colors(self) -> list[str]:
        return [DISTRACTOR_COLORS[i]["name"] for i in self.cube_color_indices]

    @property
    def bin_names(self) -> list[str]:
        return [f"{DISTRACTOR_BIN_PREFIX}_{i}" for i in range(len(self.bins))]

    @property
    def cube_names(self) -> list[str]:
        return [f"{DISTRACTOR_CUBE_PREFIX}_{j}_{color}" for j, color in enumerate(self.cube_colors)]

    def bin_obbs(self, cube_half_size: float, pad: float = 0.0) -> list[Obb2d]:
        return [bin_obb2d(x, y, yaw, cube_half_size, pad) for x, y, yaw in self.bins]

    def same_geometry(self, other: "DistractorLayout") -> bool:
        """Whether positions, cube mapping and colors are identical value by value (trials not compared)."""
        return (
            [list(map(float, b)) for b in self.bins] == [list(map(float, b)) for b in other.bins]
            and int(self.cube_count) == int(other.cube_count)
            and [int(v) for v in self.cube_bins] == [int(v) for v in other.cube_bins]
            and [int(v) for v in self.color_order] == [int(v) for v in other.color_order]
        )

    def to_spec(self) -> dict[str, Any]:
        """Pure-data view of the unified schema (fields correspond one-to-one to what ``commit_distractor_layout`` writes into the spec)."""
        return {
            "bins": [[float(x), float(y), float(yaw)] for x, y, yaw in self.bins],
            "cube_count": int(self.cube_count),
            "cube_bins": [int(v) for v in self.cube_bins],
            "color_order": [int(v) for v in self.color_order],
            "cube_colors": self.cube_colors,
            "cube_names": self.cube_names,
            "bin_names": self.bin_names,
        }


class DistractorPlacementError(SceneGenerationError):
    """A container found no legal position within ``max_trials`` (candidate-level rejection). ``placed`` is the number already placed."""

    def __init__(self, message: str, placed: int):
        super().__init__(message)
        self.placed = int(placed)


#: Extra rejection callback: ``(index i, x, y, yaw_deg, list of placed containers) -> reject?``; must not draw random numbers.
ExtraReject = Callable[[int, float, float, float, Sequence[tuple[float, float, float]]], bool]


def sample_distractor_layout(
    cfg: dict | DistractorConfig,
    *,
    obstacles: Sequence[Obb2d],
    generator: torch.Generator,
    cube_half_size: float,
    extra_reject: ExtraReject | None = None,
) -> DistractorLayout:
    """Draw a whole distractor layout purely geometrically (no actors built, recorder untouched).

    ``obstacles`` are the 2D OBBs of objects already in the scene (converted from the avoidance list with :func:`obstacle_obbs`; actors already inflated by min_gap).
    See the module docs for the random call order; raises :class:`DistractorPlacementError` when a container cannot be placed.
    """
    c = parse_distractor_cfg(cfg)
    half, _reach_any_yaw, _height = bin_geometry(cube_half_size)
    min_gap = float(cube_half_size) * c.min_gap_factor
    reject_reach = half + min_gap
    span = c.ring[1]
    obbs = list(obstacles)
    bins: list[tuple[float, float, float]] = []
    trials: list[int] = []
    for i in range(c.count):
        placed = None
        used = 0
        for _ in range(c.max_trials):
            used += 1
            x = float(torch.rand(1, generator=generator).item() * 2.0 * span - span)
            y = float(torch.rand(1, generator=generator).item() * 2.0 * span - span)
            if not in_ring(x, y, c.ring):
                continue
            if not bin_visible(x, y, cube_half_size):
                continue
            if point_hits_obbs(np.array([x, y], dtype=np.float64), obbs, reject_reach):
                continue
            yaw = float(torch.rand(1, generator=generator).item() * 90.0)
            if extra_reject is not None and extra_reject(i, x, y, yaw, list(bins)):
                continue
            placed = (x, y, yaw)
            break
        if placed is None:
            raise DistractorPlacementError(
                f"xhard distractor containers cannot all be placed: requested {c.count}, container {i} has no feasible position within {c.max_trials} attempts", placed=len(bins)
            )
        bins.append(placed)
        trials.append(used)
        obbs.append(bin_obb2d(placed[0], placed[1], placed[2], cube_half_size, pad=min_gap))

    lo, hi = c.cube_count_range
    n_cubes = int(torch.randint(lo, hi + 1, (1,), generator=generator).item())
    cube_bins = torch.randperm(c.count, generator=generator)[:n_cubes].tolist()
    color_order = torch.randperm(len(DISTRACTOR_COLORS), generator=generator).tolist()
    return DistractorLayout(bins=bins, cube_count=n_cubes, cube_bins=cube_bins, color_order=color_order, trials=trials)


def verify_distractor_layout(
    layout: DistractorLayout,
    cfg: dict | DistractorConfig,
    *,
    obstacles: Sequence[Obb2d],
    cube_half_size: float,
    extra_reject: ExtraReject | None = None,
) -> list[str]:
    """Re-check a whole layout by the same rules as sampling and return the violations (empty list means valid); draws no random numbers."""
    c = parse_distractor_cfg(cfg)
    half, _r, _h = bin_geometry(cube_half_size)
    min_gap = float(cube_half_size) * c.min_gap_factor
    reach = half + min_gap
    problems: list[str] = []
    if len(layout.bins) != c.count:
        problems.append(f"container count {len(layout.bins)} != {c.count}")
    obbs = list(obstacles)
    for i, (x, y, yaw) in enumerate(layout.bins):
        if not in_ring(x, y, c.ring):
            problems.append(f"container {i} not in annulus {list(c.ring)}: ({x:.4f},{y:.4f})")
        if not bin_visible(x, y, cube_half_size):
            problems.append(f"container {i} not inside the image: ({x:.4f},{y:.4f})")
        if point_hits_obbs(np.array([x, y], dtype=np.float64), obbs, reach):
            problems.append(f"container {i} has insufficient spacing to obstacles or placed containers: ({x:.4f},{y:.4f})")
        if not 0.0 <= float(yaw) <= 90.0:
            problems.append(f"container {i} yaw {yaw} not in [0,90]")
        if extra_reject is not None and extra_reject(i, float(x), float(y), float(yaw), list(layout.bins[:i])):
            problems.append(f"container {i} rejected by the caller's extra rule")
        obbs.append(bin_obb2d(x, y, yaw, cube_half_size, pad=min_gap))
    lo, hi = c.cube_count_range
    if not lo <= int(layout.cube_count) <= hi:
        problems.append(f"number of cubes contained {layout.cube_count} not in [{lo},{hi}]")
    cube_bins = [int(v) for v in layout.cube_bins]
    if len(cube_bins) != int(layout.cube_count) or len(set(cube_bins)) != len(cube_bins) or any(
        not 0 <= v < len(layout.bins) for v in cube_bins
    ):
        problems.append(f"cube_bins {cube_bins} inconsistent with count {layout.cube_count} / number of containers {len(layout.bins)}: mismatch")
    if sorted(int(v) for v in layout.color_order) != list(range(len(DISTRACTOR_COLORS))):
        problems.append(f"color_order {layout.color_order} is not a permutation of 0..{len(DISTRACTOR_COLORS) - 1} (inclusive)")
    return problems


def commit_distractor_layout(
    layout: DistractorLayout,
    *,
    cfg: dict | DistractorConfig,
    recorder,
    obstacles: Sequence[Obb2d],
    cube_half_size: float,
    spec_prefix: str = "objects.distractors",
    decision_prefix: str = "xhard.distractor",
    extra_reject: ExtraReject | None = None,
) -> DistractorLayout:
    """Write an **already accepted** layout into the spec via ``recorder`` and return the layout actually used to build the scene.

    Export mode returns the original layout; replay mode returns the layout assembled from frozen values and re-checks it by the same rules (spirit of N17); violations raise
    :class:`SceneGenerationError`. Each sampling point calls ``recorder.value`` only once (N18).
    """
    c = parse_distractor_cfg(cfg)
    recorder.record(f"{spec_prefix}.requested", c.count)
    bins = []
    for i, entry in enumerate(layout.bins):
        x, y, yaw = recorder.value(f"{spec_prefix}.bins.{i}", [float(v) for v in entry],
                                   decision_key=f"{decision_prefix}.ring_max_abs_xy")
        bins.append((float(x), float(y), float(yaw)))
    recorder.record(f"{spec_prefix}.placed", len(bins))
    recorder.record(f"{spec_prefix}.trials", [int(v) for v in layout.trials])
    cube_count = int(recorder.value(f"{spec_prefix}.cube_count", int(layout.cube_count),
                                    decision_key=f"{decision_prefix}.cube_count_range"))
    cube_bins = [int(v) for v in recorder.value(f"{spec_prefix}.cube_bins", [int(v) for v in layout.cube_bins],
                                                decision_key=f"{decision_prefix}.count")]
    color_order = [int(v) for v in recorder.value(f"{spec_prefix}.color_order", [int(v) for v in layout.color_order],
                                                  decision_key=f"{decision_prefix}.color_rule")]
    final = DistractorLayout(bins=bins, cube_count=cube_count, cube_bins=cube_bins, color_order=color_order,
                             trials=list(layout.trials))
    # in replay the frozen layout must be re-checked against this tier's obstacles and extra rules (e.g. this tier's inner-ring sweep)
    if getattr(recorder, "replaying", False):
        problems = verify_distractor_layout(final, c, obstacles=obstacles, cube_half_size=cube_half_size,
                                            extra_reject=extra_reject)
        if problems:
            raise SceneGenerationError("replayed frozen distractor layout violates V5 rules: " + "; ".join(problems))
    recorder.record(f"{spec_prefix}.cube_colors", final.cube_colors)
    recorder.record(f"{spec_prefix}.cube_names", final.cube_names)
    return final


#: Accept callback of whole-layout resampling mode: ``layout -> (accepted?, attached result when accepted / reason string when rejected)``.
#: The callback may keep drawing from the same random stream (e.g. ``randperm`` for the Swap outer-ring initiator); this is part of the caller's random-stream design.
AcceptFn = Callable[[DistractorLayout], tuple[bool, Any]]


def resample_distractor_layout(
    cfg: dict | DistractorConfig,
    *,
    obstacles: Sequence[Obb2d],
    generator: torch.Generator,
    cube_half_size: float,
    recorder,
    accept: AcceptFn,
    max_attempts: int,
    extra_reject: ExtraReject | None = None,
    spec_prefix: str = "objects.distractors",
    decision_prefix: str = "xhard.distractor",
    attempts_path: str = "layout.distractor_layout_attempts",
    failures_path: str = "layout.distractor_layout_failures",
    require_replay_match: bool = True,
) -> tuple[DistractorLayout, Any]:
    """Whole-layout resampling driver: at most ``max_attempts`` rounds of "place -> accept"; only the first accepted layout goes through ``recorder.value`` (N18).

    * a placement that cannot place all containers (:class:`DistractorPlacementError`) counts as one failed attempt (reason ``placement``) and resampling continues;
    * ``accept`` returning ``(False, reason)`` likewise counts as a failed attempt;
    * once accepted, ``record`` the attempt index and earlier failure reasons, then :func:`commit_distractor_layout`;
    * in replay mode, if the frozen layout differs from the layout accepted this time (meaning code or rules changed), the attached result of ``accept`` no longer corresponds to the frozen layout,
      and :class:`SceneGenerationError` is raised when ``require_replay_match`` is true;
    * if all attempts fail, ``record`` the failure reasons and raise :class:`SceneGenerationError` (candidate-level resampling).
    """
    attempts = int(max_attempts)
    if attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            layout = sample_distractor_layout(cfg, obstacles=obstacles, generator=generator,
                                              cube_half_size=cube_half_size, extra_reject=extra_reject)
        except DistractorPlacementError:
            failures.append("placement")
            continue
        ok, payload = accept(layout)
        if not ok:
            failures.append(str(payload) if payload else "rejected")
            continue
        recorder.record(attempts_path, attempt)
        recorder.record(failures_path, list(failures))
        final = commit_distractor_layout(layout, cfg=cfg, recorder=recorder, obstacles=obstacles,
                                         cube_half_size=cube_half_size, spec_prefix=spec_prefix,
                                         decision_prefix=decision_prefix, extra_reject=extra_reject)
        if require_replay_match and getattr(recorder, "replaying", False) \
                and not final.same_geometry(layout):
            raise SceneGenerationError("replayed frozen distractor layout differs from the layout accepted in this resampling; the planning result cannot correspond to the frozen layout")
        return final, payload
    recorder.record(attempts_path, attempts)
    recorder.record(failures_path, list(failures))
    raise SceneGenerationError(f"distractor layout infeasible after {attempts} whole-layout resamplings: {failures}")


# -- Build actors ---------------------------------------------------------------------
def build_distractor_actors(env, layout: DistractorLayout, *, hidden_half_size: float):
    """Build distractor containers and cubes from the layout; returns ``(bins, cubes)``; the j-th cube is placed at the XY of ``bins[cube_bins[j]]``.

    Container names ``distractor_bin_<i>``, cube names ``distractor_cube_<j>_<colour>`` (indexed, so repeated colors never collide);
    **no ``bin_<i>`` attribute is set**, so the reveal / swap logic that scans ``bin_<i>`` does not pick them up by mistake.
    """
    from .object_generation import build_bin, spawn_fixed_cube

    names = layout.bin_names + layout.cube_names
    if len(set(names)) != len(names):
        raise SceneGenerationError(f"duplicate distractor actor names: {names}")
    bins = [
        build_bin(env, callsign=name, position=[x, y, 0.002], z_rotation_deg=yaw)
        for name, (x, y, yaw) in zip(layout.bin_names, layout.bins)
    ]
    cubes = []
    for j, (b_idx, name, c_idx) in enumerate(zip(layout.cube_bins, layout.cube_names, layout.cube_color_indices)):
        x, y, _yaw = layout.bins[int(b_idx)]
        cubes.append(spawn_fixed_cube(
            env,
            position=[float(x), float(y)],
            half_size=hidden_half_size,
            color=DISTRACTOR_COLORS[int(c_idx)]["rgba"],
            name_prefix=name,
            yaw=0.0,
            dynamic=True,
        ))
    return bins, cubes


def distractor_cube_bin_pairs(layout: DistractorLayout, bins: Sequence[Any], cubes: Sequence[Any]):
    """``[(cube_j, its container)]``; used by the two Swap envs when outer-ring cubes follow their containers in swap windows."""
    return [(cubes[j], bins[int(b_idx)]) for j, b_idx in enumerate(layout.cube_bins)]


def spawn_distractor_layout(
    env,
    *,
    cfg: dict | DistractorConfig,
    avoid: list,
    generator: torch.Generator,
    recorder,
    hidden_half_size: float,
    spec_prefix: str = "objects.distractors",
    decision_prefix: str = "xhard.distractor",
):
    """One-stop entry for VU / BU: avoidance list -> pure-geometry sampling -> recording -> building actors; returns ``(bins, cubes, layout)``.

    Same convention as V4 ``spawn_ring_distractor_bins``: the caller uses the main scene generator and places it after all existing sampling points (N5);
    placed containers are appended to ``avoid``; if not all can be placed, :class:`SceneGenerationError` is raised directly (no silent truncation).
    """
    c = parse_distractor_cfg(cfg)
    chs = float(env.cube_half_size)
    obstacles = obstacle_obbs(avoid, chs * c.min_gap_factor)
    try:
        layout = sample_distractor_layout(c, obstacles=obstacles, generator=generator, cube_half_size=chs)
    except DistractorPlacementError as exc:
        recorder.record(f"{spec_prefix}.requested", c.count)
        recorder.record(f"{spec_prefix}.placed", exc.placed)
        raise
    layout = commit_distractor_layout(layout, cfg=c, recorder=recorder, obstacles=obstacles, cube_half_size=chs,
                                      spec_prefix=spec_prefix, decision_prefix=decision_prefix)
    bins, cubes = build_distractor_actors(env, layout, hidden_half_size=hidden_half_size)
    avoid.extend(bins)
    return bins, cubes, layout


# -- Independent park points (L14) --------------------------------------------------------------
#: Origin of the park grid. Separated from statechange's (10,10,10) by >= 14 m, behind the front camera, far from the table and robot arm.
XHARD_PARK_ORIGIN = (20.0, 20.0, 10.0)
#: Spacing between adjacent park points (meters); the max object diagonal is about 0.085 m, and objects are teleported back to the park point on every control step in the window, falling freely less than 2 cm.
XHARD_PARK_PITCH_M = 0.5
#: Number of park points per row.
XHARD_PARK_ROW_LEN = 16
#: Max park points per group (4 rows).
XHARD_PARK_GROUP_CAPACITY = 64
#: Groups: distractor containers, distractor cubes, inner-ring containers, inner-ring hidden cubes each occupy a non-overlapping block.
XHARD_PARK_GROUPS = ("distractor_bin", "distractor_cube", "bin", "hidden_cube")


def xhard_park_point(group: str, index: int) -> np.ndarray:
    """Park point (float32 xyz) of the ``index``-th object in group ``group``. Pure function, independent of call order."""
    if group not in XHARD_PARK_GROUPS:
        raise ValueError(f"unknown park group {group!r}, choices {XHARD_PARK_GROUPS}")
    index = int(index)
    if not 0 <= index < XHARD_PARK_GROUP_CAPACITY:
        raise ValueError(f"park index {index} exceeds per-group capacity {XHARD_PARK_GROUP_CAPACITY}")
    rows_per_group = XHARD_PARK_GROUP_CAPACITY // XHARD_PARK_ROW_LEN
    row = XHARD_PARK_GROUPS.index(group) * rows_per_group + index // XHARD_PARK_ROW_LEN
    col = index % XHARD_PARK_ROW_LEN
    x0, y0, z0 = XHARD_PARK_ORIGIN
    return np.array([x0 + col * XHARD_PARK_PITCH_M, y0 + row * XHARD_PARK_PITCH_M, z0], dtype=np.float32)


def _pose_quat(obj) -> np.ndarray:
    quat = obj.pose.q if hasattr(obj, "pose") else obj.get_pose().q
    if hasattr(quat, "detach"):
        quat = quat.detach().cpu().numpy()
    return np.asarray(quat, dtype=np.float32).flatten()


def _pose_xyz(obj) -> np.ndarray:
    p = obj.pose.p if hasattr(obj, "pose") else obj.get_pose().p
    if hasattr(p, "detach"):
        p = p.detach().cpu().numpy()
    return np.asarray(p, dtype=np.float64).reshape(-1)[:3]


def _teleport(obj, target_pos, quat) -> None:
    """Same semantics as ``_teleport`` in statechange: set the pose and zero the velocity; failures are just swallowed."""
    import sapien

    try:
        obj.set_pose(sapien.Pose(p=[float(v) for v in target_pos], q=quat))
        try:
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def lift_and_park_back_to_original(env, obj, start_step: int, end_step: int, cur_step: int, park_xyz) -> None:
    """Park-point version of ``statechange.lift_and_drop_objects_back_to_original``, with a step-for-step identical timeline:

    each step in ``[start, drop)`` teleports to ``park_xyz``; at step ``drop = min(end, start + max(1, (end-start)//2))`` put back to
    the original position recorded on first window entry (including the original quaternion); no motion on other steps. Cache attribute ``_xhard_park_cache``, separate from statechange's.
    """
    start_step, end_step, cur_step = int(start_step), int(end_step), int(cur_step)
    if not hasattr(env, "_xhard_park_cache"):
        env._xhard_park_cache = {}
    cache_all = env._xhard_park_cache
    key = (id(obj), start_step, end_step)
    if cur_step > end_step:
        cache_all.pop(key, None)
        return
    if cur_step < start_step:
        return
    cache = cache_all.get(key)
    if cache is None:
        duration = max(1, end_step - start_step)
        cache = {
            "origin": _pose_xyz(obj).astype(np.float32),
            "quat": _pose_quat(obj),
            "park": np.asarray(park_xyz, dtype=np.float32).reshape(3),
            "drop_step": min(end_step, start_step + max(1, duration // 2)),
        }
        cache_all[key] = cache
    if cur_step < cache["drop_step"]:
        _teleport(obj, cache["park"], cache["quat"])
        return
    if cur_step == cache["drop_step"]:
        _teleport(obj, cache["origin"], cache["quat"])
        return
    cache_all.pop(key, None)


def lift_and_park_onto(env, obj_a, obj_b, start_step: int, end_step: int, cur_step: int, park_xyz) -> None:
    """Park-point version of ``statechange.lift_and_drop_objectA_onto_objectB``, with a step-for-step identical timeline:

    each step in ``[start, end)`` teleports ``obj_a`` to ``park_xyz``; at step ``end`` it is placed at ``obj_b``'s current XY, at ``obj_a``'s height on first window entry.
    Cache attribute ``_xhard_park_onto_cache``.
    """
    start_step, end_step, cur_step = int(start_step), int(end_step), int(cur_step)
    if not hasattr(env, "_xhard_park_onto_cache"):
        env._xhard_park_onto_cache = {}
    cache_all = env._xhard_park_onto_cache
    key = (id(obj_a), id(obj_b), start_step, end_step)
    if cur_step < start_step:
        return
    if cur_step > end_step:
        cache_all.pop(key, None)
        return
    cache = cache_all.get(key)
    if cache is None:
        cache = {
            "quat_a": _pose_quat(obj_a),
            "park": np.asarray(park_xyz, dtype=np.float32).reshape(3),
            "origin_z": float(_pose_xyz(obj_a)[2]),
        }
        cache_all[key] = cache
    if cur_step < end_step:
        _teleport(obj_a, cache["park"], cache["quat_a"])
        return
    bx, by, _bz = _pose_xyz(obj_b)
    _teleport(obj_a, np.array([bx, by, cache["origin_z"]], dtype=np.float32), cache["quat_a"])
    cache_all.pop(key, None)


def reveal_actors_parked(env, actors: Sequence[Any], *, group: str, start_step: int, end_step: int, cur_step: int) -> None:
    """Apply :func:`lift_and_park_back_to_original` to a group of actors one by one; the i-th parks at ``xhard_park_point(group, i)``."""
    for index, actor in enumerate(actors):
        if actor is None:
            continue
        lift_and_park_back_to_original(env, actor, start_step, end_step, cur_step, xhard_park_point(group, index))


def reveal_distractor_bins_parked(env, *, start_step: int, end_step: int, cur_step: int) -> None:
    """Park-point version of ``unmask_distractors.reveal_distractor_bins``: window and drop step unchanged, one park point per distractor container."""
    reveal_actors_parked(env, list(getattr(env, "distractor_bins", None) or []), group="distractor_bin",
                         start_step=start_step, end_step=end_step, cur_step=cur_step)


def park_cubes_onto_bins(env, pairs: Sequence[tuple[Any, Any]], *, group: str, start_step: int, end_step: int,
                         cur_step: int) -> None:
    """Apply :func:`lift_and_park_onto` to each ``[(cube, bin)]`` pair; the j-th cube parks at ``xhard_park_point(group, j)``.

    The two Swap envs: inner-ring ``cube_bin_pairs`` use ``group="hidden_cube"``, outer-ring cubes use ``group="distractor_cube"``.
    """
    for index, (cube, bin_actor) in enumerate(pairs):
        if cube is None or bin_actor is None:
            continue
        lift_and_park_onto(env, cube, bin_actor, start_step, end_step, cur_step, xhard_park_point(group, index))
