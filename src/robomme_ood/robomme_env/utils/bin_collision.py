"""Real collision-box criteria between containers / cubes (NEW_VALUE_INJECTION_TEST_PLAN section 3 and sections 8.1-8.3).

This module does geometry only: it builds no actors, samples nothing and does not touch the SAPIEN scene, so the external CPU spec generator and the simulation
runtime can share one criterion, avoiding a double source of truth of "one geometry for screening, another for actual runs".

Three layers of interface:

* :func:`bin_shape_specs` / :func:`cube_shape_specs` -- local box descriptions from the same source as
  ``object_generation.py::build_bin`` (6 boxes for a container, 1 for a cube).
* :func:`check_bin_layout` -- static criterion: 3D separating-axis test over every box pair of every object pair;
  only ``g > ε`` counts as separated, ``0 <= g <= ε`` is recorded as contact and ``g < 0`` as penetration, both excluded.
* :func:`check_swap_sweep` -- continuous criterion: replicates the curve and quaternion interpolation of
  ``statechange.py::swap_flat_two_lane`` and proves by interval bisection that the whole path
  stays separated; anything that cannot be proven is excluded as ``uncertified``, never relaxed to frame sampling.
* :func:`check_multi_swap_sweep` -- added in V5: joint continuous criterion for several pairs swapping simultaneously in the same window, reusing the same
  interval-bisection proof; with a certified prefilter enabled by default (401 sample points + Lipschitz bound, only skipping pairs proven separated).
  :func:`check_swap_sweep_prefiltered` is its single-pair wrapper; helpers such as :func:`static_box_state` build
  arbitrary oriented boxes as static obstacles (button base, distractor containers, etc.). :func:`check_swap_sweep` itself does no prefiltering; its verdict is unchanged.

All criterion values are fixed in module constants: ``EPS_M``, ``MAX_DEPTH``, ``MAX_INTERVALS``,
``DEGENERATE_NORM``. No switch is provided to relax thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "BinCollisionError",
    "SpecBindingError",
    "nearest_partner_index",
    "CollisionRejection",
    "ShapeSpec",
    "ObjectState",
    "EPS_M",
    "MAX_DEPTH",
    "MAX_INTERVALS",
    "DEGENERATE_NORM",
    "LANE_OFFSET",
    "bin_shape_specs",
    "cube_shape_specs",
    "bin_actor_pose",
    "cube_actor_pose",
    "quat_to_matrix",
    "euler_xyz_to_quat",
    "sat_gap",
    "check_pair_static",
    "check_bin_layout",
    "check_swap_sweep",
    "PREFILTER_SAMPLES",
    "PREFILTER_MARGIN_M",
    "check_multi_swap_sweep",
    "check_swap_sweep_prefiltered",
    "static_box_state",
    "static_rect_state",
    "static_state_from_obb2d",
    "button_base_state",
    "check_bin_state",
    "shape_specs_from_actor",
    "object_state_from_actor",
]


# -- Fixed criterion values -------------------------------------------------------------
#: Separation threshold in meters. 1e-6 m = 0.001 mm; no extra safety gap is added.
EPS_M = 1e-6
#: Max interval-bisection depth for a single box pair on a single path segment.
MAX_DEPTH = 20
#: Max number of intervals examined for a single box pair on a single path segment.
MAX_INTERVALS = 4096
#: Lower bound on the norm of the linear quaternion blend; below it the interpolation is considered degenerate and no angular velocity bound can be given.
DEGENERATE_NORM = 1e-6
#: Actual lateral curve offset of ``swap_flat_two_lane`` in the two video tasks, meters.
LANE_OFFSET = 0.07
#: Threshold at which ``swap_flat_two_lane`` judges "start and end coincide, normal set to zero"; same as the original function.
COINCIDENT_XY = 1e-9


class BinCollisionError(RuntimeError):
    """A spec or runtime state failed the collision criterion. ``rejection`` carries structured rejection evidence."""

    def __init__(self, rejection: "CollisionRejection") -> None:
        super().__init__(rejection.summary())
        self.rejection = rejection


class SpecBindingError(RuntimeError):
    """The objects / actions measured at runtime differ from those pre-written in the spec; corresponds to "actual object / action mismatch" among the seven result classes.

    The most typical case is the swap partner: the ``partner`` pre-written in the spec is the design convention (computed from the nominal pose after the previous segment ends),
    while ``step`` recomputes the nearest neighbor from the **actual** poses at the moment the swap starts. Step 0a of section 5 of the plan explicitly requires
    "mismatch means failure, switching partners is forbidden" -- so this raises to abort the sample and never continues with the actual partner on the spot.

    ``detail`` carries the full candidate distance table, to judge afterwards whether it was a borderline tie or a genuine mis-binding.
    """

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


@dataclass(frozen=True)
class CollisionRejection:
    """Complete evidence of one rejection. ``reason`` takes only three named values.

    * ``contact``: ``g < 0``, the two boxes already penetrate.
    * ``numerical_boundary``: ``0 <= g <= ε`` (including exact tangency ``g == 0``), or a non-finite criterion value.
    * ``uncertified``: depth / interval count exhausted, quaternion degenerate, bound not computable -- safety cannot be proven.
    """

    reason: str
    stage: str  # "initial" / "sweep" / "state"
    object_a: str
    object_b: str
    shape_a: int
    shape_b: int
    gap_m: float | None = None
    s: float | None = None
    sweep_index: int | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    depth: int | None = None
    intervals_used: int | None = None
    detail: str = ""

    def summary(self) -> str:
        where = f"{self.stage}"
        if self.sweep_index is not None:
            where += f"#{self.sweep_index}"
        gap = "n/a" if self.gap_m is None else f"{self.gap_m:.12g}"
        return (
            f"collision excluded [{self.reason}] {where} object {self.object_a}(shape {self.shape_a})"
            f" vs {self.object_b}(shape {self.shape_b}) g={gap}"
            + (f" s={self.s:.12g}" if self.s is not None else "")
            + (f" {self.detail}" if self.detail else "")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "stage": self.stage,
            "object_a": self.object_a,
            "object_b": self.object_b,
            "shape_a": self.shape_a,
            "shape_b": self.shape_b,
            "gap_m": self.gap_m,
            "s": self.s,
            "sweep_index": self.sweep_index,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "depth": self.depth,
            "intervals_used": self.intervals_used,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ShapeSpec:
    """Pose and half sizes of one box in the actor's local frame."""

    local_p: np.ndarray  # (3,)
    local_q: np.ndarray  # (4,) wxyz
    half: np.ndarray  # (3,)

    def vertex_radius(self) -> float:
        """Max distance from the box's 8 vertices to the actor origin, used for the linear velocity bound of the rotation term."""
        rot = quat_to_matrix(self.local_q)
        best = 0.0
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    offset = rot @ (self.half * np.array([sx, sy, sz]))
                    best = max(best, float(np.linalg.norm(self.local_p + offset)))
        return best


@dataclass
class ObjectState:
    """Current world pose of one object in the scene and all its boxes."""

    name: str
    p: np.ndarray  # (3,)
    q: np.ndarray  # (4,) wxyz
    shapes: Sequence[ShapeSpec]
    radii: tuple[float, ...] = field(default=())

    def __post_init__(self) -> None:
        self.p = np.asarray(self.p, dtype=np.float64).reshape(3)
        self.q = _normalize_quat(np.asarray(self.q, dtype=np.float64).reshape(4))
        if not self.radii:
            self.radii = tuple(shape.vertex_radius() for shape in self.shapes)


# -- Quaternions and rotations -------------------------------------------------------------
def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= DEGENERATE_NORM:
        raise ValueError(f"degenerate quaternion, norm {norm}")
    return q / norm


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """wxyz quaternion to 3×3 rotation matrix (same wxyz order as SAPIEN / ManiSkill)."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    cos, sin = math.cos(angle), math.sin(angle)
    if axis == "X":
        return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float64)
    if axis == "Y":
        return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float64)
    if axis == "Z":
        return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"unknown rotation axis {axis}")


def euler_xyz_to_quat(angles_rad: Sequence[float]) -> np.ndarray:
    """Replicate ``euler_angles_to_matrix(..., convention="XYZ")`` and then take the quaternion.

    ManiSkill's ``rotation_conversions`` is a port of pytorch3d; ``"XYZ"`` means
    ``R = Rx(a) @ Ry(b) @ Rz(c)``; both ``build_bin`` and ``spawn_fixed_cube`` go through this path.
    """
    a, b, c = (float(v) for v in angles_rad)
    matrix = _axis_rotation("X", a) @ _axis_rotation("Y", b) @ _axis_rotation("Z", c)
    return matrix_to_quat(matrix)


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix to wxyz quaternion, sign chosen with w >= 0, consistent with pytorch3d's implementation."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    if quat[0] < 0.0:
        quat = -quat
    return _normalize_quat(quat)


# -- Geometry descriptions from the same source as build_bin / spawn_random_cube ---------------------------
def bin_shape_specs(cube_half_size: float) -> tuple[ShapeSpec, ...]:
    """The container's 6 boxes, replicating ``build_bin``'s ``poses`` and ``half_sizes`` word for word.

    ⚠ ``build_bin``'s comment says "floor plate plus four walls", but the source also builds a central block, so there are actually 6.
    """
    inner_side = cube_half_size * 2.5
    wall_thickness = 0.005
    wall_height = cube_half_size * 2.5
    floor_thickness = 0.004

    inner_half = inner_side * 0.5
    t = wall_thickness * 0.5
    h = wall_height * 0.5
    tf = floor_thickness * 0.5

    bottom_half = [inner_half + t, inner_half + t, tf]
    lr_wall_half = [t, inner_half + t, h]
    fb_wall_half = [inner_half + t, t, h]

    base_z = tf
    offset = inner_half + t
    z_wall = tf + h

    identity = np.array([1.0, 0.0, 0.0, 0.0])
    local = [
        ([0.0, 0.0, 0.0], [cube_half_size] * 3),
        ([0.0, 0.0, base_z], bottom_half),
        ([-offset, 0.0, z_wall], lr_wall_half),
        ([+offset, 0.0, z_wall], lr_wall_half),
        ([0.0, -offset, z_wall], fb_wall_half),
        ([0.0, +offset, z_wall], fb_wall_half),
    ]
    return tuple(
        ShapeSpec(np.array(p, dtype=np.float64), identity.copy(), np.array(half, dtype=np.float64))
        for p, half in local
    )


def cube_shape_specs(half_size: float) -> tuple[ShapeSpec, ...]:
    """The cube's single box: the collision body of ``actors.build_cube`` sits at the actor origin."""
    return (
        ShapeSpec(
            np.zeros(3, dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.full(3, float(half_size), dtype=np.float64),
        ),
    )


def bin_actor_pose(xy: Sequence[float], z_rotation_deg: float, cube_half_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Replicate ``build_bin``'s ``builder.set_initial_pose``: flipped 180° and upside down on the table."""
    wall_height = cube_half_size * 2.5
    floor_thickness = 0.004
    h = wall_height * 0.5
    tf = floor_thickness * 0.5
    p = np.array([float(xy[0]), float(xy[1]), tf + 2.0 * h], dtype=np.float64)
    q = euler_xyz_to_quat([math.pi, 0.0, math.radians(float(z_rotation_deg))])
    return p, q


def cube_actor_pose(xy: Sequence[float], yaw_rad: float, half_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Replicate ``spawn_random_cube``'s placement: bottom face on the table, yaw about the z axis."""
    p = np.array([float(xy[0]), float(xy[1]), float(half_size)], dtype=np.float64)
    q = euler_xyz_to_quat([0.0, 0.0, float(yaw_rad)])
    return p, q


# -- Static criterion: 3D separating axes -----------------------------------------------------
def _world_box(state_p: np.ndarray, state_rot: np.ndarray, shape: ShapeSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Actor world pose × shape local pose -> world box (center, 3×3 with axes as columns, half sizes)."""
    center = state_p + state_rot @ shape.local_p
    axes = state_rot @ quat_to_matrix(shape.local_q)
    return center, axes, shape.half


def _radius(half: np.ndarray, axes: np.ndarray, n: np.ndarray) -> float:
    return float(np.sum(half * np.abs(axes.T @ n)))


def sat_gap(
    center_a: np.ndarray,
    axes_a: np.ndarray,
    half_a: np.ndarray,
    center_b: np.ndarray,
    axes_b: np.ndarray,
    half_b: np.ndarray,
) -> float:
    """Separating-axis criterion value ``g = max_n gap(n)`` of a single box pair.

    15 candidate axes: 3 face normals of each side + 9 edge cross products; degenerate axes with cross-product norm <= 1e-12 are skipped.
    ``g`` is the separating-axis criterion value, generally not equal to the shortest Euclidean distance; non-finite values are treated as unsafe (NaN returned).
    """
    delta = center_b - center_a
    best = -np.inf
    for k in range(3):
        for axis in (axes_a[:, k], axes_b[:, k]):
            gap = abs(float(np.dot(delta, axis))) - _radius(half_a, axes_a, axis) - _radius(half_b, axes_b, axis)
            if gap > best:
                best = gap
    for i in range(3):
        for j in range(3):
            axis = np.cross(axes_a[:, i], axes_b[:, j])
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                continue
            axis = axis / norm
            gap = abs(float(np.dot(delta, axis))) - _radius(half_a, axes_a, axis) - _radius(half_b, axes_b, axis)
            if gap > best:
                best = gap
    if not math.isfinite(best):
        return float("nan")
    return float(best)


def _classify(gap: float) -> str | None:
    """Translate a criterion value into a rejection reason; ``None`` means this pair is separated."""
    if not math.isfinite(gap):
        return "numerical_boundary"
    if gap < 0.0:
        return "contact"  # penetration
    if gap <= EPS_M:
        return "numerical_boundary"  # contact or within the numerical boundary band [0, ε], including exact tangency g == 0
    return None


def check_pair_static(
    obj_a: ObjectState,
    obj_b: ObjectState,
    *,
    stage: str = "initial",
    sweep_index: int | None = None,
) -> tuple[float, CollisionRejection | None]:
    """Static check over all box pairs of one object pair; returns the minimum criterion value and rejection evidence.

    ⚠ A conclusion is drawn only after all box pairs are traversed; the evidence points to the shape pair with the **smallest criterion value**, not the first
    one encountered in the traversal. Otherwise the same geometric state could report different ``reason`` depending on shape order:
    with two container centers 0.04 apart the minimum ``g = −0.01`` (front and back walls truly penetrate), but the central block and the front wall pair
    have exactly ``g = 0``; an early exit would report a true penetration as a numerical boundary.
    """
    rot_a = quat_to_matrix(obj_a.q)
    rot_b = quat_to_matrix(obj_b.q)
    worst = np.inf
    worst_pair: tuple[int, int] | None = None
    worst_finite = True
    for ia, shape_a in enumerate(obj_a.shapes):
        box_a = _world_box(obj_a.p, rot_a, shape_a)
        for ib, shape_b in enumerate(obj_b.shapes):
            box_b = _world_box(obj_b.p, rot_b, shape_b)
            gap = sat_gap(*box_a, *box_b)
            if not math.isfinite(gap):
                # non-finite criterion values are always treated as unsafe and take precedence over any finite value as evidence
                return float("nan"), CollisionRejection(
                    reason="numerical_boundary",
                    stage=stage,
                    object_a=obj_a.name,
                    object_b=obj_b.name,
                    shape_a=ia,
                    shape_b=ib,
                    gap_m=None,
                    sweep_index=sweep_index,
                    detail="separating-axis criterion value is non-finite",
                )
            if gap < worst:
                worst, worst_pair = gap, (ia, ib)
    if worst_pair is None:
        return float("nan"), CollisionRejection(
            reason="uncertified",
            stage=stage,
            object_a=obj_a.name,
            object_b=obj_b.name,
            shape_a=-1,
            shape_b=-1,
            sweep_index=sweep_index,
            detail="object has no collision shapes",
        )
    reason = _classify(worst)
    if reason is None:
        return float(worst), None
    return float(worst), CollisionRejection(
        reason=reason,
        stage=stage,
        object_a=obj_a.name,
        object_b=obj_b.name,
        shape_a=worst_pair[0],
        shape_b=worst_pair[1],
        gap_m=float(worst),
        sweep_index=sweep_index,
    )


def check_bin_layout(
    objects: Sequence[ObjectState],
    *,
    stage: str = "initial",
    raise_on_reject: bool = False,
    exhaustive: bool = False,
) -> tuple[float, CollisionRejection | None]:
    """Static check between all pairs of objects; if any pair is rejected, the whole layout is rejected.

    Returns ``(min criterion value, rejection evidence or None)``; the evidence points to the object and shape pair with the smallest criterion value,
    independent of traversal order. When ``raise_on_reject`` is true, raises :class:`BinCollisionError` instead,
    so runtime checkpoints can abort the sample directly.

    ⚠ **The returned minimum has different semantics in the two modes**:

    * ``exhaustive=False`` (default, used at runtime): bounding spheres first coarse-filter object pairs that are necessarily separated,
      replacing 36 SAT tests with one distance computation. The **verdict** (excluded or not) is completely unaffected -- skipped pairs are necessarily
      separated -- but the returned minimum is only "the minimum **among the pairs computed exactly**". Some skipped pair's real gap
      may be smaller, but it is equally safe.
    * ``exhaustive=True`` (used for reproduction checks): no coarse filter, every pair computed exactly, returning the true scene-wide minimum g.
      ``COLLISION_REPRODUCE`` compares step by step against saved ``sat_gap_m`` and must take this path.
    """
    worst = np.inf
    coarse_worst = np.inf
    worst_rejection: CollisionRejection | None = None
    # conservative bounding-sphere radius of each object: max distance from box vertices to the actor origin
    radii = [max(item.radii) for item in objects]
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            # coarse filter: when center distance minus the two radii already exceeds ε, all 36 box pairs of this pair are necessarily separated;
            # one distance computation replaces 36 SAT tests. Only pairs that necessarily pass are skipped; the criterion is unchanged.
            clearance = float(np.linalg.norm(objects[i].p - objects[j].p)) - radii[i] - radii[j]
            if not exhaustive and clearance > EPS_M:
                # same reasoning as check_swap_sweep: the bounding-sphere gap is a conservative lower bound of the real g
                # and must not be mixed into worst, otherwise "the minimum g of the most dangerous object pair" would be dragged down by a distant pair
                coarse_worst = min(coarse_worst, clearance)
                continue
            gap, rejection = check_pair_static(objects[i], objects[j], stage=stage)
            if rejection is not None and not math.isfinite(gap):
                if raise_on_reject:
                    raise BinCollisionError(rejection)
                return gap, rejection
            if gap < worst:
                worst, worst_rejection = gap, rejection
    if worst_rejection is not None:
        if raise_on_reject:
            raise BinCollisionError(worst_rejection)
        return float(worst), worst_rejection
    if math.isfinite(worst):
        return float(worst), None
    # all object pairs passed the coarse filter: fall back to the bounding-sphere lower bound, still "proven separated"
    return (float(coarse_worst) if math.isfinite(coarse_worst) else float("nan")), None


def check_bin_state(
    objects: Sequence[ObjectState],
    *,
    stage: str = "state",
    raise_on_reject: bool = False,
) -> tuple[float, CollisionRejection | None]:
    """Read-only re-check at a runtime moment, same criterion as :func:`check_bin_layout`, only the ``stage`` label differs."""
    return check_bin_layout(objects, stage=stage, raise_on_reject=raise_on_reject)


# -- Continuous criterion: interval-bisection proof ---------------------------------------------------
def _lane_endpoints(a_xy: np.ndarray, b_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the main direction and left normal of ``swap_flat_two_lane``; the normal is zero when start and end coincide."""
    delta = b_xy - a_xy
    normal = np.array([-delta[1], delta[0]], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm > COINCIDENT_XY:
        normal = normal / norm
    else:
        normal = np.zeros(2, dtype=np.float64)
    return delta, normal


def _quat_at(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """A's ``qa(s) = normalize((1−s)qa0 + s qb0)``; keeps the original sign and the normalized linear blend."""
    blended = (1.0 - s) * q0 + s * q1
    norm = float(np.linalg.norm(blended))
    if not math.isfinite(norm) or norm <= DEGENERATE_NORM:
        raise ValueError("linear quaternion blend degenerate")
    return blended / norm


def _min_blend_norm(q0: np.ndarray, dq: np.ndarray, lo: float, hi: float) -> float:
    """``m = min_{s∈[lo,hi]} ||q0 + s·Δq||``: closest distance from the segment to the origin, attained at an endpoint or the projection point."""
    dd = float(np.dot(dq, dq))
    candidates = [lo, hi]
    if dd > 0.0:
        star = -float(np.dot(q0, dq)) / dd
        if lo < star < hi:
            candidates.append(star)
    return min(float(np.linalg.norm(q0 + s * dq)) for s in candidates)


@dataclass
class _Mover:
    """Pose function of one object in a swap over parameter ``s`` and the quantities needed for the velocity bound."""

    name: str
    shapes: Sequence[ShapeSpec]
    radii: Sequence[float]
    xy0: np.ndarray
    z: float
    delta: np.ndarray  # main-direction displacement (B−A), used by this object according to sign
    normal: np.ndarray
    sign: float  # A takes +1, B takes −1
    q0: np.ndarray
    q1: np.ndarray

    def pose_at(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        offset = LANE_OFFSET * math.sin(math.pi * s)
        xy = self.xy0 + self.sign * (self.delta * s + self.normal * offset)
        return np.array([xy[0], xy[1], self.z], dtype=np.float64), _quat_at(self.q0, self.q1, s)

    def bounding_sphere(self) -> tuple[np.ndarray, float]:
        """Conservative bounding sphere of all box points of this object over the whole path.

        The actor origin trajectory ``xy0 + sign·(δ·s + n·0.07 sin(πs))`` lies within the sphere centered at the midpoint with
        radius ``0.5‖δ‖ + 0.07``; adding the max distance from box vertices to the origin suffices.
        """
        mid_xy = self.xy0 + self.sign * (self.delta * 0.5 + self.normal * LANE_OFFSET)
        center = np.array([mid_xy[0], mid_xy[1], self.z], dtype=np.float64)
        radius = 0.5 * float(np.linalg.norm(self.delta)) + LANE_OFFSET + max(self.radii)
        return center, radius

    def speed_bound(self, lo: float, hi: float, shape_index: int) -> float:
        """Linear velocity bound w.r.t. ``s`` within ``[lo,hi]`` of any point on this box."""
        translation = float(np.linalg.norm(self.delta)) + LANE_OFFSET * math.pi
        dq = self.q1 - self.q0
        m = _min_blend_norm(self.q0, dq, lo, hi)
        if not math.isfinite(m) or m <= DEGENERATE_NORM:
            raise ValueError("quaternion interpolation degenerate, cannot compute angular velocity bound")
        omega = 2.0 * float(np.linalg.norm(dq)) / m
        return translation + self.radii[shape_index] * omega


@dataclass
class _Static:
    """A bystander object that does not move during the swap; its velocity bound is always 0."""

    name: str
    shapes: Sequence[ShapeSpec]
    radii: Sequence[float]
    p: np.ndarray
    q: np.ndarray

    def pose_at(self, s: float) -> tuple[np.ndarray, np.ndarray]:  # noqa: ARG002
        return self.p, self.q

    def bounding_sphere(self) -> tuple[np.ndarray, float]:
        return self.p, max(self.radii)

    def speed_bound(self, lo: float, hi: float, shape_index: int) -> float:  # noqa: ARG002
        return 0.0


def _prove_pair(
    left: Any,
    right: Any,
    ia: int,
    ib: int,
    *,
    stage: str,
    sweep_index: int | None,
) -> tuple[float, CollisionRejection | None]:
    """Interval-bisection proof for one box pair; returns ``(observed min g, rejection evidence or None)``."""
    shape_a = left.shapes[ia]
    shape_b = right.shapes[ib]

    def gap_at(s: float) -> float:
        pa, qa = left.pose_at(s)
        pb, qb = right.pose_at(s)
        return sat_gap(
            *_world_box(pa, quat_to_matrix(qa), shape_a),
            *_world_box(pb, quat_to_matrix(qb), shape_b),
        )

    def reject(reason: str, s: float | None, gap: float | None, lo: float, hi: float, depth: int, used: int, detail: str) -> CollisionRejection:
        return CollisionRejection(
            reason=reason,
            stage=stage,
            object_a=left.name,
            object_b=right.name,
            shape_a=ia,
            shape_b=ib,
            gap_m=gap if gap is not None and math.isfinite(gap) else None,
            s=s,
            sweep_index=sweep_index,
            interval_low=lo,
            interval_high=hi,
            depth=depth,
            intervals_used=used,
            detail=detail,
        )

    worst = np.inf
    used = 0

    # check both endpoints first: passing endpoints does not allow release, but a failing endpoint can be rejected immediately.
    for s in (0.0, 1.0):
        try:
            gap = gap_at(s)
        except ValueError as exc:
            return float("nan"), reject("uncertified", s, None, 0.0, 1.0, 0, used, str(exc))
        used += 1
        if math.isfinite(gap):
            worst = min(worst, gap)
        reason = _classify(gap)
        if reason is not None:
            return gap, reject(reason, s, gap, s, s, 0, used, "endpoint check")

    # start from the midpoint of the root interval; left subinterval before right.
    stack: list[tuple[float, float, int]] = [(0.0, 1.0, 0)]
    while stack:
        lo, hi, depth = stack.pop()
        if used >= MAX_INTERVALS:
            return (
                float(worst) if math.isfinite(worst) else float("nan"),
                reject("uncertified", None, None, lo, hi, depth, used, f"interval count exceeds limit {MAX_INTERVALS}"),
            )
        if depth > MAX_DEPTH:
            return (
                float(worst) if math.isfinite(worst) else float("nan"),
                reject("uncertified", None, None, lo, hi, depth, used, f"bisection depth exceeds limit {MAX_DEPTH}"),
            )
        mid = 0.5 * (lo + hi)
        try:
            gap = gap_at(mid)
            bound_a = left.speed_bound(lo, hi, ia)
            bound_b = right.speed_bound(lo, hi, ib)
        except ValueError as exc:
            return (
                float(worst) if math.isfinite(worst) else float("nan"),
                reject("uncertified", mid, None, lo, hi, depth, used, str(exc)),
            )
        used += 1
        if math.isfinite(gap):
            worst = min(worst, gap)
        reason = _classify(gap)
        if reason is not None:
            return gap, reject(reason, mid, gap, lo, hi, depth, used, "midpoint check")
        margin = (bound_a + bound_b) * (hi - lo) * 0.5
        if not math.isfinite(margin):
            return (
                float(worst) if math.isfinite(worst) else float("nan"),
                reject("uncertified", mid, gap, lo, hi, depth, used, "velocity bound non-finite"),
            )
        if gap > EPS_M + margin:
            continue  # the whole interval is proven separated
        # push the right subinterval onto the stack in LIFO order so the left subinterval is proven first.
        stack.append((mid, hi, depth + 1))
        stack.append((lo, mid, depth + 1))

    return (float(worst) if math.isfinite(worst) else float("nan")), None


def check_swap_sweep(
    moving_a: ObjectState,
    moving_b: ObjectState,
    bystanders: Sequence[ObjectState] = (),
    *,
    sweep_index: int | None = None,
    stage: str = "sweep",
    raise_on_reject: bool = False,
) -> tuple[float, CollisionRejection | None]:
    """Continuous check over the whole swap path: between the two swappers, and between each swapper and every bystander.

    The path is recomputed with the actual semantics of ``swap_flat_two_lane``: ``A`` goes along the left-normal curve ``+0.07 sin(πs)``
    to ``B``'s position, ``B`` goes along the ``−`` curve to ``A``'s position, and the quaternions are exchanged with a normalized linear blend.
    Bystanders are held still frame by frame by the original function during the whole segment, so their velocity bound is 0.

    ⚠ No frame sampling: every box pair must be proven separated over the whole segment by interval bisection; anything unprovable is rejected as ``uncertified``.
    ⚠ Unlike :func:`check_bin_layout`, this function returns at the first disproven box pair -- the number of box pairs is
    dozens of times that of the static check and each pair needs bisection, so running all of them just to pick the "most severe" pair is not worth it. Hence the returned
    evidence is "the first disproven box pair", not necessarily the one with the smallest criterion value.
    """
    a_xy = moving_a.p[:2]
    b_xy = moving_b.p[:2]
    delta, normal = _lane_endpoints(a_xy, b_xy)

    mover_a = _Mover(
        name=moving_a.name,
        shapes=moving_a.shapes,
        radii=moving_a.radii,
        xy0=a_xy.copy(),
        z=float(moving_a.p[2]),
        delta=delta,
        normal=normal,
        sign=1.0,
        q0=moving_a.q.copy(),
        q1=moving_b.q.copy(),
    )
    mover_b = _Mover(
        name=moving_b.name,
        shapes=moving_b.shapes,
        radii=moving_b.radii,
        xy0=b_xy.copy(),
        z=float(moving_b.p[2]),
        delta=delta,
        normal=normal,
        sign=-1.0,
        q0=moving_b.q.copy(),
        q1=moving_a.q.copy(),
    )
    statics = [
        _Static(name=item.name, shapes=item.shapes, radii=item.radii, p=item.p.copy(), q=item.q.copy())
        for item in bystanders
    ]

    pairs: list[tuple[Any, Any]] = [(mover_a, mover_b)]
    for item in statics:
        pairs.append((mover_a, item))
        pairs.append((mover_b, item))

    worst = np.inf
    coarse_worst = np.inf
    for left, right in pairs:
        # coarse filter: when the two conservative bounding spheres are separated, all 36 box pairs of this pair are necessarily separated over the whole segment; bisection is skipped.
        # only pairs that necessarily pass are skipped and the criterion itself is unchanged; most bystander containers pass at this step, greatly reducing bisections.
        # ⚠ skipped pairs **do not enter worst**: the bounding-sphere gap is a conservative lower bound of the real g; mixing it in would push
        # "the minimum g of the most dangerous object pair" down to the lower bound of a distant pair, misleading both reports and figures.
        center_l, radius_l = left.bounding_sphere()
        center_r, radius_r = right.bounding_sphere()
        clearance = float(np.linalg.norm(center_l - center_r)) - radius_l - radius_r
        if clearance > EPS_M:
            coarse_worst = min(coarse_worst, clearance)
            continue
        for ia in range(len(left.shapes)):
            for ib in range(len(right.shapes)):
                gap, rejection = _prove_pair(left, right, ia, ib, stage=stage, sweep_index=sweep_index)
                if rejection is not None:
                    if raise_on_reject:
                        raise BinCollisionError(rejection)
                    return gap, rejection
                if math.isfinite(gap):
                    worst = min(worst, gap)
    if math.isfinite(worst):
        return float(worst), None
    # all object pairs passed the coarse filter: no exact values, fall back to the bounding-sphere lower bound, still "proven separated"
    return (float(coarse_worst) if math.isfinite(coarse_worst) else float("nan")), None


# -- V5: joint continuous criterion and certified prefilter for multiple simultaneous swaps (additions only, nothing changed) ---------------------
#: Number of equally spaced sample points of the certified prefilter on s∈[0,1] (V5 plan 2.5 / L23).
PREFILTER_SAMPLES = 401
#: Release margin of the certified prefilter, meters. The "vertical cylinder gap lower bound" proven by the prefilter must exceed it to skip the exact proof.
#: 1 mm rather than ``EPS_M``: leaves a three-orders-of-magnitude buffer between skipped pairs and pairs "the exact proof would certainly prove",
#: pushing the theoretical divergence "truly separated but the separating-axis value falls in the numerical boundary band, or bisection is exhausted" into an impossible region (see
#: the notes of :func:`check_multi_swap_sweep`). It only decides whether to skip the exact proof and takes no part in any rejection.
PREFILTER_MARGIN_M = 1e-3


def _swap_movers(moving_a: ObjectState, moving_b: ObjectState) -> tuple[_Mover, _Mover]:
    """Build a pair of swappers exactly as :func:`check_swap_sweep` does (field-by-field identical, guaranteeing bitwise identity for a single pair)."""
    a_xy = moving_a.p[:2]
    b_xy = moving_b.p[:2]
    delta, normal = _lane_endpoints(a_xy, b_xy)
    mover_a = _Mover(
        name=moving_a.name,
        shapes=moving_a.shapes,
        radii=moving_a.radii,
        xy0=a_xy.copy(),
        z=float(moving_a.p[2]),
        delta=delta,
        normal=normal,
        sign=1.0,
        q0=moving_a.q.copy(),
        q1=moving_b.q.copy(),
    )
    mover_b = _Mover(
        name=moving_b.name,
        shapes=moving_b.shapes,
        radii=moving_b.radii,
        xy0=b_xy.copy(),
        z=float(moving_b.p[2]),
        delta=delta,
        normal=normal,
        sign=-1.0,
        q0=moving_b.q.copy(),
        q1=moving_a.q.copy(),
    )
    return mover_a, mover_b


def _quat_to_matrix_batch(q: np.ndarray) -> np.ndarray:
    """Batch-convert ``(N,4)`` wxyz quaternions into ``(N,3,3)`` rotation matrices, same formula as :func:`quat_to_matrix`."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - z * w)
    out[:, 0, 2] = 2 * (x * z + y * w)
    out[:, 1, 0] = 2 * (x * y + z * w)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - x * w)
    out[:, 2, 0] = 2 * (x * z - y * w)
    out[:, 2, 1] = 2 * (y * z + x * w)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def _local_vertices(shapes: Sequence[ShapeSpec]) -> np.ndarray:
    """Positions of the 8 vertices of all boxes in the actor's local frame, ``(8·number of shapes, 3)``."""
    signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    blocks = []
    for shape in shapes:
        rot = quat_to_matrix(shape.local_q)
        blocks.append(shape.local_p[None, :] + (signs * shape.half[None, :]) @ rot.T)
    return np.concatenate(blocks, axis=0)


@dataclass
class _PrefilterTrack:
    """Vertical bounding cylinder of an object at the prefilter sample points: origin XY trajectory, cylinder radius, and their combined Lipschitz constant.

    ``xy[i]`` and ``rho[i]`` are the actor origin XY at ``s = PREFILTER_S[i]`` and "the max horizontal distance from all box vertices to the origin's vertical
    axis". The object lies entirely within the infinitely tall vertical cylinder with axis ``xy[i]`` and radius ``rho[i]`` (the convex hull is spanned by the vertices).
    ``lipschitz`` is the sum of the rate-of-change bounds of ``‖xy(s)‖`` and ``rho(s)`` w.r.t. ``s``: the origin translation speed bound
    ``‖δ‖ + 0.07π`` plus the rotation-induced vertex speed bound ``max(radii)·2‖Δq‖/m`` (same formula as ``_Mover.speed_bound``,
    with ``m`` the minimum blend norm over the whole ``[0,1]``). Zero for static objects.
    """

    xy: np.ndarray  # (N, 2)
    rho: np.ndarray  # (N,)
    lipschitz: float


def _prefilter_samples() -> np.ndarray:
    return np.linspace(0.0, 1.0, PREFILTER_SAMPLES)


def _prefilter_track(obj: Any, samples: np.ndarray) -> _PrefilterTrack | None:
    """Build the prefilter cylinder for a swapper or static object; returns ``None`` when the quaternion blend is degenerate (no angular velocity bound), and that object does not take part in prefiltering."""
    vertices = _local_vertices(obj.shapes)
    if isinstance(obj, _Static):
        rot = quat_to_matrix(obj.q)
        rho = float(np.max(np.linalg.norm((vertices @ rot.T)[:, :2], axis=1)))
        n = samples.shape[0]
        return _PrefilterTrack(
            xy=np.repeat(np.asarray(obj.p, dtype=np.float64)[None, :2], n, axis=0),
            rho=np.full(n, rho, dtype=np.float64),
            lipschitz=0.0,
        )
    if not isinstance(obj, _Mover):
        raise TypeError(f"object type not recognized by the prefilter: {type(obj).__name__}")
    dq = obj.q1 - obj.q0
    m = _min_blend_norm(obj.q0, dq, 0.0, 1.0)
    if not math.isfinite(m) or m <= DEGENERATE_NORM:
        # degenerate objects are left to the exact proof to report uncertified; the prefilter never releases them
        return None
    s = samples[:, None]
    offset = LANE_OFFSET * np.sin(np.pi * s)
    xy = obj.xy0[None, :] + obj.sign * (obj.delta[None, :] * s + obj.normal[None, :] * offset)
    blended = (1.0 - s) * obj.q0[None, :] + s * obj.q1[None, :]
    blended = blended / np.linalg.norm(blended, axis=1, keepdims=True)
    rots = _quat_to_matrix_batch(blended)
    horizontal = np.einsum("nij,vj->nvi", rots[:, :2, :], vertices)
    rho = np.max(np.linalg.norm(horizontal, axis=2), axis=1)
    translation = float(np.linalg.norm(obj.delta)) + LANE_OFFSET * math.pi
    omega = 2.0 * float(np.linalg.norm(dq)) / m
    lipschitz = translation + max(obj.radii) * omega
    if not (np.all(np.isfinite(xy)) and np.all(np.isfinite(rho)) and math.isfinite(lipschitz)):
        return None
    return _PrefilterTrack(xy=xy, rho=rho, lipschitz=float(lipschitz))


def _prefilter_clearance(left: _PrefilterTrack, right: _PrefilterTrack, step: float) -> float:
    """Lower bound on the horizontal gap between two vertical cylinders over the whole ``s∈[0,1]``.

    ``G(s) = ‖xy_l(s) − xy_r(s)‖ − rho_l(s) − rho_r(s)`` is ``L = L_l + L_r`` -Lipschitz; between two adjacent sample points
    ``[s_i, s_i + h]``, ``G(s) >= max(G_i − L(s−s_i), G_{i+1} − L(s_{i+1}−s)) >= (G_i + G_{i+1})/2 − L·h/2``.
    Taking the minimum over all subintervals gives the whole-segment lower bound; a positive bound proves the two cylinders (hence the two objects) separated over the whole segment.
    """
    gaps = np.linalg.norm(left.xy - right.xy, axis=1) - left.rho - right.rho
    if gaps.shape[0] < 2:
        return float("nan")
    worst_mid = float(np.min(0.5 * (gaps[:-1] + gaps[1:])))
    bound = worst_mid - (left.lipschitz + right.lipschitz) * step * 0.5
    return bound if math.isfinite(bound) else float("nan")


def check_multi_swap_sweep(
    pairs: Sequence[tuple[ObjectState, ObjectState]],
    bystanders: Sequence[ObjectState] = (),
    *,
    sweep_index: int | None = None,
    stage: str = "sweep",
    raise_on_reject: bool = False,
    prefilter: bool = True,
    stats: dict[str, int] | None = None,
) -> tuple[float, CollisionRejection | None]:
    """Continuous check for multiple pairs swapping simultaneously in the same window (V5 plan 2.5: outer ring swaps in sync with the inner ring; 2.15: button base as a static obstacle).

    Each item ``(A, B)`` in ``pairs`` swaps per ``swap_flat_two_lane(lane_offset=0.07, smooth=True)`` within **the same window**:
    on the same control step all pairs have the same progress ``α`` (after smoothstep), so all poses are functions of the same parameter ``s``,
    and :func:`_prove_pair` can be used directly (it can already prove two moving objects). ``bystanders`` are objects static over the whole segment,
    which can be containers, cubes, or arbitrary oriented boxes built by :func:`static_box_state` / :func:`static_rect_state` /
    :func:`static_state_from_obb2d` / :func:`button_base_state`.

    Object pairs checked and their order:

    1. each pair itself (the two swappers), in ``pairs`` order;
    2. swappers across pairs, pairwise (swappers sorted as ``A1, B1, A2, B2, …``, taking ``i < j`` from different pairs);
    3. each static object in turn against all swappers (static object first, then swapper, consistent with the bystander order of :func:`check_swap_sweep`).

    **With a single pair, the object pairs checked, their order, the coarse filter and every box-pair proof are exactly the same as** :func:`check_swap_sweep`;
    with ``prefilter=False`` the return value is bitwise identical (verified with many random cases in unit tests).

    With ``prefilter=True`` (default, enabled as decided in L23), an extra certified prefilter layer runs after the bounding-sphere coarse filter and before the exact proof: for each object
    take ``PREFILTER_SAMPLES = 401`` equally spaced points in ``s``, compute the horizontal gap of the vertical bounding cylinders, and use the Lipschitz bound to fill the gaps between samples
    into a whole-segment lower bound (:func:`_prefilter_clearance`); object pairs whose lower bound exceeds ``PREFILTER_MARGIN_M`` are **already proven separated over the whole
    segment**, and the per-box-pair bisection proof is skipped. The prefilter **only skips, never rejects**: rejections always come from :func:`_prove_pair`, given by the first
    disproven box pair met in the same object-pair order, so the rejection evidence is bitwise identical to running without the prefilter.

    ⚠ For the same reason as the bounding-sphere coarse filter of :func:`check_swap_sweep`, skipped pairs **do not enter the returned min g**; on pass the return value is
    "the min g among the box pairs computed exactly", falling back to the minimum of the coarse-filter / prefilter lower bounds when all object pairs are skipped. The prefilter switch may therefore
    change the number returned **on pass**, but not the verdict or the rejection evidence.
    ⚠ The only theoretically possible divergence: a pair whose real gap > 1 mm but the exact proof cannot prove it because the separating-axis value <= ε or bisection is exhausted --
    this is the same premise as the original function's bounding-sphere coarse filter (pairs skipped by the coarse filter are not exactly proven either); the 1 mm margin keeps it from
    occurring with actual geometry. Swappers with a degenerate quaternion blend do not take part in prefiltering and are still left to the exact proof to report ``uncertified``.

    When a dict is passed as ``stats``, counts are accumulated in place: ``object_pairs`` (object pairs examined), ``coarse_skipped`` (skipped by the bounding-sphere
    coarse filter), ``prefilter_skipped`` (skipped by the certified prefilter), ``proved_object_pairs`` (object pairs entering the exact proof),
    ``proved_shape_pairs`` (box pairs actually proven or disproven).

    Returns ``(min criterion value, rejection evidence or None)``, same structure as :func:`check_swap_sweep`; when ``raise_on_reject`` is true
    raises :class:`BinCollisionError` instead. The same object (by ``name``) must not appear in two places, otherwise ``ValueError`` is raised.
    """
    pair_list = [tuple(item) for item in pairs]
    if not pair_list:
        raise ValueError("check_multi_swap_sweep needs at least one swap pair")
    names: list[str] = []
    for item in pair_list:
        if len(item) != 2:
            raise ValueError(f"each pair must have exactly two objects, got {len(item)} objects")
        names.extend(obj.name for obj in item)
    names.extend(obj.name for obj in bystanders)
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"duplicate object names (one object cannot be in two pairs, or both swapping and static): {duplicated}")

    movers: list[_Mover] = []
    pair_of: list[int] = []
    for index, (moving_a, moving_b) in enumerate(pair_list):
        mover_a, mover_b = _swap_movers(moving_a, moving_b)
        movers.extend((mover_a, mover_b))
        pair_of.extend((index, index))
    statics = [
        _Static(name=item.name, shapes=item.shapes, radii=item.radii, p=item.p.copy(), q=item.q.copy())
        for item in bystanders
    ]

    checks: list[tuple[Any, Any]] = [(movers[2 * k], movers[2 * k + 1]) for k in range(len(pair_list))]
    for i in range(len(movers)):
        for j in range(i + 1, len(movers)):
            if pair_of[i] != pair_of[j]:
                checks.append((movers[i], movers[j]))
    for item in statics:
        for mover in movers:
            checks.append((mover, item))

    counts = {
        "object_pairs": 0,
        "coarse_skipped": 0,
        "prefilter_skipped": 0,
        "proved_object_pairs": 0,
        "proved_shape_pairs": 0,
    }
    samples = _prefilter_samples() if prefilter else None
    step = float(samples[1] - samples[0]) if samples is not None else 0.0
    tracks: dict[int, _PrefilterTrack | None] = {}

    def track_of(obj: Any) -> _PrefilterTrack | None:
        key = id(obj)
        if key not in tracks:
            tracks[key] = _prefilter_track(obj, samples)
        return tracks[key]

    def flush_stats() -> None:
        if stats is not None:
            for key, value in counts.items():
                stats[key] = stats.get(key, 0) + value

    worst = np.inf
    coarse_worst = np.inf
    for left, right in checks:
        counts["object_pairs"] += 1
        # layer 1: the whole-segment bounding-sphere coarse filter, exactly the same as check_swap_sweep
        center_l, radius_l = left.bounding_sphere()
        center_r, radius_r = right.bounding_sphere()
        clearance = float(np.linalg.norm(center_l - center_r)) - radius_l - radius_r
        if clearance > EPS_M:
            coarse_worst = min(coarse_worst, clearance)
            counts["coarse_skipped"] += 1
            continue
        # layer 2: certified prefilter (only skips pairs proven separated)
        if samples is not None:
            track_l = track_of(left)
            track_r = track_of(right)
            if track_l is not None and track_r is not None:
                bound = _prefilter_clearance(track_l, track_r, step)
                if math.isfinite(bound) and bound > PREFILTER_MARGIN_M:
                    coarse_worst = min(coarse_worst, bound)
                    counts["prefilter_skipped"] += 1
                    continue
        # layer 3: per-box-pair interval-bisection proof, the same _prove_pair as check_swap_sweep
        counts["proved_object_pairs"] += 1
        for ia in range(len(left.shapes)):
            for ib in range(len(right.shapes)):
                gap, rejection = _prove_pair(left, right, ia, ib, stage=stage, sweep_index=sweep_index)
                counts["proved_shape_pairs"] += 1
                if rejection is not None:
                    flush_stats()
                    if raise_on_reject:
                        raise BinCollisionError(rejection)
                    return gap, rejection
                if math.isfinite(gap):
                    worst = min(worst, gap)
    flush_stats()
    if math.isfinite(worst):
        return float(worst), None
    return (float(coarse_worst) if math.isfinite(coarse_worst) else float("nan")), None


def check_swap_sweep_prefiltered(
    moving_a: ObjectState,
    moving_b: ObjectState,
    bystanders: Sequence[ObjectState] = (),
    *,
    sweep_index: int | None = None,
    stage: str = "sweep",
    raise_on_reject: bool = False,
    prefilter: bool = True,
    stats: dict[str, int] | None = None,
) -> tuple[float, CollisionRejection | None]:
    """Prefiltered wrapper for a single swap pair: equals ``check_multi_swap_sweep([(moving_a, moving_b)], bystanders, ...)``.

    :func:`check_swap_sweep` itself stays as-is and never prefilters; to speed up single-pair checks (e.g. the inner-vs-inner
    prejudgment at reset of the two Swap envs, VideoRepick's partner feasibility filter), call this function instead. With ``prefilter=False`` it is
    bitwise identical to :func:`check_swap_sweep`; with ``prefilter=True`` the verdict and rejection evidence are the same, only "the min g on pass" may differ.
    """
    return check_multi_swap_sweep(
        [(moving_a, moving_b)],
        bystanders,
        sweep_index=sweep_index,
        stage=stage,
        raise_on_reject=raise_on_reject,
        prefilter=prefilter,
        stats=stats,
    )


# -- V5: building static obstacles (arbitrary oriented boxes / rectangles) ------------------------------------
def static_box_state(
    name: str,
    center: Sequence[float],
    half_size: Sequence[float],
    *,
    yaw_rad: float = 0.0,
) -> ObjectState:
    """A solid box rotated by ``yaw_rad`` about z as a static object: world center ``center (3,)``, half sizes ``half_size (3,)``.

    The actor origin is placed at the box center with identity local pose, so both the bounding sphere and the prefilter cylinder are tight.
    """
    center_arr = np.asarray(center, dtype=np.float64).reshape(3)
    half_arr = np.asarray(half_size, dtype=np.float64).reshape(3)
    if not (np.all(np.isfinite(center_arr)) and np.all(np.isfinite(half_arr))) or np.any(half_arr <= 0.0):
        raise ValueError(f"invalid static box parameters: center={center_arr.tolist()} half={half_arr.tolist()}")
    shape = ShapeSpec(np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]), half_arr)
    return ObjectState(name=name, p=center_arr, q=euler_xyz_to_quat([0.0, 0.0, float(yaw_rad)]), shapes=(shape,))


def static_rect_state(
    name: str,
    center_xy: Sequence[float],
    half_xy: Sequence[float],
    *,
    yaw_rad: float = 0.0,
    z_range: tuple[float, float],
) -> ObjectState:
    """A static box extruded along z from an oriented rectangle on the table: XY center, XY half sizes, orientation about z, plus an explicitly required
    ``z_range = (z_low, z_high)`` (meters). ``z_range`` deliberately has no default: how tall an obstacle is directly decides whether it blocks an object,
    and a wrong height would silently let things through, so the caller must fill it per actual geometry (e.g. "only block things on the floor" can use ``(0.0, 1.0)``).
    """
    z_low, z_high = (float(v) for v in z_range)
    if not (math.isfinite(z_low) and math.isfinite(z_high)) or z_high <= z_low:
        raise ValueError(f"invalid z_range: {z_range}")
    cx, cy = (float(v) for v in center_xy)
    hx, hy = (float(v) for v in half_xy)
    return static_box_state(
        name,
        [cx, cy, 0.5 * (z_low + z_high)],
        [hx, hy, 0.5 * (z_high - z_low)],
        yaw_rad=yaw_rad,
    )


def static_state_from_obb2d(
    name: str,
    obb2d: tuple[Sequence[float], Any, Sequence[float]],
    *,
    z_range: tuple[float, float],
    tol: float = 1e-6,
) -> ObjectState:
    """Convert the 2D OBB triple ``(c (2,), A (2×2, axes as columns), h (2,))`` used by the repository's placement logic into a static box.

    The triple format matches the return values of ``object_generation._trimesh_box_to_obb2d`` / ``create_button_obb``.
    The two columns of ``A`` must be orthonormal (within ``tol``), otherwise ``ValueError`` is raised -- non-orthogonal axes projected from tilted objects
    cannot be treated as a rectangle. The second column may form a left-handed system with the first (that is the projection of a container flipped 180° upside down); the rectangle is symmetric about its axes,
    so only the first column determines orientation.
    """
    c, axes, half = obb2d
    axes_arr = np.asarray(axes, dtype=np.float64).reshape(2, 2)
    gram = axes_arr.T @ axes_arr
    if not np.all(np.isfinite(gram)) or float(np.max(np.abs(gram - np.eye(2)))) > tol:
        raise ValueError(f"2D OBB axes are not orthonormal: A={axes_arr.tolist()}")
    yaw = math.atan2(float(axes_arr[1, 0]), float(axes_arr[0, 0]))
    return static_rect_state(name, c, half, yaw_rad=yaw, z_range=z_range)


def button_base_state(
    name: str,
    center_xy: Sequence[float],
    *,
    scale: float = 1.0,
    base_half: Sequence[float] = (0.025, 0.025, 0.005),
) -> ObjectState:
    """Replicate the collision box of the button **base** of ``object_generation.build_button``: half sizes ``base_half × scale``,
    axis-aligned, bottom face on the table (center z = half height).

    Only the base, not the cylindrical button cap on top (the collision model only has boxes); V5 plan L54 requires the base OBB to be a static obstacle
    for swap sweeps. ``center_xy`` must be the center finally used by ``build_button`` (after random offset and spec replay injection).
    """
    half = np.asarray(base_half, dtype=np.float64).reshape(3) * float(scale)
    return static_box_state(name, [float(center_xy[0]), float(center_xy[1]), float(half[2])], half)


def nearest_partner_index(reference: Sequence[float], candidates: Sequence[tuple[int, Sequence[float]]]) -> tuple[int, list[tuple[int, float]]]:
    """Scan the nearest neighbor with the original semantics of ``step``; returns ``(selected index, distance table of all candidates)``.

    Replicates two details of the original implementation: the check uses ``dist < closest_dist`` **strictly less**, and candidates are traversed in the given order
    (i.e. generation order), so on ties the earlier one wins -- exactly ``tie_break=first_in_spawn_order``.
    The distance table is returned as-is to keep evidence on mismatch: borderline ties and genuine mis-bindings must be distinguishable.
    """
    best_index = -1
    best_dist = float("inf")
    table: list[tuple[int, float]] = []
    origin = np.asarray(reference, dtype=np.float64)[:2]
    for index, position in candidates:
        dist = float(np.linalg.norm(origin - np.asarray(position, dtype=np.float64)[:2]))
        table.append((index, dist))
        if dist < best_dist:
            best_dist = dist
            best_index = index
    return best_index, table


# -- Runtime: reading boxes from real actors ----------------------------------------------
def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def shape_specs_from_actor(actor: Any) -> tuple[ShapeSpec, ...]:
    """Read the ``half_size`` and ``local_pose`` of every ``PhysxCollisionShapeBox`` on the actor.

    Shape order is kept as returned by the engine, neither sorted nor merged; non-box shapes raise immediately,
    not replaced by an overall AABB -- that can only pre-screen and cannot serve as the criterion.
    """
    entity = getattr(actor, "_objs", None)
    if entity:
        entity = entity[0]
    else:
        entity = actor
    components = None
    for attr in ("find_component_by_type", "components"):
        if hasattr(entity, attr):
            break
    try:
        from sapien.physx import PhysxRigidBaseComponent  # type: ignore

        component = entity.find_component_by_type(PhysxRigidBaseComponent)
        components = component.get_collision_shapes()
    except Exception as exc:  # pragma: no cover - only triggered without SAPIEN
        raise BinCollisionError(
            CollisionRejection(
                reason="uncertified",
                stage="geometry",
                object_a=str(getattr(actor, "name", actor)),
                object_b="-",
                shape_a=-1,
                shape_b=-1,
                detail=f"cannot read collision shapes: {exc}",
            )
        ) from exc

    specs: list[ShapeSpec] = []
    for index, shape in enumerate(components):
        half = getattr(shape, "half_size", None)
        if half is None:
            raise BinCollisionError(
                CollisionRejection(
                    reason="uncertified",
                    stage="geometry",
                    object_a=str(getattr(actor, "name", actor)),
                    object_b="-",
                    shape_a=index,
                    shape_b=-1,
                    detail=f"shape {index} is not a box: {type(shape).__name__}",
                )
            )
        pose = shape.local_pose
        specs.append(
            ShapeSpec(
                np.asarray(pose.p, dtype=np.float64).reshape(3),
                np.asarray(pose.q, dtype=np.float64).reshape(4),
                np.asarray(half, dtype=np.float64).reshape(3),
            )
        )
    return tuple(specs)


def object_state_from_actor(actor: Any, name: str | None = None) -> ObjectState:
    """Pack a real actor's current world pose and real boxes into :class:`ObjectState`."""
    pose = actor.pose if hasattr(actor, "pose") else actor.get_pose()
    p = _to_numpy(pose.p)[:3]
    q = _to_numpy(pose.q)[:4]
    return ObjectState(
        name=name or str(getattr(actor, "name", "actor")),
        p=p,
        q=q,
        shapes=shape_specs_from_actor(actor),
    )


def object_states_from_actors(actors_map: Iterable[tuple[str, Any]]) -> list[ObjectState]:
    return [object_state_from_actor(actor, name) for name, actor in actors_map]
