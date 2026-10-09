"""Shared constants and sampling tools for the V4 xhard tier (NEWTASK_RELEASE_V4_PLAN 2.0 (1)).

Only read in xhard branches; the original three tiers never import anything from this module for sampling, so V0/V1 are unaffected.

* ``DISTRACTOR_COLORS``: global color pool of "other color" distractors (B2: yellow / cyan / magenta, shared by six envs, A5 fixes 3).
* ``corner_push``: corner-bias mapping. Pushes a uniform sample on ``[0,1]`` toward both ends according to ``corner_bias``;
  with ``corner_bias=0`` it **returns the same object unchanged** (callers rely on this to stay byte-equivalent to the existing uniform sampling).
* ``cube_obb2d_exact`` (V5 2.0 (1)): builds a prefab 2D obstacle ``(c, A, h)`` from the cube's true yaw, replacing
  the actor path through ``object_generation._trimesh_box_to_obb2d``, which degenerates to a line segment on a cube.
"""

from __future__ import annotations

import colorsys
import math

import numpy as np

# Fixed order: yellow, cyan, magenta. Names are used in task text and spec records; rgba is byte-identical to decision B2.
DISTRACTOR_COLORS = (
    {"name": "yellow", "rgba": (1, 1, 0, 1)},
    {"name": "cyan", "rgba": (0, 1, 1, 1)},
    {"name": "magenta", "rgba": (1, 0, 1, 1)},
)

# V7 fixed-value table (0928 proposal §3.2.2, R11): distractor cubes 1/2/3/4 of PickXtimes / SwingXtimes need a 4th distractor color.
# Only used by these two envs; the three-color pool above is also the pool of the four Unmask / Swap tasks (the sampler requires byte equality), untouched.
# The 4th color must be distinguishable from the red / blue / green target colors and the three-color pool; chosen by visual inspection of stage-3 figures (proposal §1 item 9 C1).
BLOCK_DISTRACTOR_COLORS = DISTRACTOR_COLORS + (
    {"name": "orange", "rgba": (1, 0.5, 0, 1)},
)

# Gamut for "arbitrary cube color" (user 2026-09-22: "set saturation / value lower bounds"): any hue,
# saturation >= 0.5, value >= 0.4, excluding near-white (confusable with the white highlight disk), near-black and near-gray.
# Still draws only 3 uniform numbers in [0,1) (same number of random calls as the original uniform RGB draw), then maps deterministically.
HSV_FLOOR_COLOR = {"h_range": [0.0, 1.0], "s_range": [0.5, 1.0], "v_range": [0.4, 1.0]}


def hsv_floor_rgb(u, cfg=None):
    """3 uniform numbers in [0,1) -> RGB within the restricted gamut (float triple)."""
    cfg = HSV_FLOOR_COLOR if cfg is None else cfg
    (h0, h1), (s0, s1), (v0, v1) = cfg["h_range"], cfg["s_range"], cfg["v_range"]
    h = h0 + float(u[0]) * (h1 - h0)
    s = s0 + float(u[1]) * (s1 - s0)
    v = v0 + float(u[2]) * (v1 - v0)
    return list(colorsys.hsv_to_rgb(h % 1.0, s, v))


# Exponent 1/(1+CORNER_GAIN) at corner_bias=1; 4 ⇒ exponent 0.2, t=0.5 is pushed to about 0.87.
CORNER_GAIN = 4.0


def corner_push(u, corner_bias):
    """Push a single ``u ∈ [0,1]`` toward the 0 or 1 end according to the corner bias.

    Mapping: ``t = 2u-1``, ``t' = sign(t)·|t|^p``, ``p = 1/(1+CORNER_GAIN·b)``, returns ``(t'+1)/2``.
    Monotone, endpoint-preserving, symmetric about 0.5; the two coordinates are pushed independently ⇒ the joint distribution is biased toward the four corners.
    ``b`` must be in ``[0,1]``; when ``b == 0`` no floating-point operation is done and ``u`` is returned directly.
    """
    b = float(corner_bias)
    if b == 0.0:
        return u
    if not 0.0 <= b <= 1.0:
        raise ValueError(f"corner_bias must be in [0,1], got {corner_bias}")
    t = 2.0 * float(u) - 1.0
    p = 1.0 / (1.0 + CORNER_GAIN * b)
    mag = abs(t) ** p
    return (1.0 + (mag if t >= 0 else -mag)) / 2.0


def _as_numpy(value):
    """torch tensor / sapien array / list -> float64 numpy array (does not import torch; duck-typed)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _xy_yaw_from_pose(pose):
    """Get center xy and yaw from a pose (``.p`` is xyz, ``.q`` is a wxyz quaternion; a batch dim ``[1, ·]`` is allowed).

    yaw is the in-table heading of "the most horizontal of the three body axes": under any rotation the squared z components of the three axes sum to 1,
    so the most horizontal one has |z| <= 1/√3 and its xy projection length >= √(2/3), hence it **never degenerates**; the four side faces of a cube are equivalent,
    so any horizontal axis yields the same square. For an upright cube (rotated only about z) column 0 is the body x axis and yaw matches the yaw used to build the cube (mod 2π).
    """
    p = _as_numpy(pose.p).reshape(-1)[:3]
    w, x, y, z = _as_numpy(pose.q).reshape(-1)[:4]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("cube_obb2d_exact: pose quaternion is zero")
    w, x, y, z = w / n, x / n, y / n, z / n
    # the three columns of the rotation matrix (directions of body x/y/z axes in the world frame)
    cols = (
        (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
        (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
        (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)),
    )
    k = min(range(3), key=lambda i: abs(cols[i][2]))  # ties go to the lower index; an upright cube takes column 0
    return float(p[0]), float(p[1]), math.atan2(cols[k][1], cols[k][0])


def cube_obb2d_exact(pose_or_xy_yaw, half, pad=0.0):
    """Exact 2D obstacle ``(c, A, h)`` of a cube, which can be placed directly into ``avoid`` of ``spawn_random_cube`` / ``spawn_random_target``.

    * ``pose_or_xy_yaw``: three numbers ``(x, y, yaw)`` (yaw in radians about z), or a pose with ``.p`` / ``.q``
      (mani_skill ``Pose``, ``sapien.Pose``), or an actor with ``.pose`` (its current pose is taken).
    * ``half``: cube half extent (meters, scalar); ``pad``: margin added to each of the two half axes, same semantics as ``extra_pad`` of the actor path
      ``avoid=[(actor, pad)]``.
    * Returns ``(c, A, h)``: ``c`` of shape ``(2,)``, ``A`` of shape ``(2, 2)`` (each column a unit axis, ``[[cos, -sin], [sin, cos]]``),
      ``h`` of shape ``(2,)``, all float64 ``np.ndarray`` -- exactly the format the two spawn functions recognize as a "prefab obstacle"
      (a triple whose first two items are ``np.ndarray``), and bitwise isomorphic to ``_build_new_cube_obb2d(x, y, half, yaw, pad)``.

    Pure function: draws no random numbers and reads/writes no env state; the two axes are always orthogonal unit vectors and never degenerate
    into a line segment the way ``_trimesh_box_to_obb2d`` does when the vertical axis falls into the first two columns (plan 2.0 (1)).
    """
    if hasattr(pose_or_xy_yaw, "p") and hasattr(pose_or_xy_yaw, "q"):
        x, y, yaw = _xy_yaw_from_pose(pose_or_xy_yaw)
    elif hasattr(pose_or_xy_yaw, "pose"):
        x, y, yaw = _xy_yaw_from_pose(pose_or_xy_yaw.pose)
    else:
        values = _as_numpy(pose_or_xy_yaw).reshape(-1)
        if values.shape != (3,):
            raise ValueError(f"cube_obb2d_exact: needs three numbers (x, y, yaw), got shape {values.shape}")
        x, y, yaw = (float(v) for v in values)
    half = float(half)
    pad = float(pad)
    if not half > 0.0:
        raise ValueError(f"cube_obb2d_exact: half must be positive, got {half}")
    if pad < 0.0:
        raise ValueError(f"cube_obb2d_exact: pad must not be negative, got {pad}")
    # same formulation as object_generation._build_new_cube_obb2d, so the same (x, y, yaw) yields bitwise-identical arrays
    c = np.array([x, y], dtype=np.float64)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    A = np.array([[cos_y, -sin_y],
                  [sin_y, cos_y]], dtype=np.float64)
    h = np.array([half + pad, half + pad], dtype=np.float64)
    return c, A, h
