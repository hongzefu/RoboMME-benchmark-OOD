"""V4 xhard: outer-ring distractor containers for the Unmask family (NEWTASK_RELEASE_V4_PLAN 2.7 (1) / B3 / B13 / H1).

Only called from xhard branches; the original three tiers never enter this module, so V0/V1 are unaffected.

Approach (all criteria are this module's own; the existing semantics of ``object_generation.spawn_random_bin`` are **not changed**):

* Position: draw ``(x, y)`` uniformly on ``[-R, R]²`` (``R = ring_max_abs_xy[1]``), rejecting
  points with ``max(|x|,|y|) < ring_max_abs_xy[0]`` ⇒ the center lies strictly in the square ring ``max(|x|,|y|) ∈ [r_in, r_out]``;
* Camera visibility: project the 8 corner points of the bottom and top faces of the container's "circumscribed square for any yaw" into the front camera
  (``eye / target / fov / resolution`` identical to the hard-coded values in the env's ``_default_sensor_configs``); accepted only if all fall inside the image;
* Avoidance: same criterion as ``spawn_random_bin`` -- actors in ``avoid`` are OBBs inflated by ``min_gap``, prefab OBB tuples are used as-is,
  the point distance from the candidate center to an obstacle OBB must be ``>= bin_half_size + min_gap``; placed distractor containers are added to ``avoid`` immediately;
* yaw: ``u·90°`` is drawn only after the position passes all checks (consistent with ``spawn_random_bin``);
* Containing cubes: randomly take a count in the closed interval ``cube_count_range``, randomly pick containers, pick colors from ``DISTRACTOR_COLORS`` without replacement.

The random call order is fixed: per container (2 rand per rejection-loop iteration + 1 yaw after passing) -> cube count -> pick containers -> pick colors.
Callers must place this function **after all existing sampling points** of the env (red line N5).

Every sampling point goes through ``recorder.value`` (with ``decision_key``); the requested and actual counts are recorded via ``recorder.record``;
if not all can be placed, :class:`SceneGenerationError` is raised directly (2.2 (4): no silent truncation).
"""

from __future__ import annotations

import numpy as np
import torch

from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb

from .object_generation import _trimesh_box_to_obb2d, build_bin, spawn_fixed_cube
from .SceneGenerationError import SceneGenerationError
from .xhard import DISTRACTOR_COLORS

# Byte-identical to _default_sensor_configs of VideoUnmask / ButtonUnmask (hard-coded literals there)
BASE_CAMERA_EYE = (0.3, 0.0, 0.4)
BASE_CAMERA_TARGET = (0.0, 0.0, -0.2)
BASE_CAMERA_FOV = np.pi / 2
BASE_CAMERA_RES = 256

# Name prefix for distractor containers: deliberately not bin_<i>, otherwise the reveal animation in step would sweep them in (plan 2.7 (1))
DISTRACTOR_BIN_PREFIX = "distractor_bin"
DISTRACTOR_CUBE_PREFIX = "distractor_cube"


def bin_geometry(cube_half_size: float) -> tuple[float, float, float]:
    """Return ``(sampling-criterion half extent, any-yaw circumscribed half extent, height)``, using the same dimensions as ``build_bin`` / ``spawn_random_bin``."""
    inner_side = cube_half_size * 2.5
    wall_thickness = 0.005
    floor_thickness = 0.004
    half = (inner_side + wall_thickness) * 0.5          # bin_half_size of spawn_random_bin
    outer_half = inner_side * 0.5 + wall_thickness       # outer half extent (0.03)
    height = floor_thickness + cube_half_size * 2.5      # floor plate + wall height
    return half, outer_half * np.sqrt(2.0), height


def _camera_axes(eye, target):
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return eye, forward, right, up


def visible_in_camera(points, eye=BASE_CAMERA_EYE, target=BASE_CAMERA_TARGET,
                      fov=BASE_CAMERA_FOV, margin_px: float = 0.0, res: int = BASE_CAMERA_RES) -> bool:
    """Pinhole model: visible only if all points are in front of the camera and project into the ``[margin, res-margin]`` pixel box."""
    eye, forward, right, up = _camera_axes(eye, target)
    tan_half = np.tan(fov / 2.0)
    limit = 1.0 - 2.0 * margin_px / res
    for point in points:
        d = np.asarray(point, dtype=np.float64) - eye
        depth = float(d @ forward)
        if depth <= 1e-6:
            return False
        if abs(float(d @ right) / depth) / tan_half > limit:
            return False
        if abs(float(d @ up) / depth) / tan_half > limit:
            return False
    return True


def bin_corners(x: float, y: float, reach: float, height: float):
    """8 corner points of the bottom and top faces of the container's circumscribed square for any yaw (conservative: holds for every yaw)."""
    return [(x + sx * reach, y + sy * reach, z)
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for z in (0.0, height)]


def in_ring(x: float, y: float, ring) -> bool:
    m = max(abs(x), abs(y))
    return float(ring[0]) <= m <= float(ring[1])


def _obstacle_obbs(avoid, min_gap):
    """Same collection convention as spawn_random_bin: actors inflated by min_gap, prefab OBB tuples as-is."""
    out = []
    for item in avoid:
        if isinstance(item, tuple):
            if len(item) == 3 and isinstance(item[0], np.ndarray) and isinstance(item[1], np.ndarray):
                out.append(item)
                continue
            actor, pad = item
        else:
            actor, pad = item, min_gap
        try:
            out.append(_trimesh_box_to_obb2d(get_actor_obb(actor, to_world_frame=True, vis=False),
                                             extra_pad=float(pad)))
        except Exception:  # noqa: BLE001 consistent with the original implementation: objects without a physical mesh are ignored
            pass
    return out


def _hits(pos, obbs, reach):
    for c_obs, a_obs, h_obs in obbs:
        local = a_obs.T @ (pos - c_obs)
        closest = c_obs + a_obs @ np.clip(local, -h_obs, h_obs)
        if np.linalg.norm(pos - closest) < reach:
            return True
    return False


def spawn_ring_distractor_bins(env, *, cfg: dict, avoid: list, generator: torch.Generator,
                               recorder, hidden_half_size: float,
                               spec_prefix: str = "objects.distractors",
                               decision_prefix: str = "xhard.distractor"):
    """Place ``cfg['count']`` outer-ring distractor containers, a random ``cube_count_range`` of which contain a distractor-color cube.

    ``cfg`` is the ``xhard.distractor`` subtree of decision; returns two lists ``(bins, cubes)``,
    and appends the placed containers to ``avoid`` (callers can keep using it for later avoidance).
    """
    count = int(cfg["count"])
    ring = [float(v) for v in cfg["ring_max_abs_xy"]]
    low, high = (int(v) for v in cfg["cube_count_range"])
    min_gap = env.cube_half_size * float(cfg["min_gap_factor"])
    max_trials = int(cfg["max_trials"])
    if not (0 < ring[0] < ring[1]):
        raise ValueError(f"invalid ring_max_abs_xy: {ring}")
    if not (0 <= low <= high <= count) or high > len(DISTRACTOR_COLORS):
        raise ValueError(f"invalid cube_count_range: {[low, high]} ({count} containers, {len(DISTRACTOR_COLORS)} distractor colors)")

    pool = [c["name"] for c in DISTRACTOR_COLORS]
    if list(cfg["color_pool"]) != pool:
        # the color pool is global decision B2 (yellow/cyan/magenta) and must not change per episode; declared in decision only for visibility in records
        raise ValueError(f"color_pool must equal the global distractor color pool {pool}, got {cfg['color_pool']}")

    half, reach_any_yaw, height = bin_geometry(env.cube_half_size)
    reject_reach = half + min_gap
    span = ring[1]
    recorder.record(f"{spec_prefix}.requested", count)

    bins = []
    for i in range(count):
        placed = None
        # obstacle OBBs are collected only once per container (trimesh OBB is slow); avoid is unchanged within this container's rejection loop
        obbs = _obstacle_obbs(avoid, min_gap)
        for _ in range(max_trials):
            x = float(torch.rand(1, generator=generator).item() * 2.0 * span - span)
            y = float(torch.rand(1, generator=generator).item() * 2.0 * span - span)
            if not in_ring(x, y, ring):
                continue
            if not visible_in_camera(bin_corners(x, y, reach_any_yaw, height)):
                continue
            if _hits(np.array([x, y], dtype=np.float64), obbs, reject_reach):
                continue
            yaw = float(torch.rand(1, generator=generator).item() * 90.0)
            placed = (x, y, yaw)
            break
        if placed is None:
            recorder.record(f"{spec_prefix}.placed", len(bins))
            raise SceneGenerationError(
                f"xhard distractor containers cannot all be placed: requested {count}, container {i} has no feasible position within {max_trials} attempts"
            )
        x, y, yaw = recorder.value(f"{spec_prefix}.bins.{i}", list(placed),
                                   decision_key=f"{decision_prefix}.ring_max_abs_xy")
        actor = build_bin(env, callsign=f"{DISTRACTOR_BIN_PREFIX}_{i}", position=[x, y, 0.002],
                          z_rotation_deg=yaw)
        bins.append(actor)
        avoid.append(actor)
    recorder.record(f"{spec_prefix}.placed", len(bins))

    n_cubes = int(recorder.value(
        f"{spec_prefix}.cube_count",
        torch.randint(low, high + 1, (1,), generator=generator).item(),
        decision_key=f"{decision_prefix}.cube_count_range",
    ))
    cube_bins = recorder.value(
        f"{spec_prefix}.cube_bins",
        torch.randperm(count, generator=generator)[:n_cubes].tolist(),
        decision_key=f"{decision_prefix}.count",
    )
    color_idx = recorder.value(
        f"{spec_prefix}.cube_colors",
        torch.randperm(len(DISTRACTOR_COLORS), generator=generator)[:n_cubes].tolist(),
        decision_key=f"{decision_prefix}.color_pool",
    )
    if len(cube_bins) != n_cubes or len(color_idx) != n_cubes:
        raise SceneGenerationError(f"distractor cube spec is self-contradictory: count {n_cubes}, containers {cube_bins}, colors {color_idx}")

    cubes = []
    for j, (b_idx, c_idx) in enumerate(zip(cube_bins, color_idx)):
        color = DISTRACTOR_COLORS[int(c_idx)]
        p = bins[int(b_idx)].pose.p
        if isinstance(p, torch.Tensor):
            p = p[0].detach().cpu().numpy()
        cube = spawn_fixed_cube(
            env,
            position=[float(p[0]), float(p[1])],
            half_size=hidden_half_size,
            color=color["rgba"],
            name_prefix=f"{DISTRACTOR_CUBE_PREFIX}_{j}_{color['name']}",
            yaw=0.0,
            dynamic=True,
        )
        cubes.append(cube)
    return bins, cubes


# -- Reveal of distractor containers and failure on mis-grasp (user decision 2026-09-22: "take part in reveal + mis-grasp means failure") --------------
# Shared by the xhard of the four Unmask envs (VideoUnmask / ButtonUnmask / VideoUnmaskSwap / ButtonUnmaskSwap);
# the distractor containers of the two Swap envs are generated by unmask_swap_xhard.build_distractors and are likewise attached to env.distractor_bins.
# Only called in new-value family branches (V6 family check); the original three tiers never enter.


def reveal_distractor_bins(env, *, start_step: int, end_step: int, cur_step: int) -> None:
    """Make distractor containers use **the same reveal mechanism, in the same period** as the in-region containers: call one by one
    ``statechange.lift_and_drop_objects_back_to_original`` (moved far away in the first half of the window to expose contents, put back at the half-window step).

    Empty distractor containers are lifted too (exposing nothing). Only reveals; not added to ``spawned_bins``, not part of swap / nearest neighbor.
    """
    from .statechange import lift_and_drop_objects_back_to_original

    for actor in list(getattr(env, "distractor_bins", None) or []):
        if actor is None:
            continue
        lift_and_drop_objects_back_to_original(
            env, obj=actor, start_step=start_step, end_step=end_step, cur_step=cur_step,
        )


def any_distractor_bin_lifted(env):
    """True if any distractor container is lifted; same criterion as in-region containers (``subgoal_evaluate_func.is_bin_pickup``: z > 0.15)."""
    from .subgoal_evaluate_func import is_any_bin_pickup

    return is_any_bin_pickup(env, [a for a in (getattr(env, "distractor_bins", None) or []) if a is not None])


def add_distractor_misgrasp_failure(env, tasks) -> int:
    """For every entry in the task list that **already has** a ``failure_func``, append "fail if any distractor container is lifted"; returns the number of entries changed.

    * Only entries whose ``failure_func`` is not None are wrapped (i.e. the grasp / drop kinds; entries such as static or buttons that originally never fail are untouched);
    * the new ``failure_func`` returns a list ``[original result, distractor criterion]``, and ``_coerce_failure_result`` takes any --
      the shape of the original result (e.g. the single-element list returned by ButtonUnmask's first grasp task) is kept as-is as the first item of the list;
    * when the original ``failure_func`` is not callable (a precomputed value), it is used as-is as the first item;
    * only the ``failure_func`` key is changed; ``solve`` (the object replaced by ``inject_fail_grasp``) and other keys are untouched, and call order does not matter.
    """
    changed = 0
    for task in tasks:
        if not isinstance(task, dict):
            continue
        original = task.get("failure_func")
        if original is None:
            continue

        def _combined(original=original):
            first = original() if callable(original) else original
            return [first, any_distractor_bin_lifted(env)]

        _combined.__name__ = "xhard_distractor_misgrasp_failure"
        task["failure_func"] = _combined
        changed += 1
    return changed
