"""Shared components for the xhard tier of the two UnmaskSwap envs (VideoUnmaskSwap / ButtonUnmaskSwap).

Based on NEWTASK_RELEASE_V4_PLAN 2.7 (1)(3), 2.10, 2.11, 2.21:

* **Swap windows**: the original three tiers swap 50 steps per segment, first segment starting at 64; xhard swap speed ×1.5 (B4)
  ⇒ ``round(50 / 1.5) = 33`` steps per segment, start 64 unchanged. :func:`scaled_window_steps` with multiplier 1
  **returns the integer base as-is**, so the original three tiers' schedule is item-by-item identical to before the change.
* **Distractor containers** (B3 / B13 / H1): 3 extra containers in the outer ring outside the current region
  ``max(|x|,|y|) ∈ [0.2675, 0.45]``, where the camera can see them; 1-2 contain an "other color" cube (B2 color pool),
  stored separately in ``distractor_bins`` and **not in ``spawned_bins``** (naturally never chosen as swap partners, not in the reveal animation,
  not in nearest neighbor), but they **must be in collision checks**: at reset this module rehearses every segment's sweep along the deterministic swap sequence,
  and a candidate position intersecting any segment's sweep is rejected and redrawn; at runtime the env explicitly includes the distractor containers in the initial-state and sweep checks.
* **Random stream (N5)**: all distractor-container sampling goes through an **independent dedicated stream** (seed = episode seed + fixed salt),
  and the main stream draws nothing extra -- the original three tiers never enter here, and xhard's main-stream value sequence is the same as without distractor containers.

The S5/O4 entries of this module are only called in new-value tier branches; the original three tiers never go through these planning functions.

V5 (NEWTASK_RELEASE_V5_PLAN 2.5-2.7, L13-L23) adds a section at the end of this file, and the xhard of both envs now calls it:

* **Distractor containers** now use the unified sampler (``unmask_distractor_sampler``): V4 annulus ``[0.2675, 0.45]``, 10 containers, cubes ``[5, 5]``,
  OBB spacing, exact 8-corner visibility, 1024 attempts, indexed names, ``cube_bins`` mapping; still on the independent stream ``distractor_generator(seed)``,
  still **not in ``spawned_bins`` and not named ``bin_<i>``**.
* **Outer ring swaps in sync with the inner ring**: planned at reset (:func:`plan_distractor_swaps`), exactly one outer swap per inner window; initiators rotate through all by
  ``randperm(count)`` (L17), infeasible ones advance along the permutation (L18), the partner is the XY nearest neighbor among distractor containers, paths stay inside the image the whole way and
  keep a circle clearance >= 0.04 from inner containers, BUS >= 0.122 from the button center (L19), lane 0.07 (L21); if a window is entirely infeasible the whole layout is resampled, at most 16 times.
* **Collision**: at reset, inner-vs-inner sweep prejudgment (L20); outer planning uses the joint continuous proof of two pairs ``check_multi_swap_sweep``; at runtime
  :func:`joint_sweep_from_actual` re-checks the two pairs jointly from actual poses; all three enable certified prefiltering (L23).
* The V4 ``XHARD_DISTRACTOR`` / ``sample_distractors`` / ``build_distractors`` / ``visible_on_camera`` above are no longer called by the envs;
  they are kept as-is (old unit tests still pin their semantics) and affect no behavior.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from .bin_collision import (
    EPS_M,
    LANE_OFFSET,
    PREFILTER_MARGIN_M,
    CollisionRejection,
    ObjectState,
    ShapeSpec,
    _lane_endpoints,
    _prefilter_clearance,
    _prefilter_samples,
    _prefilter_track,
    _prove_pair,
    _Static,
    _swap_movers,
    bin_actor_pose,
    bin_shape_specs,
    check_multi_swap_sweep,
    check_swap_sweep,
    check_swap_sweep_prefiltered,
    object_state_from_actor,
)
from .SceneGenerationError import SceneGenerationError
from .episode_spec import EpisodeSpecError
from . import swap_uniform as _su
from .unmask_distractor_sampler import (
    V5_DISTRACTOR_PRESETS,
    DistractorLayout,
    build_distractor_actors,
    distractor_cube_bin_pairs,
    obstacle_obbs,
    parse_distractor_cfg,
    resample_distractor_layout,
)
from .unmask_distractors import BASE_CAMERA_EYE, BASE_CAMERA_FOV, BASE_CAMERA_TARGET, _camera_axes, bin_geometry
from .xhard import DISTRACTOR_COLORS

# -- Swap windows (B4) --------------------------------------------------------------
#: Start (control step) of the first swap segment; same for the original three tiers and xhard; the end of the pre-swap lock segment [0, 64) must equal it.
SWAP_WINDOW_START = 64
#: Steps per swap segment in the original three tiers (speed multiplier 1).
SWAP_WINDOW_STEPS = 50
#: xhard swap speed multiplier (user's original words: "swap speed x1.5").
NEWVALUE_SWAP_SPEED_MULTIPLIER = 1.5


def validate_hidden_bin_selection(indices: Sequence[int], *, permutation_size: int = 3) -> list[int]:
    """Re-check the M5(b) hiding slots: exactly the first three containers are selected, and the always-empty ``bin_3`` must not be selected."""
    raw = list(indices)
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in raw):
        raise EpisodeSpecError(f"objects.selected must be integer slots, got {raw!r}; M5(b) requires bin_3 to be always empty")
    values = [int(v) for v in raw]
    if (len(values) != permutation_size or len(set(values)) != len(values)
            or any(v < 0 or v >= permutation_size for v in values)):
        raise EpisodeSpecError(
            f"objects.selected must be a complete non-repeating selection from 0..{permutation_size - 1}, got {values!r}; M5(b) requires bin_3 to be always empty"
        )
    return values


def scaled_window_steps(base_steps: int, multiplier) -> int:
    """Round the per-segment swap steps by the speed multiplier: ``round(base / multiplier)``.

    When the multiplier is exactly 1, no floating-point operation is done and ``int(base_steps)`` is returned as-is, keeping the original three tiers byte-identical.
    """
    m = float(multiplier)
    if not math.isfinite(m) or m <= 0:
        raise ValueError(f"swap_speed_multiplier must be a positive finite number, got {multiplier!r}")
    if m == 1.0:
        return int(base_steps)
    steps = int(round(float(base_steps) / m))
    if steps < 1:
        raise ValueError(f"multiplier {multiplier} makes the per-segment swap steps less than 1")
    return steps


# -- Distractor containers (B3 / B13) --------------------------------------------------------
#: Three-container distractor config from V4 old snapshots, used only for release-aware legacy export.
LEGACY_V4_DISTRACTOR = {
    # number of extra containers (B3: 3)
    "count": 3,
    # number of them holding a cube, closed interval (B13: 1-2 of the 3)
    "with_cube_range": [1, 2],
    # lower and upper bounds of outer-ring max(|x|,|y|) (B13 measured recommended values)
    "ring_half_extent": [0.2675, 0.45],
    # min gap in meters between the distractor container's circumscribed circle and other objects' circumscribed circles (the min_gap used by B13 to derive the outer-ring lower bound)
    "min_gap": 0.04,
    # cube color pool (B2: yellow / cyan / magenta, globally shared)
    "colors": [item["name"] for item in DISTRACTOR_COLORS],
}

#: Rejection-sampling budget per distractor container; when exhausted SceneGenerationError is raised (2.2 (4): no silent truncation).
DISTRACTOR_MAX_TRIALS = 512
#: Seed salt of the dedicated random stream: seed = seed + salt, fully separate from the main stream (N5).
DISTRACTOR_STREAM_SALT = 0x5D157AC7

# Camera-visible ∩ table (B13 measured table, front camera eye=[0.3,0,0.4], target=[0,0,-0.2], fov 90°):
# x <= 0.43; the visible y half-width narrows linearly with x; the 7 points of the table are reproduced point by point by 0.49 − 0.45·x (error <= 0.005).
VISIBLE_X_MAX = 0.43
VISIBLE_Y_AT_X0 = 0.49
VISIBLE_Y_SLOPE = 0.45


def bin_footprint_radius(cube_half_size: float) -> float:
    """Circumscribed-circle radius of the container's outer (square) outline; same source as build_bin: outer half extent = (2.5·h + 0.005)/2."""
    half = (cube_half_size * 2.5 + 0.005) * 0.5
    return half * math.sqrt(2.0)


def visible_on_camera(x: float, y: float, radius: float) -> bool:
    """Conservatively judge via the circumscribed circle whether the whole container lies within front-camera-visible ∩ table."""
    far_x = x + radius
    if far_x > VISIBLE_X_MAX:
        return False
    return abs(y) + radius <= VISIBLE_Y_AT_X0 - VISIBLE_Y_SLOPE * far_x


def distractor_generator(seed: int) -> torch.Generator:
    """Dedicated random stream for distractor containers (N5: the main stream draws nothing extra)."""
    generator = torch.Generator()
    generator.manual_seed((int(seed) + DISTRACTOR_STREAM_SALT) % (2**63))
    return generator


def solve_hold_obj_xhard(env, planner, static_steps: int) -> None:
    """xhard-only wait in place: same semantics as ``solve_hold_obj(close=False)``, swallowing only ``AttributeError``.

    The shared function ``utils/subgoal_planner_func.py::solve_hold_obj`` wraps
    ``planner.open_gripper()`` in a bare ``except:``: after xhard enables runtime collision checking (H1), the ``BinCollisionError`` raised by ``env.step`` at swap start
    would be swallowed, ``elapsed_steps`` would not advance, and the wait loop would never end (measured locally to hang until the external timeout).
    Here collision rejections (and every other non-``AttributeError`` exception) propagate as-is, and ``_worker`` classifies them as task-level failures.
    Per N12 the shared function is not fixed in place; the original three tiers keep using it, new-value tiers use this function.
    """
    start_step = int(getattr(env, "elapsed_steps", 0))
    target_step = start_step + static_steps
    while int(getattr(env, "elapsed_steps", 0)) < target_step:
        try:
            planner.open_gripper()
        except AttributeError:
            pass
    return None


def predict_swap_sweeps(env, partner_axes: Sequence[int]) -> list[tuple[ObjectState, ObjectState]]:
    """Rehearse all swap segments with the same semantics as ``step``, returning the starting state ``(initiator, partner)`` of each segment.

    The initiator is ``swap_pair{k}_idx1``; the partner is the other ``spawned_bins`` closest in XY at swap start
    (strictly less, ties go to the earlier in generation order, consistent with ``step``'s scan); after each segment the two poses are exchanged
    (the final state of ``swap_flat_two_lane``: both position and quaternion move to the other's starting state). Distractor containers are not in
    ``spawned_bins``, so they do not affect partner selection -- the rehearsal result is independent of whether distractor containers exist.
    """
    bins = list(env.spawned_bins)
    states = [object_state_from_actor(actor, f"bin_{index}") for index, actor in enumerate(bins)]
    positions = [np.asarray(env._get_actor_position(actor), dtype=np.float32) for actor in bins]
    axes = list(partner_axes)
    sweeps = []
    for k in range(int(env.swap_times)):
        initiator = getattr(env, f"swap_pair{k + 1}_idx1")
        a = next(index for index, actor in enumerate(bins) if actor is initiator)
        b, best = None, float("inf")
        for index, position in enumerate(positions):
            if index == a:
                continue
            dist = np.linalg.norm(positions[a][axes] - position[axes])
            if dist < best:
                b, best = index, dist
        if b is None:
            break
        sweeps.append((states[a], states[b]))
        state_a, state_b = states[a], states[b]
        states[a] = ObjectState(name=state_a.name, p=state_b.p.copy(), q=state_b.q.copy(), shapes=state_a.shapes)
        states[b] = ObjectState(name=state_b.name, p=state_a.p.copy(), q=state_a.q.copy(), shapes=state_b.shapes)
        positions[a], positions[b] = positions[b].copy(), positions[a].copy()
    return sweeps


def sample_distractors(
    *,
    generator: torch.Generator,
    cfg: dict,
    obstacles: Sequence[tuple[Sequence[float], float]],
    sweeps: Sequence[tuple[ObjectState, ObjectState]],
    recorder,
    cube_half_size: float,
) -> dict[str, Any]:
    """Rejection-sample all distractor container positions in the outer ring; returns ``{"placements": [...], "cube_colors": [...]}``.

    ``obstacles``: ``(xy, circumscribed radius)`` of objects already in the scene (containers, button, etc.);
    ``sweeps``: result of :func:`predict_swap_sweeps`; a candidate intersecting any segment's sweep is rejected (H1).
    Every sampling point goes through ``recorder`` (SpecRecorder); if the requested and actual counts differ, raise directly.
    """
    lo, hi = (int(v) for v in cfg["with_cube_range"])
    count = int(cfg["count"])
    if not 0 <= lo <= hi <= count:
        raise ValueError(f"with_cube_range {cfg['with_cube_range']} must lie within [0, count={count}]")
    inner, outer = (float(v) for v in cfg["ring_half_extent"])
    min_gap = float(cfg["min_gap"])
    pool = list(cfg["colors"])
    known = {item["name"] for item in DISTRACTOR_COLORS}
    if not set(pool) <= known:
        raise ValueError(f"distractor colors {pool} exceed the global color pool {sorted(known)}")

    n_with_cube = recorder.value(
        "objects.distractors.n_with_cube",
        int(torch.randint(lo, hi + 1, (1,), generator=generator).item()),
        decision_key="xhard.distractor.with_cube_range",
    )
    order = torch.randperm(len(pool), generator=generator).tolist()
    cube_colors = recorder.value(
        "objects.distractors.cube_colors",
        [pool[i] for i in order][: int(n_with_cube)],
        decision_key="xhard.distractor.colors",
    )

    radius = bin_footprint_radius(cube_half_size)
    shapes = bin_shape_specs(cube_half_size)
    occupied = [(np.asarray(xy, dtype=np.float64)[:2], float(r)) for xy, r in obstacles]
    recorder.record("layout.distractors_requested", count)
    placements = []
    for i in range(count):
        chosen = None
        for _trial in range(DISTRACTOR_MAX_TRIALS):
            x = float((torch.rand(1, generator=generator).item() * 2.0 - 1.0) * outer)
            y = float((torch.rand(1, generator=generator).item() * 2.0 - 1.0) * outer)
            yaw = float(torch.rand(1, generator=generator).item() * 90.0)
            if max(abs(x), abs(y)) < inner:
                continue
            if not visible_on_camera(x, y, radius):
                continue
            here = np.array([x, y], dtype=np.float64)
            if any(np.linalg.norm(here - xy) < r + radius + min_gap for xy, r in occupied):
                continue
            p, q = bin_actor_pose([x, y], yaw, cube_half_size)
            candidate = ObjectState(name=f"distractor_bin_{i}", p=p, q=q, shapes=shapes)
            if any(check_swap_sweep(a, b, [candidate], sweep_index=k, stage="distractor")[1] is not None
                   for k, (a, b) in enumerate(sweeps)):
                continue
            chosen = (x, y, yaw)
            break
        if chosen is None:
            raise SceneGenerationError(
                f"distractor container {i} found no legal position within {DISTRACTOR_MAX_TRIALS} attempts (outer ring {inner}~{outer})"
            )
        x, y, yaw = recorder.value(f"layout.distractors.{i}", list(chosen))
        placements.append({"xy": [float(x), float(y)], "yaw_deg": float(yaw)})
        occupied.append((np.array([x, y], dtype=np.float64), radius))
    recorder.record("layout.distractors_placed", len(placements))
    if len(placements) != count:
        raise SceneGenerationError(f"distractor containers requested {count}, but {len(placements)} placed")
    return {"placements": placements, "cube_colors": list(cube_colors)}


def build_distractors(env, layout: dict, build_bin, spawn_fixed_cube, cube_divisor: float = 1.2):
    """Build distractor containers and cubes from the result of :func:`sample_distractors`; the first ``len(cube_colors)`` containers hold cubes.

    Names are always ``distractor_bin_<i>`` / ``distractor_cube_<color name>``; **no ``bin_<i>`` attribute is set**,
    so the reveal / swap logic that scans ``bin_<i>`` does not pick them up by mistake.
    """
    rgba = {item["name"]: item["rgba"] for item in DISTRACTOR_COLORS}
    bins, cubes = [], []
    for i, entry in enumerate(layout["placements"]):
        x, y = entry["xy"]
        bins.append(build_bin(env, callsign=f"distractor_bin_{i}", position=[x, y, 0.002],
                              z_rotation_deg=entry["yaw_deg"]))
    for i, color in enumerate(layout["cube_colors"]):
        x, y = layout["placements"][i]["xy"]
        cubes.append(spawn_fixed_cube(
            env,
            position=[x, y],
            half_size=env.cube_half_size / cube_divisor,
            color=rgba[color],
            name_prefix=f"distractor_cube_{color}",
            yaw=0.0,
            dynamic=True,
        ))
    return bins, cubes


# ════════════════════════════════════════════════════════════════════════════════
# V5 (NEWTASK_RELEASE_V5_PLAN 2.5-2.7): unified sampler with 10 distractor containers + outer ring swapping in sync with the inner ring
# ════════════════════════════════════════════════════════════════════════════════
#: The two envs served by this section.
V5_SWAP_TASKS = ("VideoUnmaskSwap", "ButtonUnmaskSwap")
#: Outer-ring initiator rule (L17 b): ``perm = randperm(count)``; window k starts from ``perm[k % count]``.
OUTER_INITIATOR_RULE = "permutation_cycle"
#: Fallback when a window's first candidate is infeasible (L18 a): advance to the next initiator along the permutation; resample the whole layout only if none works.
OUTER_FALLBACK_RULE = "next_in_permutation"
#: Outer-ring partner rule (convention 5): the distractor container nearest in 2-axis XY Euclidean distance, strict <, ties go to the lower index; planned at reset.
OUTER_PARTNER_RULE = {
    "selection": "nearest",
    "position_axes": [0, 1],
    "tie_break": "first_in_index_order",
    "population": "distractor_bins",
    "resolve_at": "reset_plan",
}


#: V6 (plan 2.2 outer ring O4, M7(a)): per window, among feasible object pairs take the minimum by "max, then sum of the two participation counts", ties uniform,
#: immediate undo forbidden (unless no other choice); after placement indices are reshuffled by ``randperm(count)`` (uniform across episodes). The four path checks are not relaxed.
OUTER_BALANCED_RULE = "balanced_greedy_o4"
OUTER_BALANCED_FALLBACK = "undo_only_if_no_other_feasible"
OUTER_BALANCED_PARTNER_RULE = {
    "selection": "balanced_greedy_with_initiator",
    "score": "max_then_sum_of_participation",
    "tie_break": "uniform_local_generator",
    "population": "distractor_bins",
    "relabel": "randperm_count_after_placement",
    "resolve_at": "reset_plan",
}


def v5_distractor_cfg(task: str) -> dict:
    """V5 declared value of ``decision.xhard.distractor``: preset of the unified sampler (L16 b: V4 annulus, 10 containers, cubes [5,5])."""
    if task not in V5_SWAP_TASKS:
        raise ValueError(f"only {V5_SWAP_TASKS} are supported, got {task!r}")
    return copy.deepcopy(V5_DISTRACTOR_PRESETS[task])


def v6_distractor_cfg(task: str, count: int) -> dict:
    """Adjust the number of outer-ring distractor containers per the V6 gradient; other V5 region, color, spacing and sampling rules are reused item by item."""
    if task not in V5_SWAP_TASKS:
        raise ValueError(f"only {V5_SWAP_TASKS} are supported, got {task!r}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2 or count % 2:
        raise ValueError(f"the number of outer-ring distractor containers must be an even number >= 2, got {count!r}")
    cfg = v5_distractor_cfg(task)
    cfg["count"] = count
    cfg["cube_count_range"] = [count // 2, count // 2]
    return cfg


def v5_distractor_swap_cfg(task: str) -> dict:
    """V5 declared value of ``decision.xhard.distractor_swap`` (config block of the plan 2.5 pseudocode).

    Compared with the pseudocode there are three extra explicit keys (decided by the implementer, none changes the design intent): ``path_samples`` (number of sample points for path constraints, 401, same density as
    certified prefiltering), ``plan_pad_m`` (each distractor container's collision box is inflated by 5 mm during outer planning to absorb deviations between runtime and nominal poses, 2.5 "key design point");
    VUS has no button, so ``min_button_center_dist_m`` is ``None``.
    """
    if task not in V5_SWAP_TASKS:
        raise ValueError(f"only {V5_SWAP_TASKS} are supported, got {task!r}")
    return {
        "enabled": True,
        "initiator_rule": OUTER_INITIATOR_RULE,
        "fallback": OUTER_FALLBACK_RULE,
        "partner": copy.deepcopy(OUTER_PARTNER_RULE),
        "lane_offset": 0.07,
        "smooth": True,
        "path_constraints": {
            "camera_visible": True,
            "min_inner_circle_clearance_m": 0.04,
            "min_button_center_dist_m": 0.122 if task == "ButtonUnmaskSwap" else None,
        },
        "path_samples": 401,
        "plan_pad_m": 0.005,
        "layout_max_attempts": 16,
    }


@dataclass(frozen=True)
class DistractorSwapConfig:
    """Validated outer-ring swap config."""

    enabled: bool
    lane_offset: float
    camera_visible: bool
    min_inner_circle_clearance_m: float
    min_button_center_dist_m: float | None
    path_samples: int
    plan_pad_m: float
    layout_max_attempts: int
    #: V6 (2.2 outer ring O4): ``OUTER_INITIATOR_RULE`` (V5 permutation rotation + nearest neighbor) or ``OUTER_BALANCED_RULE`` (balancing greedy + index reshuffle)
    rule: str = "permutation_cycle"


DISTRACTOR_SWAP_CFG_KEYS = ("enabled", "initiator_rule", "fallback", "partner", "lane_offset", "smooth",
                            "path_constraints", "path_samples", "plan_pad_m", "layout_max_attempts")


def parse_distractor_swap_cfg(cfg: dict | DistractorSwapConfig) -> DistractorSwapConfig:
    """Validate ``decision.xhard.distractor_swap``; only the rule names fixed by the plan are accepted, other values are range-checked."""
    if isinstance(cfg, DistractorSwapConfig):
        return cfg
    keys = set(cfg)
    missing = [k for k in DISTRACTOR_SWAP_CFG_KEYS if k not in keys]
    extra = sorted(keys - set(DISTRACTOR_SWAP_CFG_KEYS))
    if missing or extra:
        raise ValueError(f"distractor_swap keys mismatch: missing {missing}, extra {extra}")
    if cfg["initiator_rule"] == OUTER_BALANCED_RULE:
        # V6 (plan 2.2 outer ring O4): initiator and partner are chosen together among feasible pairs by balancing greedy; fallback and partner rules switch to V6's declared values accordingly
        if cfg["fallback"] != OUTER_BALANCED_FALLBACK:
            raise ValueError(f"fallback only supports {OUTER_BALANCED_FALLBACK!r}, got {cfg['fallback']!r}")
        if dict(cfg["partner"]) != OUTER_BALANCED_PARTNER_RULE:
            raise ValueError(f"partner only supports {OUTER_BALANCED_PARTNER_RULE}, got {cfg['partner']}")
    else:
        if cfg["initiator_rule"] != OUTER_INITIATOR_RULE:
            raise ValueError(f"initiator_rule only supports {OUTER_INITIATOR_RULE!r}, got {cfg['initiator_rule']!r}")
        if cfg["fallback"] != OUTER_FALLBACK_RULE:
            raise ValueError(f"fallback only supports {OUTER_FALLBACK_RULE!r}, got {cfg['fallback']!r}")
        if dict(cfg["partner"]) != OUTER_PARTNER_RULE:
            raise ValueError(f"partner only supports {OUTER_PARTNER_RULE}, got {cfg['partner']}")
    lane = float(cfg["lane_offset"])
    if lane != LANE_OFFSET:
        # the joint proof (_Mover.pose_at) recomputes paths with bin_collision.LANE_OFFSET; the two must be equal for the proof to mean anything
        raise ValueError(f"lane_offset must equal the collision criterion's LANE_OFFSET={LANE_OFFSET}, got {lane}")
    if cfg["smooth"] is not True:
        raise ValueError("smooth must be True (smoothstep, same as the inner ring)")
    pc = cfg["path_constraints"]
    if set(pc) != {"camera_visible", "min_inner_circle_clearance_m", "min_button_center_dist_m"}:
        raise ValueError(f"path_constraints keys mismatch: {sorted(pc)}")
    clearance = float(pc["min_inner_circle_clearance_m"])
    button = pc["min_button_center_dist_m"]
    button = None if button is None else float(button)
    samples = int(cfg["path_samples"])
    pad = float(cfg["plan_pad_m"])
    attempts = int(cfg["layout_max_attempts"])
    if not (math.isfinite(clearance) and clearance >= 0.0):
        raise ValueError(f"invalid min_inner_circle_clearance_m: {clearance}")
    if button is not None and not (math.isfinite(button) and button >= 0.0):
        raise ValueError(f"invalid min_button_center_dist_m: {button}")
    if samples < 2:
        raise ValueError(f"path_samples must be >= 2, got {samples}")
    if not (math.isfinite(pad) and pad >= 0.0):
        raise ValueError(f"invalid plan_pad_m: {pad}")
    if attempts < 1:
        raise ValueError(f"layout_max_attempts must be >= 1, got {attempts}")
    return DistractorSwapConfig(bool(cfg["enabled"]), lane, bool(pc["camera_visible"]), clearance, button,
                                samples, pad, attempts, str(cfg["initiator_rule"]))


# -- Inner-ring rehearsal -------------------------------------------------------------------
@dataclass(frozen=True)
class InnerWindow:
    """Rehearsal of the k-th inner window: initiator ``a``, partner ``b`` (``spawned_bins`` indices) and the nominal states of all inner containers at window start."""

    a: int
    b: int
    states: tuple[ObjectState, ...]


def predict_inner_windows_from_states(states: Sequence[ObjectState], positions: Sequence[Sequence[float]],
                                      initiators: Sequence[int], partner_axes: Sequence[int]) -> list[InnerWindow]:
    """Same semantics as :func:`predict_swap_sweeps` word for word (float32 positions, strict <, ties go to the earlier in generation order, poses exchanged after each segment),
    additionally returning all inner states at each window start (the joint proof treats the other inner containers as static bystanders)."""
    states = list(states)
    pos = [np.asarray(p, dtype=np.float32) for p in positions]
    axes = list(partner_axes)
    windows: list[InnerWindow] = []
    for a in initiators:
        a = int(a)
        b, best = None, float("inf")
        for index, position in enumerate(pos):
            if index == a:
                continue
            dist = np.linalg.norm(pos[a][axes] - position[axes])
            if dist < best:
                b, best = index, dist
        if b is None:
            break
        windows.append(InnerWindow(a=a, b=int(b), states=tuple(states)))
        state_a, state_b = states[a], states[b]
        states[a] = ObjectState(name=state_a.name, p=state_b.p.copy(), q=state_b.q.copy(), shapes=state_a.shapes)
        states[b] = ObjectState(name=state_b.name, p=state_a.p.copy(), q=state_a.q.copy(), shapes=state_b.shapes)
        pos[a], pos[b] = pos[b].copy(), pos[a].copy()
    return windows


def predict_inner_windows(env, partner_axes: Sequence[int]) -> list[InnerWindow]:
    """Rehearse all inner windows from the env's ``spawned_bins`` and ``swap_pair{k}_idx1`` (poses read from the actual actors)."""
    bins = list(env.spawned_bins)
    states = [object_state_from_actor(actor, f"bin_{index}") for index, actor in enumerate(bins)]
    positions = [np.asarray(env._get_actor_position(actor), dtype=np.float32) for actor in bins]
    initiators = []
    for k in range(int(env.swap_times)):
        initiator = getattr(env, f"swap_pair{k + 1}_idx1")
        initiators.append(next(index for index, actor in enumerate(bins) if actor is initiator))
    return predict_inner_windows_from_states(states, positions, initiators, partner_axes)


def prejudge_inner_windows(windows: Sequence[InnerWindow], *, prefilter: bool = True,
                           stats: dict | None = None) -> tuple[int, CollisionRejection] | None:
    """L20: at reset, prejudge inner-vs-inner sweeps per window (other inner containers static); return the first rejection ``(k, evidence)`` or ``None``.

    The check is the same as V4's runtime ``check_swap_sweep`` (for a single pair ``check_swap_sweep_prefiltered`` agrees with it bitwise),
    and certified prefiltering only skips pairs already proven separated (L23).
    """
    for k, window in enumerate(windows):
        others = [state for j, state in enumerate(window.states) if j not in (window.a, window.b)]
        _gap, rejection = check_swap_sweep_prefiltered(
            window.states[window.a], window.states[window.b], others,
            sweep_index=k, stage="inner_prejudge", prefilter=prefilter, stats=stats,
        )
        if rejection is not None:
            return k, rejection
    return None


# -- Geometry helpers -------------------------------------------------------------------
def padded_bin_shapes(cube_half_size: float, pad: float) -> tuple[ShapeSpec, ...]:
    """Inflate each of the container's 6 boxes by ``pad`` (only a margin for reset planning; the runtime re-check still uses the real collision boxes)."""
    return tuple(
        ShapeSpec(shape.local_p.copy(), shape.local_q.copy(), shape.half + float(pad))
        for shape in bin_shape_specs(cube_half_size)
    )


def distractor_bin_state(index: int, x: float, y: float, yaw_deg: float, cube_half_size: float,
                         shapes: Sequence[ShapeSpec]) -> ObjectState:
    """Build the nominal state of the ``index``-th distractor container from layout values ``(x, y, yaw)`` (pose replicates ``build_bin``)."""
    p, q = bin_actor_pose([float(x), float(y)], float(yaw_deg), cube_half_size)
    return ObjectState(name=f"distractor_bin_{int(index)}", p=p, q=q, shapes=tuple(shapes))


def path_samples(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(n))


def lane_center_paths(xy_a, xy_b, samples: np.ndarray, lane: float = LANE_OFFSET) -> tuple[np.ndarray, np.ndarray]:
    """Center XY trajectories of the two swappers in ``swap_flat_two_lane`` (same formula as ``bin_collision._Mover.pose_at``)."""
    a = np.asarray(xy_a, dtype=np.float64)[:2]
    b = np.asarray(xy_b, dtype=np.float64)[:2]
    delta, normal = _lane_endpoints(a, b)
    s = np.asarray(samples, dtype=np.float64)[:, None]
    offset = float(lane) * np.sin(np.pi * s)
    return a + delta * s + normal * offset, b - delta * s - normal * offset


def bins_visible_many(xy: np.ndarray, cube_half_size: float) -> np.ndarray:
    """Vectorized exact 8-corner visibility criterion: pointwise equivalent to ``unmask_distractor_sampler.bin_visible`` (``visible_in_camera(bin_corners(...))``)
    (checked with random points in unit tests). ``xy`` has shape ``(N, 2)``; returns a ``(N,)`` boolean."""
    _half, reach, height = bin_geometry(cube_half_size)
    corners = np.array([(sx * reach, sy * reach, z) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for z in (0.0, height)])
    eye, forward, right, up = _camera_axes(BASE_CAMERA_EYE, BASE_CAMERA_TARGET)
    tan_half = np.tan(BASE_CAMERA_FOV / 2.0)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    points = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)[:, None, :] + corners[None]
    d = points - eye
    depth = d @ forward
    ok = depth > 1e-6
    safe = np.where(ok, depth, 1.0)
    ok &= np.abs((d @ right) / safe) / tan_half <= 1.0
    ok &= np.abs((d @ up) / safe) / tan_half <= 1.0
    return ok.all(axis=1)


def _min_pointwise(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum distance at the same ``s`` between two identically parameterized trajectories (or a trajectory and a point)."""
    return float(np.min(np.linalg.norm(a - b, axis=-1)))


# -- H1: candidate distractor container × rehearsed inner sweeps (only proves "candidate × the two swappers") ----------------------
class InnerSweepGuard:
    """Extra rejection when placing distractor containers: reject if the (static) candidate intersects the inner swappers' sweep in any window (continuation of V4 H1).

    Item-by-item identical to the two "static object × swapper" pair checks in ``check_multi_swap_sweep([(a, b)], [candidate])`` (same bounding-sphere coarse filter,
    same certified prefilter, same ``_prove_pair``), except that the inner pair itself is not re-proven (already proven in the L20 prejudgment) -- otherwise every candidate
    would re-prove the inner pair and reset would be an order of magnitude slower. The swappers and their prefilter trajectories of each window are computed once at construction.
    """

    def __init__(self, windows: Sequence[InnerWindow]):
        self._samples = _prefilter_samples()
        self._step = float(self._samples[1] - self._samples[0])
        self._windows = []
        for k, window in enumerate(windows):
            mover_a, mover_b = _swap_movers(window.states[window.a], window.states[window.b])
            self._windows.append((k, [(mover_a, _prefilter_track(mover_a, self._samples)),
                                      (mover_b, _prefilter_track(mover_b, self._samples))]))

    def first_rejection(self, candidate: ObjectState, stage: str = "distractor") -> CollisionRejection | None:
        static = _Static(name=candidate.name, shapes=candidate.shapes, radii=candidate.radii,
                         p=candidate.p.copy(), q=candidate.q.copy())
        track = _prefilter_track(static, self._samples)
        center_r, radius_r = static.bounding_sphere()
        for k, movers in self._windows:
            for mover, mover_track in movers:
                center_l, radius_l = mover.bounding_sphere()
                if float(np.linalg.norm(center_l - center_r)) - radius_l - radius_r > EPS_M:
                    continue
                if mover_track is not None and track is not None:
                    bound = _prefilter_clearance(mover_track, track, self._step)
                    if math.isfinite(bound) and bound > PREFILTER_MARGIN_M:
                        continue
                for ia in range(len(mover.shapes)):
                    for ib in range(len(static.shapes)):
                        _gap, rejection = _prove_pair(mover, static, ia, ib, stage=stage, sweep_index=k)
                        if rejection is not None:
                            return rejection
        return None


# -- Outer-ring planning -------------------------------------------------------------------
def nearest_distractor(positions_xy: Sequence[Sequence[float]], initiator: int) -> int | None:
    """The distractor container nearest to ``initiator`` (XY, float32, strict <, ties go to the lower index); same semantics as the inner ``step`` scan."""
    pos = [np.asarray(p, dtype=np.float32)[:2] for p in positions_xy]
    best, best_dist = None, float("inf")
    for index, position in enumerate(pos):
        if index == initiator:
            continue
        dist = np.linalg.norm(pos[initiator] - position)
        if dist < best_dist:
            best, best_dist = index, dist
    return best


def evaluate_outer_candidate(
    k: int,
    window: InnerWindow,
    o: int,
    p: int,
    outer_states: Sequence[ObjectState],
    *,
    cfg: DistractorSwapConfig,
    cube_half_size: float,
    buttons_xy: Sequence[Sequence[float]] = (),
    stats: dict | None = None,
) -> tuple[bool, str | None, CollisionRejection | None]:
    """Whether the candidate outer pair ``(o, p)`` for window k is feasible; returns ``(feasible, rejection reason, collision evidence)``. Checked from cheap to expensive (L19):

    1. ``vis``: both center paths of o and p are exactly visible the whole way (when ``camera_visible`` is true);
    2. ``btn``: (BUS only) both paths keep >= ``min_button_center_dist_m`` from every button center;
    3. ``inner_clear``: the center distance of both paths to the inner ring (same-time positions on this window's two swap paths, static positions of other inner containers) minus the two circumscribed
       radii is >= ``min_inner_circle_clearance_m``;
    4. ``exact``: ``check_multi_swap_sweep([inner pair k, (o, p)], all others static)`` passes (distractor collision boxes already inflated by ``plan_pad_m``).
    """
    samples = path_samples(cfg.path_samples)
    path_o, path_p = lane_center_paths(outer_states[o].p[:2], outer_states[p].p[:2], samples, cfg.lane_offset)
    if cfg.camera_visible and not bins_visible_many(np.vstack([path_o, path_p]), cube_half_size).all():
        return False, "vis", None
    if cfg.min_button_center_dist_m is not None:
        for button in buttons_xy:
            center = np.asarray(button, dtype=np.float64)[None, :2]
            if min(_min_pointwise(path_o, center), _min_pointwise(path_p, center)) < cfg.min_button_center_dist_m:
                return False, "btn", None
    inner = window.states
    path_a, path_b = lane_center_paths(inner[window.a].p[:2], inner[window.b].p[:2], samples, LANE_OFFSET)
    clearance = min(_min_pointwise(path_o, path_a), _min_pointwise(path_o, path_b),
                    _min_pointwise(path_p, path_a), _min_pointwise(path_p, path_b))
    for j, state in enumerate(inner):
        if j in (window.a, window.b):
            continue
        clearance = min(clearance, _min_pointwise(path_o, state.p[None, :2]), _min_pointwise(path_p, state.p[None, :2]))
    if clearance - 2.0 * bin_footprint_radius(cube_half_size) < cfg.min_inner_circle_clearance_m:
        return False, "inner_clear", None
    bystanders = [state for j, state in enumerate(inner) if j not in (window.a, window.b)]
    bystanders += [state for j, state in enumerate(outer_states) if j not in (o, p)]
    if stats is not None:
        stats["joint_calls"] = stats.get("joint_calls", 0) + 1
    _gap, rejection = check_multi_swap_sweep(
        [(inner[window.a], inner[window.b]), (outer_states[o], outer_states[p])], bystanders,
        sweep_index=k, stage="outer_plan", stats=stats,
    )
    if rejection is not None:
        return False, "exact", rejection
    return True, None, None


@dataclass
class OuterSwapPlan:
    """Result of one outer planning. ``pairs[k] = (o, p)``; ``fallbacks[k]`` is how many positions the accepted initiator of window k advanced along the permutation (0 = first candidate)."""

    ok: bool
    perm: list[int]
    pairs: list[tuple[int, int]] = field(default_factory=list)
    fallbacks: list[int] = field(default_factory=list)
    fail_window: int | None = None
    reasons: dict[str, int] = field(default_factory=dict)
    final_xy: list[list[float]] = field(default_factory=list)


def plan_distractor_swaps(
    layout: DistractorLayout,
    perm: Sequence[int],
    windows: Sequence[InnerWindow],
    *,
    cfg: DistractorSwapConfig,
    cube_half_size: float,
    buttons_xy: Sequence[Sequence[float]] = (),
    stats: dict | None = None,
) -> OuterSwapPlan:
    """Outer planning at reset (the window loop of the plan 2.5 pseudocode). Pure geometry, draws no random numbers.

    For each inner window k try ``o = perm[(k + j) % count]`` (j = 0..count−1) in turn, partner ``p`` = XY nearest neighbor among distractor containers;
    the first ``(o, p)`` satisfying :func:`evaluate_outer_candidate` is accepted, their poses are nominally exchanged, and the next window follows;
    if a window is entirely infeasible, return ``ok=False`` (the caller resamples the whole distractor layout).
    """
    count = layout.count
    perm = [int(v) for v in perm]
    if sorted(perm) != list(range(count)):
        raise ValueError(f"outer initiator permutation {perm} is not a permutation of 0..{count - 1} (inclusive)")
    shapes = padded_bin_shapes(cube_half_size, cfg.plan_pad_m)
    outer = [distractor_bin_state(i, x, y, yaw, cube_half_size, shapes) for i, (x, y, yaw) in enumerate(layout.bins)]
    plan = OuterSwapPlan(ok=False, perm=perm)
    for k, window in enumerate(windows):
        chosen = None
        for j in range(count):
            o = perm[(k + j) % count]
            p = nearest_distractor([state.p[:2] for state in outer], o)
            if p is None:
                continue
            ok, reason, _rejection = evaluate_outer_candidate(
                k, window, o, p, outer, cfg=cfg, cube_half_size=cube_half_size, buttons_xy=buttons_xy, stats=stats,
            )
            if ok:
                chosen = (o, p, j)
                break
            plan.reasons[reason] = plan.reasons.get(reason, 0) + 1
        if chosen is None:
            plan.fail_window = k
            return plan
        o, p, j = chosen
        plan.pairs.append((o, p))
        plan.fallbacks.append(j)
        state_o, state_p = outer[o], outer[p]
        outer[o] = ObjectState(name=state_o.name, p=state_p.p.copy(), q=state_p.q.copy(), shapes=state_o.shapes)
        outer[p] = ObjectState(name=state_p.name, p=state_o.p.copy(), q=state_o.q.copy(), shapes=state_p.shapes)
    plan.ok = True
    plan.final_xy = [[float(v) for v in state.p[:2]] for state in outer]
    return plan


def verify_distractor_swap_plan(
    layout: DistractorLayout,
    pairs: Sequence[Sequence[int]],
    windows: Sequence[InnerWindow],
    *,
    cfg: DistractorSwapConfig,
    cube_half_size: float,
    buttons_xy: Sequence[Sequence[float]] = (),
) -> list[str]:
    """Independently re-check a set of outer swap pairs (without re-choosing partners): window count, p of each window is indeed o's nearest neighbor, all four feasibility conditions hold. Returns violations."""
    problems: list[str] = []
    if len(pairs) != len(windows):
        problems.append(f"outer swap count {len(pairs)} != inner window count {len(windows)} (mismatch)")
    shapes = padded_bin_shapes(cube_half_size, cfg.plan_pad_m)
    outer = [distractor_bin_state(i, x, y, yaw, cube_half_size, shapes) for i, (x, y, yaw) in enumerate(layout.bins)]
    for k, (window, pair) in enumerate(zip(windows, pairs)):
        o, p = (int(v) for v in pair)
        if not (0 <= o < len(outer) and 0 <= p < len(outer)) or o == p:
            problems.append(f"window {k} outer pair {pair} out of range or duplicated")
            continue
        if nearest_distractor([state.p[:2] for state in outer], o) != p:
            problems.append(f"window {k}: distractor_bin_{p} is not the nearest neighbor of distractor_bin_{o} (partner mismatch)")
        ok, reason, rejection = evaluate_outer_candidate(k, window, o, p, outer, cfg=cfg, cube_half_size=cube_half_size,
                                                         buttons_xy=buttons_xy)
        if not ok:
            problems.append(f"window {k} outer pair ({o},{p}) infeasible: {reason}"
                            + ("" if rejection is None else f" {rejection.summary()}"))
        state_o, state_p = outer[o], outer[p]
        outer[o] = ObjectState(name=state_o.name, p=state_p.p.copy(), q=state_p.q.copy(), shapes=state_o.shapes)
        outer[p] = ObjectState(name=state_p.name, p=state_o.p.copy(), q=state_o.q.copy(), shapes=state_p.shapes)
    return problems


# -- reset entry (shared by both envs; still called by _spawn_xhard_distractors, still the last statement of _load_scene) ----------
@dataclass
class SwapDistractorResult:
    """Products of reset planning and actor building, attached to instance attributes by the env."""

    bins: list
    cubes: list
    cube_bin_pairs: list
    layout: DistractorLayout
    pairs: list[tuple[int, int]]
    predicted_inner_pairs: list[tuple[int, int]]
    timing: dict[str, float]
    stats: dict[str, int]


def plan_swap_distractors(
    *,
    windows: Sequence[InnerWindow],
    obstacles: Sequence[Any],
    buttons_xy: Sequence[Sequence[float]],
    generator: torch.Generator,
    recorder,
    distractor_cfg: dict,
    swap_cfg: dict,
    cube_half_size: float,
    guard_windows: Sequence[InnerWindow] | None = None,
    difficulty: str = "xhard4",
) -> tuple[DistractorLayout, list[tuple[int, int]], dict[str, float], dict[str, int]]:
    """Pure-geometry reset planning (no actors built, convenient for offline recomputation and unit tests): L20 prejudgment -> whole-layout resampled placement + outer planning -> recording.

    ``obstacles`` is a list of 2D OBBs (output of ``unmask_distractor_sampler.obstacle_obbs``). Returns
    ``(accepted layout, outer pairs per window, wall clock, counts)``; raises ``SceneGenerationError`` if the prejudgment rejects or all 16 attempts are infeasible.
    For recording discipline see :func:`spawn_swap_distractors_v5`.
    """
    dcfg = parse_distractor_cfg(distractor_cfg)
    scfg = parse_distractor_swap_cfg(swap_cfg)
    chs = float(cube_half_size)
    buttons_xy = [np.asarray(b, dtype=np.float64)[:2] for b in buttons_xy]
    if scfg.min_button_center_dist_m is not None and not buttons_xy:
        raise ValueError("distractor_swap declares a button center distance, but no button was passed")
    timing: dict[str, float] = {}
    stats: dict[str, int] = {}

    t0 = time.perf_counter()
    recorder.record("actions.predicted_inner_swap_pairs", [[int(w.a), int(w.b)] for w in windows])
    prejudge = prejudge_inner_windows(windows)
    timing["inner_prejudge_s"] = time.perf_counter() - t0
    if prejudge is not None:
        k, rejection = prejudge
        recorder.record("layout.inner_sweep_prejudge", {"window": int(k), "rejection": rejection.as_dict()})
        raise SceneGenerationError(f"xhard inner-vs-inner sweep rejected in reset prejudgment (L20): {rejection.summary()}")

    if scfg.rule == OUTER_BALANCED_RULE:
        # V6: outer O4 + post-placement index reshuffle, split out into _plan_swap_distractors_balanced (the V5 branch below is untouched word for word)
        return _plan_swap_distractors_balanced(
            windows=windows, guard_windows=guard_windows, obstacles=obstacles, buttons_xy=buttons_xy,
            generator=generator, recorder=recorder, dcfg=dcfg, scfg=scfg, chs=chs, timing=timing, stats=stats,
            difficulty=difficulty,
        )

    t0 = time.perf_counter()
    guard = InnerSweepGuard(windows if guard_windows is None else guard_windows)
    pad_shapes = padded_bin_shapes(chs, scfg.plan_pad_m)
    attempt_reasons: list[dict[str, int]] = []

    def extra_reject(i, x, y, yaw, _placed):
        # H1: candidates with the plan_pad_m margin are rejected if they intersect any window's inner sweep; draws no random numbers (also called during replay re-check)
        return guard.first_rejection(distractor_bin_state(i, x, y, yaw, chs, pad_shapes)) is not None

    def accept(layout: DistractorLayout):
        if not scfg.enabled:
            return True, None
        perm = torch.randperm(layout.count, generator=generator).tolist()  # appended after placement and cube sampling
        plan = plan_distractor_swaps(layout, perm, windows, cfg=scfg, cube_half_size=chs, buttons_xy=buttons_xy,
                                     stats=stats)
        attempt_reasons.append(dict(plan.reasons))
        return (True, plan) if plan.ok else (False, f"window_{plan.fail_window}")

    layout, plan = resample_distractor_layout(
        dcfg, obstacles=obstacles, generator=generator, cube_half_size=chs, recorder=recorder,
        accept=accept, max_attempts=scfg.layout_max_attempts, extra_reject=extra_reject,
    )
    pairs: list[tuple[int, int]] = []
    if scfg.enabled:
        order = [int(v) for v in recorder.value("objects.distractors.swap_order", list(plan.perm),
                                                decision_key="xhard.distractor_swap.initiator_rule")]
        if order != list(plan.perm):
            # can only happen during replay: frozen permutation differs from the resample ⇒ replan with the frozen permutation and re-check (N17)
            plan = plan_distractor_swaps(layout, order, windows, cfg=scfg, cube_half_size=chs, buttons_xy=buttons_xy)
            if not plan.ok:
                raise SceneGenerationError(f"replay re-check: frozen outer initiator permutation {order} is infeasible at window {plan.fail_window} (replay mismatch)")
        pairs = [(int(o), int(p)) for o, p in plan.pairs]
        recorder.record("actions.distractor_swap_pairs", [[o, p] for o, p in pairs])
        recorder.record("actions.distractor_swap_fallback", [int(v) for v in plan.fallbacks])
    timing["layout_and_plan_s"] = time.perf_counter() - t0
    stats["candidate_rejects"] = sum(sum(r.values()) for r in attempt_reasons)
    for reasons in attempt_reasons:
        for key, value in reasons.items():
            stats[f"reject_{key}"] = stats.get(f"reject_{key}", 0) + int(value)
    return layout, pairs, timing, stats


def spawn_swap_distractors_v5(
    env,
    *,
    generator: torch.Generator,
    partner_axes: Sequence[int],
    button_obbs: Sequence[Any] = (),
    hidden_half_size: float,
) -> SwapDistractorResult:
    """Distractor containers and outer swap planning for the new-value tiers of the two Swap envs (plan 2.5 pseudocode).

    Order: inner rehearsal -> L20 inner-vs-inner prejudgment (rejection raises a real ``SceneGenerationError``) -> whole-layout resampling with the unified sampler (after each placement
    ``perm = randperm(count)`` and plan all windows; resample if any window is entirely infeasible, at most ``layout_max_attempts`` times) -> only the accepted attempt goes through
    ``recorder.value`` (N18) -> outer initiator permutation ``value`` -> swap pairs and fallbacks ``record`` -> build actors.

    ``generator`` must be the independent stream ``distractor_generator(seed)``: the main stream draws nothing extra. When replaying a frozen spec the whole flow reruns,
    the frozen layout is re-checked via ``commit`` by the same rules (including the H1 callback, using the same inner rehearsal), and if the frozen initiator permutation differs from the resample, the frozen permutation
    is used to replan and re-check feasibility (N17); violations raise ``SceneGenerationError``.
    """
    decision = env._sampling["decision"][env.difficulty]
    chs = float(env.cube_half_size)
    dcfg = parse_distractor_cfg(decision["distractor"])
    inner_plan = getattr(env, "_newvalue_inner_plan", None)
    if inner_plan is not None:
        # V6 (2.2 inner S5): inner windows come from the sequence pre-planned at reset; the H1 guard covers all feasible slot pairs of G (R7)
        windows = planned_inner_windows(env)
        guard_windows = graph_guard_windows(env, inner_plan["graph"])
    else:
        windows = predict_inner_windows(env, partner_axes)
        guard_windows = None
    obstacles = obstacle_obbs(list(env.spawned_bins) + list(button_obbs), chs * dcfg.min_gap_factor)
    layout, pairs, timing, stats = plan_swap_distractors(
        windows=windows, obstacles=obstacles,
        buttons_xy=[np.asarray(c, dtype=np.float64)[:2] for c, _axes, _half in button_obbs],
        generator=generator, recorder=env._spec, distractor_cfg=decision["distractor"],
        swap_cfg=decision["distractor_swap"], cube_half_size=chs, guard_windows=guard_windows,
        difficulty=env.difficulty,
    )
    bins, cubes = build_distractor_actors(env, layout, hidden_half_size=hidden_half_size)
    return SwapDistractorResult(
        bins=bins, cubes=cubes, cube_bin_pairs=distractor_cube_bin_pairs(layout, bins, cubes), layout=layout,
        pairs=pairs, predicted_inner_pairs=[(w.a, w.b) for w in windows], timing=timing, stats=stats,
    )


# ════════════════════════════════════════════════════════════════════════════════
# V6 (NEWTASK_RELEASE_V6_PLAN 2.2 / 2.4, M5(b), M6(a), M7(a)): inner S5 + outer O4
# ════════════════════════════════════════════════════════════════════════════════
def v6_inner_swap_plan_cfg(task: str) -> dict:
    """Declared value of ``decision.<tier>.swap_plan_v6``: S5 (max->sum scoring), G connectivity as the reset acceptance condition."""
    if task not in V5_SWAP_TASKS:
        raise ValueError(f"only {V5_SWAP_TASKS} are supported, got {task!r}")
    return _su.inner_swap_plan_cfg(require_connected=True, score="max_sum")


def v6_distractor_swap_cfg(task: str) -> dict:
    """V6 declared value of ``decision.<tier>.distractor_swap``: only the three rule keys change (O4); path constraints and other values are reused from V5 item by item (M7(a))."""
    cfg = v5_distractor_swap_cfg(task)
    cfg["initiator_rule"] = OUTER_BALANCED_RULE
    cfg["fallback"] = OUTER_BALANCED_FALLBACK
    cfg["partner"] = copy.deepcopy(OUTER_BALANCED_PARTNER_RULE)
    return cfg


def inner_slot_graph(states: Sequence[ObjectState], *, stats: dict | None = None) -> list[list[bool]]:
    """Feasible slot-pair graph G of the 4 inner pose slots: run ``check_swap_sweep_prefiltered`` once per unordered slot pair (other inner containers static).

    Same check as the L20 prejudgment :func:`prejudge_inner_windows`; the slot set is fixed for the whole episode (each swap segment ends with an exact pose exchange), so only 6 checks are needed per episode.
    """
    states = list(states)

    def feasible(a: int, b: int) -> bool:
        others = [state for j, state in enumerate(states) if j not in (a, b)]
        _gap, rejection = check_swap_sweep_prefiltered(states[a], states[b], others, sweep_index=-1,
                                                       stage="inner_graph", stats=stats)
        return rejection is None

    return _su.slot_pair_graph(len(states), feasible)


def plan_inner_swaps_v6(env, generator: torch.Generator) -> dict:
    """Reset pre-planning of inner S5 for VUS/BUS xhard (plan 2.2 inner 1-3). Only called in xhard branches, after all existing main-stream sampling points.

    1. Build the states of the 4 slots from actual poses, compute G and ``record`` it into ``layout.inner_swap_graph``; if G is disconnected raise a real
       ``SceneGenerationError`` (candidate-level resampling, counted as a reset rejection);
    2. the main stream **appends one** ``objects.swap_plan_seed = randint(0, 2**62)`` (``value``); tie-breaking and resampling run on the local stream seeded by it;
    3. S5 plans the whole sequence, each swap ``value``-ed into ``actions.swap_pairs.<k> = {"initiator": "bin_a", "partner": "bin_b"}``;
       on replay the frozen values are used and re-checked per N17 ("each slot pair is feasible in G, no immediate undo"); violations raise ``EpisodeSpecError``;
    4. ``record`` ``objects.swap_plan`` = {counts, range, tries, undo} (the per-episode source of ``UNIFORM=REPORT``).

    Returns ``{"graph", "pairs", "summary"}``, attached by the env to ``_newvalue_inner_plan``.
    """
    from .episode_spec import EpisodeSpecError

    cfg = _su.parse_inner_swap_plan_cfg(env._sampling["decision"][env.difficulty]["swap_plan_v6"])
    bins = list(env.spawned_bins)
    states = [object_state_from_actor(actor, f"bin_{index}") for index, actor in enumerate(bins)]
    graph = inner_slot_graph(states)
    env._spec.record("layout.inner_swap_graph", [[int(a), int(b)] for a, b in _su.graph_edges(graph)])
    connected = _su.graph_connected(graph)
    env._spec.record("layout.inner_swap_graph_connected", bool(connected))
    if cfg["require_connected_graph"] and not connected:
        raise SceneGenerationError(
            f"{env.difficulty} inner feasible slot-pair graph is disconnected (feasible slot pairs {_su.graph_edges(graph)}); this layout is rejected per V6 S5"
        )
    seed = int(env._spec.value(
        "objects.swap_plan_seed",
        int(torch.randint(0, int(cfg["plan_seed_high_exclusive"]), (1,), generator=generator).item()),
        decision_key=f"{env.difficulty}.swap_plan_v6",
    ))
    local = torch.Generator()
    local.manual_seed(seed)
    plan = _su.plan_balanced_swaps(graph, int(env.swap_times), local, score=cfg["score"],
                                   budget=int(cfg["range_retry_budget"]), accept_range=int(cfg["accept_range"]),
                                   forbid_undo=True)
    if plan is None:
        raise SceneGenerationError(f"{env.difficulty} inner S5 cannot plan {env.swap_times} swaps (feasible slot pairs {_su.graph_edges(graph)})")
    pairs = []
    for k, (a, b) in enumerate(plan.pairs):
        chosen = env._spec.value(f"actions.swap_pairs.{k}", {"initiator": f"bin_{a}", "partner": f"bin_{b}"},
                                 decision_key=f"{env.difficulty}.swap_plan_v6")
        try:
            pairs.append((int(str(chosen["initiator"]).rsplit("_", 1)[1]), int(str(chosen["partner"]).rsplit("_", 1)[1])))
        except (TypeError, KeyError, ValueError, IndexError) as exc:
            raise EpisodeSpecError(f"{env.difficulty}: actions.swap_pairs.{k} has an invalid shape: {chosen!r}") from exc
    problems, stats = _su.verify_swap_sequence(graph, pairs, forbid_undo=True)
    if problems:
        raise EpisodeSpecError(f"{env.difficulty} inner swap sequence violates V6 S5 rules: " + "; ".join(problems))
    summary = stats.summary()
    summary["tries"] = int(plan.tries)
    env._spec.record("objects.swap_plan", summary)
    return {"graph": graph, "pairs": pairs, "summary": summary}


def planned_inner_windows(env) -> list[InnerWindow]:
    """Rehearse all inner windows along the S5 sequence pre-planned at reset (``env._newvalue_inner_plan["pairs"]``) (poses read from the actual actors)."""
    bins = list(env.spawned_bins)
    states = [object_state_from_actor(actor, f"bin_{index}") for index, actor in enumerate(bins)]
    windows: list[InnerWindow] = []
    for a, b in env._newvalue_inner_plan["pairs"]:
        windows.append(InnerWindow(a=int(a), b=int(b), states=tuple(states)))
        state_a, state_b = states[a], states[b]
        states[a] = ObjectState(name=state_a.name, p=state_b.p.copy(), q=state_b.q.copy(), shapes=state_a.shapes)
        states[b] = ObjectState(name=state_b.name, p=state_a.p.copy(), q=state_a.q.copy(), shapes=state_b.shapes)
    return windows


def graph_guard_windows(env, graph: Sequence[Sequence[bool]]) -> list[InnerWindow]:
    """Windows for the H1 guard: one window per feasible edge of G (swapping that slot pair from the initial slot state). Inner container boxes are identical, so the sweep of swapping a slot pair
    is the same at any time; hence the guard covers all feasible slot pairs in G (plan 2.2 inner 3, V6 investigation R7) and is a superset of the actual sequence."""
    bins = list(env.spawned_bins)
    states = tuple(object_state_from_actor(actor, f"bin_{index}") for index, actor in enumerate(bins))
    return [InnerWindow(a=int(a), b=int(b), states=states) for a, b in _su.graph_edges(graph)]


def relabel_distractor_layout(layout: DistractorLayout, perm: Sequence[int]) -> DistractorLayout:
    """Post-placement index reshuffle (cross-episode uniformity of V6 O4): the container with new index i = the container at placement order ``perm[i]``; the cube mapping is rewritten accordingly."""
    perm = [int(v) for v in perm]
    if sorted(perm) != list(range(layout.count)):
        raise ValueError(f"index reshuffle {perm} is not a permutation of 0..{layout.count - 1} (inclusive)")
    inverse = {old: new for new, old in enumerate(perm)}
    return DistractorLayout(
        bins=[tuple(layout.bins[old]) for old in perm],
        cube_count=int(layout.cube_count),
        cube_bins=[inverse[int(c)] for c in layout.cube_bins],
        color_order=list(layout.color_order),
        trials=[layout.trials[old] for old in perm] if len(layout.trials) == layout.count else list(layout.trials),
    )


def plan_distractor_swaps_balanced(
    layout: DistractorLayout,
    windows: Sequence[InnerWindow],
    generator: torch.Generator,
    *,
    cfg: DistractorSwapConfig,
    cube_half_size: float,
    buttons_xy: Sequence[Sequence[float]] = (),
    stats: dict | None = None,
) -> OuterSwapPlan:
    """V6 outer O4 (pure geometry + local-stream tie-breaking): per window, group all object pairs in ascending order of "max, then sum of the two participation counts", putting the pair used in the previous window last
    (immediate undo forbidden unless no other choice); if the first group has feasible pairs, one is drawn uniformly among them. The four checks are still :func:`evaluate_outer_candidate`.

    Under this rule ``OuterSwapPlan.fallbacks[k]`` records "which group was accepted in window k" (0 = minimum-score group).
    """
    count = layout.count
    shapes = padded_bin_shapes(cube_half_size, cfg.plan_pad_m)
    outer = [distractor_bin_state(i, x, y, yaw, cube_half_size, shapes) for i, (x, y, yaw) in enumerate(layout.bins)]
    plan = OuterSwapPlan(ok=False, perm=list(range(count)))
    cnt = [0] * count
    last = None
    for k, window in enumerate(windows):
        chosen = None
        for g, group in enumerate(_su.balanced_pair_groups(count, cnt, last)):
            feasible = []
            for o, p in group:
                ok, reason, _rejection = evaluate_outer_candidate(
                    k, window, o, p, outer, cfg=cfg, cube_half_size=cube_half_size, buttons_xy=buttons_xy, stats=stats,
                )
                if ok:
                    feasible.append((o, p))
                else:
                    plan.reasons[reason] = plan.reasons.get(reason, 0) + 1
            if feasible:
                o, p = feasible[int(torch.randint(0, len(feasible), (1,), generator=generator).item())]
                chosen = (o, p, g)
                break
        if chosen is None:
            plan.fail_window = k
            return plan
        o, p, g = chosen
        plan.pairs.append((o, p))
        plan.fallbacks.append(g)
        cnt[o] += 1
        cnt[p] += 1
        last = (min(o, p), max(o, p))
        state_o, state_p = outer[o], outer[p]
        outer[o] = ObjectState(name=state_o.name, p=state_p.p.copy(), q=state_p.q.copy(), shapes=state_o.shapes)
        outer[p] = ObjectState(name=state_p.name, p=state_o.p.copy(), q=state_o.q.copy(), shapes=state_p.shapes)
    plan.ok = True
    plan.final_xy = [[float(v) for v in state.p[:2]] for state in outer]
    return plan


def _plan_swap_distractors_balanced(*, windows, guard_windows, obstacles, buttons_xy, generator, recorder, dcfg, scfg,
                                    chs, timing, stats, difficulty):
    """V6 branch of :func:`plan_swap_distractors`: placement + H1 (all feasible slot pairs of G) -> after each placement the independent stream appends
    ``randperm(count)`` (index reshuffle) and a planning seed -> run O4 on the reshuffled layout -> only the accepted attempt goes through ``value``.

    Spec: the layout in placement order is still written to ``objects.distractors.*`` by ``commit_distractor_layout`` (replay re-checks in placement order, same criteria as V5);
    two new sampling points ``objects.distractors.label_perm`` / ``objects.distractors.swap_plan_seed``; the reshuffled
    public layout is ``record``-ed into ``objects.distractors.public``; actor building, swap pairs and runtime all use the public indices.
    """
    t0 = time.perf_counter()
    guard = InnerSweepGuard(windows if guard_windows is None else guard_windows)
    pad_shapes = padded_bin_shapes(chs, scfg.plan_pad_m)
    attempt_reasons: list[dict[str, int]] = []

    def extra_reject(i, x, y, yaw, _placed):
        return guard.first_rejection(distractor_bin_state(i, x, y, yaw, chs, pad_shapes)) is not None

    def plan_for(layout, perm, seed):
        local = torch.Generator()
        local.manual_seed(int(seed))
        public = relabel_distractor_layout(layout, perm)
        plan = plan_distractor_swaps_balanced(public, windows, local, cfg=scfg, cube_half_size=chs,
                                              buttons_xy=buttons_xy, stats=stats)
        plan.perm = [int(v) for v in perm]
        return public, plan

    def accept(layout: DistractorLayout):
        perm = torch.randperm(layout.count, generator=generator).tolist()  # appended after placement and cube sampling
        seed = int(torch.randint(0, 2 ** 62, (1,), generator=generator).item())
        _public, plan = plan_for(layout, perm, seed)
        attempt_reasons.append(dict(plan.reasons))
        return (True, (perm, seed, plan)) if plan.ok else (False, f"window_{plan.fail_window}")

    layout, payload = resample_distractor_layout(
        dcfg, obstacles=obstacles, generator=generator, cube_half_size=chs, recorder=recorder,
        accept=accept, max_attempts=scfg.layout_max_attempts, extra_reject=extra_reject,
    )
    perm, seed, plan = payload
    perm_v = [int(v) for v in recorder.value("objects.distractors.label_perm", list(perm),
                                             decision_key=f"{difficulty}.distractor_swap.partner.relabel")]
    seed_v = int(recorder.value("objects.distractors.swap_plan_seed", int(seed),
                                decision_key=f"{difficulty}.distractor_swap.initiator_rule"))
    public, replanned = plan_for(layout, perm_v, seed_v)
    if (perm_v, seed_v) != (list(perm), int(seed)):
        # can only happen during replay: frozen reshuffle / seed differs from the resample ⇒ replan with frozen values and re-check (N17)
        plan = replanned
        if not plan.ok:
            raise SceneGenerationError(f"replay re-check: frozen outer reshuffle / seed is infeasible at window {plan.fail_window}"
                                       f" (outer ring {public.count} containers, rejection reasons {dict(plan.reasons)})")
    recorder.record("objects.distractors.public", public.to_spec())
    pairs = [(int(o), int(p)) for o, p in plan.pairs]
    recorder.record("actions.distractor_swap_pairs", [[o, p] for o, p in pairs])
    recorder.record("actions.distractor_swap_fallback", [int(v) for v in plan.fallbacks])
    cnt = _su.participation(pairs, public.count)
    balance = {"counts": cnt, "unvisited": int(sum(1 for c in cnt if c == 0)),
               "range": int(max(cnt) - min(cnt)), "undo": int(_su.count_undo(pairs))}
    recorder.record("actions.distractor_swap_balance", balance)
    timing["layout_and_plan_s"] = time.perf_counter() - t0
    stats["candidate_rejects"] = sum(sum(r.values()) for r in attempt_reasons)
    for reasons in attempt_reasons:
        for key, value in reasons.items():
            stats[f"reject_{key}"] = stats.get(f"reject_{key}", 0) + int(value)
    stats["outer_unvisited"] = balance["unvisited"]
    stats["outer_undo"] = balance["undo"]
    return public, pairs, timing, stats


# -- Runtime ---------------------------------------------------------------------
def joint_sweep_from_actual(env, sweep_index: int, initiator, partner) -> tuple[float, CollisionRejection | None, dict]:
    """Joint re-check of two pairs at runtime window start (actual poses, real collision boxes, certified prefilter): the inner pair (resolved at runtime, L22 a) + this window's planned outer pair,
    with all other inner and distractor containers static. Returns ``(min check value, rejection evidence, attached info)``; the attached info says whether the inner pair matches the reset rehearsal."""
    inner = {
        index: object_state_from_actor(actor, f"bin_{index}")
        for index, actor in enumerate(env.spawned_bins)
        if actor is not None
    }
    a = env.spawned_bins.index(initiator)
    b = env.spawned_bins.index(partner)
    outer = [object_state_from_actor(actor, f"distractor_bin_{index}")
             for index, actor in enumerate(getattr(env, "distractor_bins", []))]
    planned = list(getattr(env, "distractor_swap_pairs", None) or [])
    moving_pairs = [(inner[a], inner[b])]
    moving_outer: tuple[int, ...] = ()
    if int(sweep_index) < len(planned):
        o, p = planned[int(sweep_index)]
        moving_pairs.append((outer[o], outer[p]))
        moving_outer = (int(o), int(p))
    bystanders = [state for index, state in sorted(inner.items()) if index not in (a, b)]
    bystanders += [state for index, state in enumerate(outer) if index not in moving_outer]
    gap, rejection = check_multi_swap_sweep(moving_pairs, bystanders, sweep_index=sweep_index, stage="sweep")
    predicted = list(getattr(env, "predicted_inner_swap_pairs", None) or [])
    expected = tuple(predicted[int(sweep_index)]) if int(sweep_index) < len(predicted) else None
    info = {
        "inner_pair": [a, b],
        "outer_pair": list(moving_outer) if moving_outer else None,
        "predicted_inner_pair": None if expected is None else list(expected),
        "inner_partner_mismatch": expected is not None and expected != (a, b),
    }
    return gap, rejection, info


def run_outer_swaps(env, timestep) -> None:
    """Outer swap execution (called in ``step``, outside the AST-locked inner partner loop): window k shares the same ``[start, end]`` as inner window k,
    calls ``swap_flat_two_lane`` for the reset-planned ``(o, p)`` (lane 0.07, smoothstep, vertical), pinning the other distractor containers;
    the first step of the window ``record``s one runtime trace. Adds no control steps."""
    from .statechange import swap_flat_two_lane  # lazy import: pure-geometry unit tests need not load simulation dependencies

    pairs = list(getattr(env, "distractor_swap_pairs", None) or [])
    schedule = list(getattr(env, "swap_schedule", None) or [])
    bins = list(getattr(env, "distractor_bins", None) or [])
    step = int(timestep)
    for k, (o, p) in enumerate(pairs):
        if k >= len(schedule):
            break
        start, end = int(schedule[k][2]), int(schedule[k][3])
        if step == start:
            env._spec.record(f"actions.distractor_swap_windows.{k}",
                             {"initiator": int(o), "partner": int(p), "start_step": start, "end_step": end})
        swap_flat_two_lane(
            env,
            cube_a=bins[o],
            cube_b=bins[p],
            start_step=start,
            end_step=end,
            cur_step=step,
            lane_offset=LANE_OFFSET,
            smooth=True,
            keep_upright=True,
            other_cube=[actor for index, actor in enumerate(bins) if index not in (o, p)],
        )
