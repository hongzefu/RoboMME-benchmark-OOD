import copy
import json
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

#Robomme
import matplotlib.pyplot as plt

import random
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

from .utils.SceneGenerationError import SceneGenerationError
from .utils import *
# V5 L3 (following VideoPlaceOrder's K2 fix): the `from .utils import *` line above lets the same-named submodule
# `utils.SceneGenerationError` shadow the name `SceneGenerationError` (confirmed by import introspection), so in the original three tiers
# raise / except become TypeError (original three tiers kept as is per H2). xhard uses the alias below to get the real exception class.
from .utils.SceneGenerationError import SceneGenerationError as _RealSceneGenerationError
from .utils.subgoal_evaluate_func import static_check, is_static
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils import reset_panda
from .utils.difficulty import NEWVALUE_DIFFICULTIES, is_newvalue_difficulty, normalize_robomme_difficulty
from .utils.episode_spec import SpecRecorder
from .utils.sampling_config import assert_native_decision, split_sampling_config
from .utils.bin_collision import (
    BinCollisionError,
    SpecBindingError,
    check_bin_state,
    check_swap_sweep,
    nearest_partner_index,
    object_state_from_actor,
)
# V5 (plan 2.15, L48/L49/L54): sweep criterion and static obstacles used by xhard partner reset planning (S2b new API, called only on the xhard path)
from .utils.bin_collision import (
    ObjectState,
    button_base_state,
    check_swap_sweep_prefiltered,
    cube_actor_pose,
    cube_shape_specs,
)
# V5 N17: re-check planned partners when replaying a frozen spec, raising EpisodeSpecError on violation (underscore alias, not leaked via from .utils import *)
from .utils.episode_spec import EpisodeSpecError as _EpisodeSpecError

from ..logging_utils import logger


def _scene_gen_error(difficulty):
    """Select the scene-generation exception class by difficulty family.

    New-value tiers return the real ``SceneGenerationError`` (retryable task failure); original three tiers return this module's
    shadowed name ``SceneGenerationError`` as is (a submodule, so raise / except still give TypeError; behavior verbatim unchanged).
    Usage: ``raise _scene_gen_error(self.difficulty)("message")``, ``except _scene_gen_error(self.difficulty):``;
    code executed only on the new-value path uses ``_RealSceneGenerationError`` directly.
    """
    return _RealSceneGenerationError if is_newvalue_difficulty(difficulty) else SceneGenerationError


PICK_CUBE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""


# ── Original-value snapshot of the native sampling inputs (newtask-v2 10.0) ──────────────────────
# Same note as BinFill: this dict is both the default when no sampling_config is passed and the --extract-config extraction target.
# Note that np.random.seed(seed) in __init__ is the only process-level global seeding point in the repo; its position must not move.
NATIVE_SAMPLING = {
    "parameters": {
        "num_repeats": {
            "sampler": "torch.randint",
            "low": 1,
            "high_exclusive": 4,
            "shape": [1],
        },
        "hard_spawn_rounds": 5,
        "object_selection": {
            "easy_medium_target_count": 1,
            "hard_target_low": 0,
            "swap_remaining_count": 2,
        },
        "swap_selection": {
            "initiator_mapping": "target_then_permuted_remaining_spawned_indices",
            "remaining_selection": "randperm_without_target",
            "partner": {
                "selection": "nearest",
                "position_axes": [0, 1],
                "resolve_at": "swap_start",
                "exclude_self": True,
                "tie_break": "first_in_spawn_order",
            },
        },
    },
    "positions": {
        "button": {
            "center_xy": [-0.2, 0],
            "randomize": True,
            "randomize_range": [0.1, 0.1],
            "sampling_expression": "(torch.rand(2, generator=generator) - 0.5) * randomize_range",
            "scale": 1.5,
            "randomize_range_origin": "passed explicitly at the VideoRepick call site",
        },
        "easy_medium_cubes": {
            "region3_tri": [[-0.05, -0.1], [-0.05, 0.1], [0.1, 0]],
            "region3_line": [[0, -0.15], [0, 0.15], [0, 0]],
            "region4": [[-0.05, -0.1], [-0.05, 0.1], [0.1, 0.1], [0.1, -0.1]],
            "region3_choice": {
                "sampler": "torch.randint",
                "low": 0,
                "high_exclusive": 2,
                "order": ["region3_tri", "region3_line"],
            },
            "layout_rotation_range_rad": [0, 180],
            "region_half_size": 0.07,
            "random_yaw": True,
            "yaw_range_rad": [0, 6.283185307179586],
            "yaw_expression": "yaw_sample * 2 * np.pi",
            "rotation_center": [0, 0],
            "min_gap": "self.cube_half_size",
            "min_gap_value": 0.02,
            "include_existing": True,
            "include_goal": True,
            "include_flags_origin": "original value is the spawn_random_cube parameter default True; the original call site did not pass it, now passed explicitly by this snapshot",
            "region4_reachable": False,
        },
        "hard_cubes": {
            "region_center": [-0.1, 0],
            "region_half_size": [0.2, 0.25],
            "random_yaw": True,
            "yaw_range_rad": [0, 6.283185307179586],
            "yaw_expression": "yaw_sample * 2 * np.pi",
            "min_gap": "self.cube_half_size",
            "min_gap_value": 0.02,
            "include_existing": False,
            "include_goal": False,
        },
    },
}


from .utils.xhard import HSV_FLOOR_COLOR, cube_obb2d_exact, hsv_floor_rgb
from .utils import swap_uniform

# V4 xhard "block color arbitrary" (C2: all cubes in an episode still share one color, only the color value is arbitrary).
# Per user decision 2026-09-22 the gamut has saturation/value floors: any hue, S>=0.5, V>=0.4 (utils/xhard.py::HSV_FLOOR_COLOR),
# alpha fixed 1; the three ranges can be overridden via sampling_config's decision.xhard.block_color without changing source.
NEWVALUE_BLOCK_COLOR = {
    "policy": "same_color_hsv_floor",
    "sampler": "torch.rand",
    **copy.deepcopy(HSV_FLOOR_COLOR),
}


def _resolve_episode_spec(spec, task):
    """Prepare this instance's private copy of the fixed spec (new-value injection); see the same-named function in BinFill.

    When ``None`` is passed (no ``--episode-specs``), returns ``None``; every consumption point then takes the original random path,
    identical to pre-change behavior.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("episode_spec must be a dict")
    if spec.get("task") != task:
        raise ValueError(f"episode_spec is the spec for {spec.get('task')}, cannot be used for {task}")
    return copy.deepcopy(spec)


def _solve_hold_obj_xhard(env, planner, static_steps):
    """V4 xhard-only in-place wait: same semantics as ``solve_hold_obj(close=False)``, only narrowing the bare ``except`` to ``AttributeError``.

    The original ``utils/subgoal_planner_func.py::solve_hold_obj`` wraps ``planner.open_gripper()`` in a bare ``except:``,
    which swallows ``BinCollisionError`` raised in step; ``elapsed_steps`` then never advances, the loop never ends, and every iteration
    appends a rejection record to ``_runtime_checks``, eventually blowing up memory and crashing (measured locally rc=139). Existing defects of shared utility functions
    are not fixed in place per N12, so a separate xhard-only path is written here.
    """
    start_step = int(getattr(env, "elapsed_steps", 0))
    target_step = start_step + static_steps
    while int(getattr(env, "elapsed_steps", 0)) < target_step:
        try:
            planner.open_gripper()
        except AttributeError:
            pass
    return None


def _solve_hold_until_step_xhard(env, planner, target_step):
    """V6 review fix N11: the in-place wait of xhard swap segments now waits until **absolute step** ``target_step`` (same semantics as ``_solve_hold_obj_xhard``,
    only replacing "static_steps relative to the current step" with an absolute step).

    Root cause (review xhard4 ep3 seed 6900300: 12 swaps but only 11 static boundaries): the internal completion check of swap-segment static tasks,
    ``static_check(static_steps=50)``, counts from the step of the **internal** task switch, while the wait solver run by the demonstration wrapper waits 50 steps from the **recording-side** task switch
    (end of the previous segment's solver) plus about 4 steps of overhead; the recording side only adopts the internal task index at the end of each segment's solver, so the two clocks drift 4 steps per segment,
    and once the accumulated drift exceeds a segment length, some internal static task completes entirely between two adoptions and its boundary is swallowed.
    Fix: completion checks and wait solvers of the four new tiers' swap segments are pinned to the same absolute clock -- segment k ends at ``swap_schedule[k][3]`` -- so drift no longer accumulates.
    """
    target_step = int(target_step)
    while int(getattr(env, "elapsed_steps", 0)) < target_step:
        try:
            planner.open_gripper()
        except AttributeError:
            pass
    return None


def _cube_index_of(name):
    """Map ``bin_<i>`` in the spec back to spawn index ``i`` (VideoRepick cubes keep the bin_ prefix)."""
    return int(str(name).rsplit("_", 1)[1])


# ── V5 xhard: partner reset planning (plan 2.15, L47-L49, L54) ──────────────────────────────────
# The three pure functions/classes below are called only in xhard's _plan_swaps_xhard; the original three tiers never go through them.


def _xhard_slot_states(slots, cube_half, margin):
    """Nominal slots ``[(x, y, yaw), ...]`` -> cube states for the sweep criterion.

    Shapes use ``cube_shape_specs(cube_half + margin)`` (planning margin, L49's 5 mm); poses replicate
    ``spawn_random_cube``'s landing (center height still uses the real half size ``cube_half``, consistent with the P4 prototype).
    """
    shapes = cube_shape_specs(float(cube_half) + float(margin))
    states = []
    for index, (x, y, yaw) in enumerate(slots):
        p, q = cube_actor_pose((x, y), yaw, cube_half)
        states.append(ObjectState(name=f"slot_{index}", p=p, q=q, shapes=shapes))
    return states


class _XhardSlotSweepFeasibility:
    """Swap sweep feasibility of nominal slot pairs, cached per unordered slot pair.

    At the end of ``swap_flat_two_lane`` the two cubes exchange xy and orientation, so the slot set is invariant over the episode and only occupants permute among slots;
    the curved paths of the two cubes depend only on the unordered slot pair (same paths regardless of who initiates), so feasibility is cached per unordered pair.
    bystander = the other slots (also inflated by the planning margin) + static obstacles (L54's button bases).
    The criterion uses S2b's ``check_swap_sweep_prefiltered`` (same decision as ``check_swap_sweep``, only adding a certified prefilter for speed).
    """

    def __init__(self, slot_states, statics=()):
        self.states = list(slot_states)
        self.statics = list(statics)
        self.cache = {}
        self.evidence = {}

    def feasible(self, slot_a, slot_b):
        key = (min(slot_a, slot_b), max(slot_a, slot_b))
        if key not in self.cache:
            bystanders = [state for index, state in enumerate(self.states) if index not in key] + self.statics
            _gap, rejection = check_swap_sweep_prefiltered(
                self.states[key[0]], self.states[key[1]], bystanders, stage="plan"
            )
            self.cache[key] = rejection is None
            self.evidence[key] = None if rejection is None else rejection.as_dict()
        return self.cache[key]


def _plan_swap_partners_xhard(slot_xy, seq, u, feasible, nearest_k, resolve=None):
    """Advance over nominal slots, planning a partner for initiator ``seq[k]`` each time (pure function, no random draws).

    * candidates = all slots other than the initiator's current slot, sorted by XY distance ascending (stable sort; ties by slot index);
    * if at least one of the nearest ``nearest_k`` is sweep-feasible => pool = the first ``nearest_k`` feasible ones in distance order;
      otherwise (fallback) => pool = all feasible ones;
    * ``u[k]`` picks one uniformly from the pool (index ``floor(u*len)``); an empty pool => raise a real ``SceneGenerationError``.
    * ``resolve(k, a, b)``: optional value hook returning the **actually adopted** partner cube index (the frozen value on replay);
      the actual partner must be another cube whose slot pair is sweep-feasible, otherwise raise ``EpisodeSpecError`` (N17).
    * after the nominal swap, continue with the next (occupancy updated accordingly).

    Returns per-swap records ``[{"initiator", "partner", "slot_a", "slot_b", "dist_m", "pool", "fallback"}, ...]``.
    """
    points = np.asarray(slot_xy, dtype=np.float64).reshape(-1, 2)
    count = len(points)
    occupant = list(range(count))  # occupant[slot] = cube index
    slot_of = list(range(count))   # slot_of[cube index] = slot
    plan = []
    for k, initiator in enumerate(seq):
        slot_a = slot_of[initiator]
        dist = np.linalg.norm(points - points[slot_a], axis=1)
        dist[slot_a] = np.inf
        order = [int(j) for j in np.argsort(dist, kind="stable")[: count - 1]]
        fallback = not any(feasible(slot_a, c) for c in order[:nearest_k])
        if not fallback:
            pool = []
            for c in order:
                if feasible(slot_a, c):
                    pool.append(c)
                    if len(pool) == nearest_k:
                        break
        else:
            pool = [c for c in order[nearest_k:] if feasible(slot_a, c)]
        if not pool:
            raise _RealSceneGenerationError(
                f"xhard: swap {k} initiator bin_{initiator} (slot {slot_a}) has no sweep-feasible partner"
            )
        slot_b = pool[min(int(float(u[k]) * len(pool)), len(pool) - 1)]
        partner = occupant[slot_b]
        if resolve is not None:
            partner = int(resolve(k, initiator, partner))
            if not 0 <= partner < count or partner == initiator:
                raise _EpisodeSpecError(f"xhard: swap {k} frozen partner bin_{partner} is invalid (initiator bin_{initiator})")
            slot_b = slot_of[partner]
            if not feasible(slot_a, slot_b):
                raise _EpisodeSpecError(
                    f"xhard: swap {k} frozen partner bin_{partner} with initiator bin_{initiator} has an infeasible sweep under the planning criteria"
                )
        plan.append({
            "initiator": initiator, "partner": partner, "slot_a": slot_a, "slot_b": slot_b,
            "dist_m": float(dist[slot_b]), "pool": len(pool), "fallback": fallback,
        })
        occupant[slot_a], occupant[slot_b] = partner, initiator
        slot_of[initiator], slot_of[partner] = slot_b, slot_a
    return plan


def native_blocks(cls, *, release="newtask-v6"):
    """Original ``(decision, native)`` blocks of this env; shared by external export and internal parsing, so there is only one source of truth."""
    native = copy.deepcopy(NATIVE_SAMPLING)
    if release in ("newtask-v4", "newtask-v5"):
        legacy_configs = {
            "easy": cls.configs["easy"], "medium": cls.configs["medium"], "hard": cls.configs["hard"],
            "xhard": {
                "cube": 6, "swap_min": 8, "swap_max": 12,
                "num_repeats_low": 4, "num_repeats_high_exclusive": 7,
                "layout_mode": "clutter", "region_center": [-0.1, 0.0],
                "region_half_size": [0.2, 0.25], "min_center_dist_m": 0.12,
                "partner_nearest_k": 3, "partner_sweep_margin_m": 0.005,
                "partner_button_obstacle": True,
            },
        }
        return _legacy_decision(legacy_configs, release), native
    # newtask-v7: same parsing path as v6, using current class constants (i.e. V7 fixed values; v6 values live only in the packaged v6 spec header, 0928 plan R3)
    if release not in ("newtask-v6", "newtask-v7"):
        raise ValueError(f"VideoRepick does not support sampling_config release {release!r}")
    native["parameters"]["configs"] = copy.deepcopy(cls.configs)
    return _native_decision(cls), native


def _legacy_decision(configs, release):
    decision = {
        "layout_mode": "native_by_difficulty",
        "num_repeats_range": {
            "low": NATIVE_SAMPLING["parameters"]["num_repeats"]["low"],
            "high_exclusive": NATIVE_SAMPLING["parameters"]["num_repeats"]["high_exclusive"],
        },
        "block_color_policy": "native_by_difficulty",
        "swap": {difficulty: {"swap_min": cfg["swap_min"], "swap_max": cfg["swap_max"]}
                 for difficulty, cfg in configs.items()},
    }
    xhard = configs["xhard"]
    decision["num_repeats_range"]["xhard"] = {
        "low": xhard["num_repeats_low"], "high_exclusive": xhard["num_repeats_high_exclusive"],
    }
    decision["xhard"] = {
        "layout": {
            "mode": xhard["layout_mode"], "cube_count": xhard["cube"],
            "region_center": list(xhard["region_center"]),
            "region_half_size": list(xhard["region_half_size"]),
        },
        "block_color": copy.deepcopy(NEWVALUE_BLOCK_COLOR),
    }
    if release == "newtask-v5":
        decision["xhard"]["layout"]["min_center_dist_m"] = xhard["min_center_dist_m"]
        decision["xhard"]["swap_plan"] = {
            "initiator_rule": "target_then_randperm_k_mod_cube_count",
            "partner_rule": "reset_plan_nearest_feasible",
            "nearest_k": xhard["partner_nearest_k"],
            "sweep_margin_m": xhard["partner_sweep_margin_m"],
            "button_obstacle": xhard["partner_button_obstacle"],
        }
    return decision


def _native_decision(cls):
    """Slice the decision block per the section-2 field table (equals the original in the original-value stage)."""
    # Section 2.10: decision covers layout mode, range of repeated pick-and-place counts, per-cube color policy, and whether/how many swaps.
    # In the original-value stage everything follows the original rules: layout mode keeps the difficulty's own anchors/region; counts and swap counts come from configs.
    # V6: each of the four new-value tiers has its own decision subtree; the part visible to the original three tiers is unchanged.
    decision = {
        "layout_mode": "native_by_difficulty",
        "num_repeats_range": {
            "low": NATIVE_SAMPLING["parameters"]["num_repeats"]["low"],
            "high_exclusive": NATIVE_SAMPLING["parameters"]["num_repeats"]["high_exclusive"],
        },
        "block_color_policy": "native_by_difficulty",
        "swap": {
            difficulty: {"swap_min": cfg["swap_min"], "swap_max": cfg["swap_max"]}
            for difficulty, cfg in cls.configs.items()
        },
    }
    for tier in NEWVALUE_DIFFICULTIES:
        tier_cfg = cls.configs[tier]
        # Original three tiers still read native.parameters.num_repeats; new-value tiers read a half-open repeat count interval per tier.
        decision["num_repeats_range"][tier] = {
            "low": tier_cfg["num_repeats_low"],
            "high_exclusive": tier_cfg["num_repeats_high_exclusive"],
        }
        decision[tier] = {
            "layout": {
                "mode": tier_cfg["layout_mode"],
                "cube_count": tier_cfg["cube"],
                "region_center": list(tier_cfg["region_center"]),
                "region_half_size": list(tier_cfg["region_half_size"]),
            },
            "block_color": copy.deepcopy(NEWVALUE_BLOCK_COLOR),
        }
        # S5 uses all feasible slot pairs; the 5 mm margin and button obstacle follow the original mechanism.
        decision[tier]["layout"]["min_center_dist_m"] = tier_cfg["min_center_dist_m"]
        decision[tier]["swap_plan"] = {
            # V6 (plan 2.5, M6(a) S5): initiator and partner are chosen together among "all sweep-feasible slot pairs" by a participation-balancing greedy
            # (compare the sum of both cubes' participation counts first, then the larger; ties drawn uniformly on the local stream, plus one more draw for who initiates), immediate swap-back forbidden,
            # reshuffle up to 20 times if the overall spread > 1; no longer limited to the nearest 3. If the feasibility graph has an isolated slot (that cube can never participate), raise SceneGenerationError for this episode,
            # the same failure criterion as V5's "some initiator has no feasible partner" (reset success rate not reduced).
            "initiator_rule": "s5_balanced_pair",
            "partner_rule": "s5_balanced_greedy",
            "s5": swap_uniform.inner_swap_plan_cfg(require_connected=False, score="sum_max"),
            # The following two items follow V5: +5 mm margin on cube half size during planning (L49); button bases as static bystanders (L54)
            "sweep_margin_m": tier_cfg["partner_sweep_margin_m"],
            "button_obstacle": tier_cfg["partner_button_obstacle"],
        }
    return decision


def _resolve_sampling_config(cls, override):
    """Prepare this instance's private copy of the sampling config; no sampling, no random stream change; see the same-named function in BinFill."""
    decision_default, native_default = native_blocks(cls, release="newtask-v6")
    decision, native = split_sampling_config(override, native_default, decision_default)
    # First round only exports/consumes original values: decision must equal the original key by key (red line R7).
    assert_native_decision(decision, decision_default, cls.__name__)
    resolved = native
    resolved["parameters"].setdefault("configs", copy.deepcopy(cls.configs))
    resolved["decision"] = decision
    for key in ("object_selection", "swap_selection"):
        if json.dumps(resolved["parameters"].get(key), sort_keys=True) != json.dumps(NATIVE_SAMPLING["parameters"][key], sort_keys=True):
            raise ValueError(f"VideoRepick.parameters.{key} must keep the original rules and types intact")
    return resolved


@register_env("VideoRepick", override=True)
class VideoRepick(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[Panda]
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)
    config_easy = {
        "cube":3,
        "swap_min":1,
        "swap_max":2,
    }
    config_medium= {
        "cube":3,
        "swap_min":2,
        "swap_max":3,
    }
    config_hard = {
        "cluster":True,
        "swap":None,
        "swap_min":0,
        "swap_max":0,
    }
    # V6 new-value tiers: cube count, swap count and repick count tiered per the final table; layout, color, center distance and S5 mechanism shared.
    config_xhard1 = {
        "cube": 4, "swap_min": 4, "swap_max": 4,  # V7 fixed values: swap 4/6/8/10, repick 2/3/4/5 (0928 plan 3.2.2)
        "num_repeats_low": 2, "num_repeats_high_exclusive": 3,
        "layout_mode": "clutter",
        "region_center": [-0.1, 0.0],
        "region_half_size": [0.2, 0.25],
        "min_center_dist_m": 0.12,
        "partner_sweep_margin_m": 0.005,
        "partner_button_obstacle": True,
    }
    config_xhard2 = {
        "cube": 5, "swap_min": 6, "swap_max": 6,
        "num_repeats_low": 3, "num_repeats_high_exclusive": 4,
        "layout_mode": "clutter", "region_center": [-0.1, 0.0], "region_half_size": [0.2, 0.25],
        "min_center_dist_m": 0.12, "partner_sweep_margin_m": 0.005, "partner_button_obstacle": True,
    }
    config_xhard3 = {
        "cube": 6, "swap_min": 8, "swap_max": 8,
        "num_repeats_low": 4, "num_repeats_high_exclusive": 5,
        "layout_mode": "clutter", "region_center": [-0.1, 0.0], "region_half_size": [0.2, 0.25],
        "min_center_dist_m": 0.12, "partner_sweep_margin_m": 0.005, "partner_button_obstacle": True,
    }
    config_xhard4 = {
        "cube": 7, "swap_min": 10, "swap_max": 10,
        "num_repeats_low": 5, "num_repeats_high_exclusive": 6,
        "layout_mode": "clutter", "region_center": [-0.1, 0.0], "region_half_size": [0.2, 0.25],
        "min_center_dist_m": 0.12, "partner_sweep_margin_m": 0.005, "partner_button_obstacle": True,
    }


    # Combine into a dictionary
    configs = {
        'hard': config_hard,
        'easy': config_easy,
        'medium': config_medium,
        'xhard1': config_xhard1,
        'xhard2': config_xhard2,
        'xhard3': config_xhard3,
        'xhard4': config_xhard4,
    }


    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     sampling_config=None,
                     episode_spec=None,
                     native_episode_spec=None,
                     **kwargs):
        # Must happen before any RNG call (including np.random.seed) and before super().__init__()
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._episode_spec = _resolve_episode_spec(episode_spec, "VideoRepick")
        self._spec = SpecRecorder(native_episode_spec, "VideoRepick", {"seed": seed},
                                  difficulty=kwargs.get("difficulty"))
        # Initialization index starts at -1; _initialize_episode increments it on each entry;
        # value points in _load_scene use index-free paths, so this is only a fallback.
        self._native_init_index = -1
        self._injection_evidence = {}
        self._runtime_checks = []
        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        self.seed = seed
        self.robomme_failure_recovery = bool(
            kwargs.pop("robomme_failure_recovery", False)
        )
        self.robomme_failure_recovery_mode = kwargs.pop(
            "robomme_failure_recovery_mode", None
        )
        if isinstance(self.robomme_failure_recovery_mode, str):
            self.robomme_failure_recovery_mode = (
                self.robomme_failure_recovery_mode.lower()
            )

        np.random.seed(seed)
        normalized_robomme_difficulty = normalize_robomme_difficulty(
            kwargs.pop("difficulty", None)
        )
        if normalized_robomme_difficulty is not None:
            self.difficulty = normalized_robomme_difficulty
        else:
            seed_mod = seed % 3
            if seed_mod == 0:
                self.difficulty = "easy"
            elif seed_mod == 1:
                self.difficulty = "medium"
            else:  # seed_mod == 2
                self.difficulty = "hard"
        #self.difficulty = "hard"
        # Use seed to randomly determine number of repetitions (1-5)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        if is_newvalue_difficulty(self.difficulty) and self._episode_spec is not None:
            # New-value specs contain per-tier clutter layouts and S5 swap plans; channel A's original three-cube format cannot be reused.
            raise ValueError("VideoRepick new-value tiers do not accept channel A's episode_spec; use native_episode_spec")
        repeats_cfg = self._sampling["parameters"]["num_repeats"]
        if self._episode_spec is None and is_newvalue_difficulty(self.difficulty):
            # Repick count drawn from the current new-value tier's half-open interval.
            tier_repeats = self._sampling["decision"]["num_repeats_range"][self.difficulty]
            self.num_repeats = self._spec.value(
                "objects.num_repeats",
                torch.randint(tier_repeats["low"], tier_repeats["high_exclusive"], tuple(repeats_cfg["shape"]), generator=self.generator).item(),
                decision_key=f"num_repeats_range.{self.difficulty}",
            )
        elif self._episode_spec is None:
            self.num_repeats = torch.randint(repeats_cfg["low"], repeats_cfg["high_exclusive"], tuple(repeats_cfg["shape"]), generator=self.generator).item()
        else:
            self.num_repeats = int(self._episode_spec["objects"]["num_repeats"])
        logger.debug(f"Task will repeat {self.num_repeats} times (pickup-drop cycles)")

        difficulty_cfg = self._sampling["parameters"]["configs"][self.difficulty]
        if self._episode_spec is None and is_newvalue_difficulty(self.difficulty):
            # Swap count drawn from the decision range of the current difficulty tier.
            tier_swap = self._sampling["decision"]["swap"][self.difficulty]
            self.swap_times = self._spec.value(
                "objects.n_swaps",
                torch.randint(tier_swap["swap_min"], tier_swap["swap_max"] + 1, (1,), generator=self.generator).item(),
                decision_key=f"swap.{self.difficulty}",
            )
        elif self._episode_spec is None:
            self.swap_times = self._spec.value(
                "objects.n_swaps",
                torch.randint(difficulty_cfg['swap_min'], difficulty_cfg['swap_max']+1, (1,), generator=self.generator).item(),
            )
        else:
            self.swap_times = int(self._episode_spec["objects"]["n_swaps"])
        logger.debug(f"Task will swap {self.swap_times} times")


        self.static_flag=False
        self.start_step=99999
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        camera_eye=[0.3,0,0.4]
        camera_target =[0,0,-0.2]
        pose = sapien_utils.look_at(
            eye=camera_eye, target=camera_target
        )
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        try:
            self.table_scene = TableSceneBuilder(
                self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            self.table_scene.build()

            spec = self._episode_spec
            button_cfg = self._sampling["positions"]["button"]
            button_center = (
                tuple(button_cfg["center_xy"]) if spec is None else tuple(spec["layout"]["button_xy"])
            )
            button_obb_1 = build_button(
                self,
                center_xy=button_center,
                scale=button_cfg["scale"],
                generator=self.generator,
                name="button",
                # The spec gives the **final** center; with randomize off, build_button draws no random numbers
                randomize=button_cfg["randomize"] if spec is None else False,
                randomize_range=tuple(button_cfg["randomize_range"])
            )
            # Store first button before building second one
            self.button_left = self.button
            self.button_joint_1 = self.button_joint

            avoid = [button_obb_1]

            options = [
                {"color": (1, 0, 0, 1), "name": "red"},
                {"color": (0, 0, 1, 1), "name": "blue"},
                {"color": (0, 1, 0, 1), "name": "green"},
            ]
            if is_newvalue_difficulty(self.difficulty):
                # The four new-value tiers share the clutter + S5 branch; the original three tiers' branch and value points are unchanged.
                self._load_cubes_newvalue(avoid)
            elif self.difficulty == "hard":
                self.spawned_cubes = []

                hard_cfg = self._sampling["positions"]["hard_cubes"]
                for idx in range(self._sampling["parameters"]["hard_spawn_rounds"]):
                    shuffle_indices = self._spec.value(
                        f"objects.hard_round_order.{idx}",
                        torch.randperm(len(options), generator=self.generator).tolist(),
                    )
                    new_options = [options[i] for i in shuffle_indices]
                    for group in new_options:
                        try:
                            cube = spawn_random_cube(
                                self,
                                color=group["color"],
                                avoid=avoid,
                                include_existing=hard_cfg["include_existing"],
                                include_goal=hard_cfg["include_goal"],
                                region_center=list(hard_cfg["region_center"]),
                                region_half_size=list(hard_cfg["region_half_size"]),
                                half_size=self.cube_half_size,
                                min_gap=self.cube_half_size,
                                random_yaw=hard_cfg["random_yaw"],
                                name_prefix=f"cube_{group['name']}_{idx}",
                                generator=self.generator,
                            )
                        except RuntimeError as e:
                            raise SceneGenerationError(
                                f"Failed to generate {group['name']} cube {idx}"
                            ) from e

                        self.spawned_cubes.append(cube)
                        avoid.append(cube)

                if not self.spawned_cubes:
                    raise SceneGenerationError("Failed to generate any cube")

                selection_cfg = self._sampling["parameters"]["object_selection"]
                target_idx = self._spec.value(
                    "objects.target",
                    torch.randint(selection_cfg["hard_target_low"], len(self.spawned_cubes), (1,), generator=self.generator).item(),
                )
                logger.debug("target index: %s", target_idx)
                self.target_cube_1 = self.spawned_cubes[target_idx]

            else:
                if spec is None:
                    idx = self._spec.value(
                        "objects.color_idx",
                        torch.randint(0, len(options), (1,), generator=self.generator).item(),
                    )
                else:
                    # The three cubes share one color, fixed by the spec (original definition table order is red / blue / green)
                    idx = [item["name"] for item in options].index(spec["objects"]["color"])
                chosen_color = options[idx]["color"]

                cube_colors = [chosen_color] * 4
                if spec is None:
                    shuffle_indices = self._spec.value(
                        "objects.color_order",
                        torch.randperm(len(cube_colors), generator=self.generator).tolist(),
                    )
                    cube_colors = [cube_colors[i] for i in shuffle_indices]
                # All four elements are the same color, so shuffling makes no difference; the injection path skips this draw

                self.spawned_cubes = []

                plain_cfg = self._sampling["positions"]["easy_medium_cubes"]
                difficulty_cfg = self._sampling["parameters"]["configs"][self.difficulty]
                region4 = [list(point) for point in plain_cfg["region4"]]
                region3_tri = [list(point) for point in plain_cfg["region3_tri"]]
                region3_line = [list(point) for point in plain_cfg["region3_line"]]

                if spec is None:
                    choice_cfg = plain_cfg["region3_choice"]
                    region3_choice = torch.randint(choice_cfg["low"], choice_cfg["high_exclusive"], (1,), generator=self.generator).item()
                    region3 = region3_tri if region3_choice == 0 else region3_line

                    if difficulty_cfg['cube'] == 4:
                        region = region4
                    else:
                        region = region3
                    angle, region = rotate_points_random(region, tuple(plain_cfg["layout_rotation_range_rad"]), self.generator)
                else:
                    # The spec's cubes[i].xy is the final position (anchor rotation + offset were computed and passed
                    # collision checks before freezing), so the injection path no longer goes through rotate_points_random and draws no random numbers
                    angle = float(spec["layout"]["theta_rad"])
                    region = None

                for i in range(difficulty_cfg['cube']):
                    fixed_xy = fixed_yaw = None
                    if spec is not None:
                        entry = spec["layout"]["cubes"][i]
                        if entry["object_id"] != f"bin_{i}":
                            raise ValueError(f"spec cube {i} has object_id {entry['object_id']}, expected bin_{i}")
                        fixed_xy = [float(entry["xy"][0]), float(entry["xy"][1])]
                        fixed_yaw = float(entry["yaw_rad"])
                    try:
                        cube_actor = spawn_random_cube(
                            self,
                            avoid=avoid,
                            region_center=region[i] if region is not None else plain_cfg["region3_tri"][0],
                            region_half_size=plain_cfg["region_half_size"],
                            min_gap=self.cube_half_size * 1,
                            half_size=self.cube_half_size,
                            name_prefix=f"bin_{i}",
                            max_trials=256,
                            color=cube_colors[i],
                            random_yaw=plain_cfg["random_yaw"],
                            include_existing=plain_cfg["include_existing"],
                            include_goal=plain_cfg["include_goal"],
                            generator=self.generator,
                            fixed_xy=fixed_xy,
                            fixed_yaw=fixed_yaw,

                        )
                    except RuntimeError as e:
                        raise SceneGenerationError(f"Failed to generate bin_{i}") from e

                    self.spawned_cubes.append(cube_actor)
                    setattr(self, f"bin_{i}", cube_actor)
                    avoid.append(cube_actor)

                if not self.spawned_cubes:
                    raise SceneGenerationError("Failed to generate any bin")

                selection_cfg = self._sampling["parameters"]["object_selection"]
                if spec is None:
                    target_indices = torch.randperm(len(self.spawned_cubes), generator=self.generator)[:selection_cfg["easy_medium_target_count"]].tolist()
                else:
                    target_indices = [_cube_index_of(spec["objects"]["target"])]
                self.target_cube_1 = self.spawned_cubes[target_indices[0]]

                if self.difficulty != "hard":
                    remaining_indices = [i for i in range(len(self.spawned_cubes)) if i not in target_indices]
                    if len(remaining_indices) < 2:
                        raise SceneGenerationError("Not enough cubes for swapping")

                    if spec is None:
                        selected_remaining = self._spec.value(
                            "objects.swap_initiators_remaining",
                            torch.randperm(len(remaining_indices), generator=self.generator)[:selection_cfg["swap_remaining_count"]].tolist(),
                        )
                        selected_indices = [remaining_indices[i] for i in selected_remaining]
                    else:
                        # WARNING: the source unconditionally assigns swap_pair{1,2,3}_idx1, so the spec stores all 3 initiators:
                        # the first must be the target cube, the last two a permutation of the other two cubes
                        spec_initiators = [_cube_index_of(name) for name in spec["objects"]["swap_initiators"]]
                        if spec_initiators[0] != target_indices[0]:
                            raise ValueError(
                                f"spec's first swap initiator bin_{spec_initiators[0]} is not the target cube bin_{target_indices[0]}"
                            )
                        selected_indices = spec_initiators[1:]
                        if sorted(selected_indices) != sorted(remaining_indices):
                            raise ValueError(
                                f"spec's later initiators {selected_indices} and the remaining two cubes {remaining_indices} are not permutations of each other"
                            )
                    swap_indices = target_indices + selected_indices

                    self.swap_pair1_idx1 = self.spawned_cubes[swap_indices[0]]
                    self.swap_pair2_idx1 = self.spawned_cubes[swap_indices[1]]
                    self.swap_pair3_idx1 = self.spawned_cubes[swap_indices[2]]
                    self.swap_pair1_idx2 = None
                    self.swap_pair2_idx2 = None
                    self.swap_pair3_idx2 = None
                    # xhard (4-5 swaps): the k-th initiator cycles through the first 3 (a,b,c,a,b); with <= 3 swaps this loop does not run
                    for k in range(3, self.swap_times):
                        setattr(self, f"swap_pair{k+1}_idx1", self.spawned_cubes[swap_indices[k % 3]])
                        setattr(self, f"swap_pair{k+1}_idx2", None)
                    self._refresh_swap_schedule()

                if spec is not None:
                    # Read-only evidence: creation input vs actual post-creation actor pose, for INJECTION_BINDING checks
                    self._injection_evidence = {
                        "spec_sha256": spec.get("spec_sha256"),
                        "episode": spec.get("episode"),
                        "theta_rad": float(spec["layout"]["theta_rad"]),
                        "layout_type": spec["layout"]["type"],
                        "n_swaps": self.swap_times,
                        "num_repeats": self.num_repeats,
                        "color": spec["objects"]["color"],
                        "target_index": target_indices[0],
                        "button_xy": [float(v) for v in spec["layout"]["button_xy"]],
                        "cubes": [
                            {
                                "object_id": entry["object_id"],
                                "requested_xy": [float(v) for v in entry["xy"]],
                                "requested_yaw_rad": float(entry["yaw_rad"]),
                                "actual_p": [float(v) for v in self._get_actor_position(actor)[:3]],
                            }
                            for entry, actor in zip(spec["layout"]["cubes"], self.spawned_cubes)
                        ],
                    }
        except _scene_gen_error(self.difficulty):  # V5 L3: xhard uses the real class; original three tiers still use the shadowed original name
            raise
        except Exception as exc:
            if is_newvalue_difficulty(self.difficulty) and isinstance(exc, _EpisodeSpecError):
                # V5 N17: a replayed frozen spec violating geometry rules (minimum center distance, planned partners) is a spec/code error,
                # not wrapped as a retryable SceneGenerationError; original three tiers never enter this branch, behavior verbatim unchanged
                raise
            raise _scene_gen_error(self.difficulty)(
                f"Failed to load VideoRepick scene for seed {self.seed}"
            ) from exc

    def _load_cubes_newvalue(self, avoid):
        """New-value tier cube generation: clutter layout, one arbitrary color per episode, minimum center distance and S5 reset planning.

        Value order (original three tiers never come here): color -> per-cube poses -> target -> remaining cube order -> planning seed ->
        reset partner planning (no random draws) -> per-swap injection of ``actions.swap_pairs.<k>``.
        New values always take layout, color and S5 config from ``decision.<tier>``; criterion parameters follow the original values of hard's whole region;
        ``min_gap`` takes ``self.cube_half_size`` like the original three tiers.
        Every value point goes through ``self._spec``; a mismatch of "requested vs actual cube count" fails the episode directly (plan 2.2-4).

        V5 (plan 2.15):
        * L50: pairwise center distance of all cubes in this tier >= ``layout.min_center_dist_m``, checked inside the rejection loop via ``spawn_random_cube(min_center_dist=...)``
          (each trial still 3 rands); placed cubes use ``cube_obb2d_exact`` exact obstacles,
          ``include_existing=False`` (no longer the degenerating actor path). When replaying frozen poses the spawn function re-checks by the same rule (N17).
        * L47 a': ``objects.swap_initiators_remaining`` keeps the original value point and draws a permutation over this tier's remaining cube count; that permutation no longer decides S5 initiators.
        * S5: ``_plan_swaps_newvalue_v6`` plans over all sweep-feasible slot pairs.
        """
        tier_decision = self._sampling["decision"][self.difficulty]
        layout = tier_decision["layout"]
        if layout["mode"] != "clutter":
            raise ValueError(f"VideoRepick {self.difficulty} only implements the clutter layout, got {layout['mode']!r}")
        region_cfg = self._sampling["positions"]["hard_cubes"]
        # V5 L50: minimum center distance; the button OBB is the first element _load_scene puts into avoid (L54 planning needs the button's final center)
        min_center_dist = float(layout["min_center_dist_m"])
        button_obb = avoid[0] if avoid else None

        color_cfg = tier_decision["block_color"]
        u = torch.rand(3, generator=self.generator).tolist()
        rgb = self._spec.value(
            "objects.color_rgb",
            hsv_floor_rgb(u, color_cfg),
            decision_key=f"{self.difficulty}.block_color",
        )
        chosen_color = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)

        requested = int(layout["cube_count"])
        self.spawned_cubes = []
        placed = []  # Exact OBBs of placed cubes (go into both avoid and the reference point set for minimum center distance)
        for i in range(requested):
            try:
                cube_actor = spawn_random_cube(
                    self,
                    avoid=avoid,
                    region_center=list(layout["region_center"]),
                    region_half_size=list(layout["region_half_size"]),
                    min_gap=self.cube_half_size,
                    half_size=self.cube_half_size,
                    name_prefix=f"bin_{i}",
                    max_trials=256,
                    color=chosen_color,
                    random_yaw=region_cfg["random_yaw"],
                    include_existing=False,
                    include_goal=region_cfg["include_goal"],
                    generator=self.generator,
                    recorder=self._spec,
                    spec_path=f"layout.cubes.{i}.xy_yaw",
                    min_center_dist=(min_center_dist, placed),
                )
            except RuntimeError as e:
                raise _RealSceneGenerationError(f"{self.difficulty}: failed to generate bin_{i} of {requested}") from e
            self.spawned_cubes.append(cube_actor)
            setattr(self, f"bin_{i}", cube_actor)
            obb = cube_obb2d_exact(cube_actor, self.cube_half_size)
            placed.append(obb)
            avoid.append(obb)
        self._spec.record("objects.cube_count.requested", requested)
        self._spec.record("objects.cube_count.actual", len(self.spawned_cubes))
        if len(self.spawned_cubes) != requested:
            raise _RealSceneGenerationError(
                f"{self.difficulty}: requested {requested} cubes but spawned {len(self.spawned_cubes)}"
            )

        target_index = self._spec.value(
            "objects.target",
            int(torch.randint(0, len(self.spawned_cubes), (1,), generator=self.generator).item()),
        )
        self.target_cube_1 = self.spawned_cubes[target_index]
        remaining_indices = [i for i in range(len(self.spawned_cubes)) if i != target_index]
        # V5 L47 a': full initiation order of the remaining 5 cubes (the same randperm as V4, just no longer truncated to [:2])
        selected_remaining = self._spec.value(
            "objects.swap_initiators_remaining",
            torch.randperm(len(remaining_indices), generator=self.generator).tolist(),
        )
        if sorted(int(i) for i in selected_remaining) != list(range(len(remaining_indices))):
            # Replaying a V4 spec (length 2) or a tampered spec: V5 requires a full permutation of the remaining cubes (N17)
            raise _EpisodeSpecError(
                f"{self.difficulty}: objects.swap_initiators_remaining should be a full permutation of range({len(remaining_indices)}), "
                f"got {selected_remaining}"
            )
        # V6 (plan 2.5 S5): the V5 initiation order value point above is still drawn (main stream position unchanged), but no longer decides initiators;
        # V5's objects.swap_partner_u (rand(n_swaps)) is replaced by one planning seed (tie-breaking and reshuffles use a local stream seeded by it)
        plan_seed = int(self._spec.value(
            "objects.swap_plan_seed",
            int(torch.randint(0, int(tier_decision["swap_plan"]["s5"]["plan_seed_high_exclusive"]),
                              (1,), generator=self.generator).item()),
            decision_key=f"{self.difficulty}.swap_plan",
        ))
        pairs = self._plan_swaps_newvalue_v6(plan_seed, button_obb)
        self._spec.record("objects.swap_initiators", [f"bin_{a}" for a, _b in pairs])
        for k, (a, _b) in enumerate(pairs):
            setattr(self, f"swap_pair{k+1}_idx1", self.spawned_cubes[a])
            setattr(self, f"swap_pair{k+1}_idx2", None)
        self._refresh_swap_schedule()

    def _plan_swaps_xhard(self, initiator_seq, partner_u, button_obb):
        """V5 xhard partner reset planning (plan 2.15, L48/L49/L54); called only in xhard's ``_load_cubes_xhard``.

        On the 6 nominal slots (poses of the cubes just spawned), filter swap sweep feasibility with ``cube_shape_specs(hs + sweep_margin_m)``
        (cached per unordered slot pair); when ``button_obstacle`` is true, the button base (``button_base_state``,
        from ``build_button``'s final center and scale) is a static bystander, and candidates pressing the button are infeasible (L54).
        For the k-th initiator, sort the other slots by distance and pick uniformly with ``u[k]`` among the feasible ones in the first ``nearest_k``; if none of the nearest
        ``nearest_k`` is feasible pick uniformly among any feasible one; if no partner is feasible at all raise a real ``SceneGenerationError`` (L3).

        Per swap, inject ``{"initiator": "bin_a", "partner": "bin_b"}`` into ``actions.swap_pairs.<k>`` via ``value``;
        on replay use frozen values and re-check "initiator matches the plan, partner is sweep-feasible" (N17). Results are stored in
        ``self._xhard_swap_partners`` (partner cube index of the k-th swap), from which ``step``'s xhard branch takes the partner.
        """
        swap_cfg = self._sampling["decision"]["xhard"]["swap_plan"]
        if swap_cfg["initiator_rule"] != "target_then_randperm_k_mod_cube_count":
            raise ValueError(f"VideoRepick xhard: unimplemented initiator rule {swap_cfg['initiator_rule']!r}")
        if swap_cfg["partner_rule"] != "reset_plan_nearest_feasible":
            raise ValueError(f"VideoRepick xhard: unimplemented partner rule {swap_cfg['partner_rule']!r}")
        nearest_k = swap_cfg["nearest_k"]
        if isinstance(nearest_k, bool) or not isinstance(nearest_k, int) or nearest_k < 1:
            raise ValueError(f"VideoRepick xhard swap_plan.nearest_k must be a positive integer, got {nearest_k!r}")
        margin = float(swap_cfg["sweep_margin_m"])
        if not margin >= 0.0:
            raise ValueError(f"VideoRepick xhard swap_plan.sweep_margin_m must not be negative, got {margin}")
        half = float(self.cube_half_size)

        slots = []
        for cube in self.spawned_cubes:
            c, axes, _h = cube_obb2d_exact(cube, half)
            slots.append((float(c[0]), float(c[1]), float(np.arctan2(axes[1, 0], axes[0, 0]))))
        statics = []
        if swap_cfg["button_obstacle"]:
            if button_obb is None:
                raise ValueError("VideoRepick xhard: planning needs the button base as a static obstacle, but avoid has no button OBB")
            statics.append(button_base_state(
                "button_base", button_obb[0], scale=float(self._sampling["positions"]["button"]["scale"])
            ))
        feasibility = _XhardSlotSweepFeasibility(_xhard_slot_states(slots, half, margin), statics)

        def resolve(k, initiator, partner):
            chosen = self._spec.value(
                f"actions.swap_pairs.{k}",
                {"initiator": f"bin_{initiator}", "partner": f"bin_{partner}"},
                decision_key="xhard.swap_plan",
            )
            if not isinstance(chosen, dict) or chosen.get("initiator") != f"bin_{initiator}":
                raise _EpisodeSpecError(
                    f"xhard: actions.swap_pairs.{k} initiator {chosen!r} and the planned bin_{initiator} do not match"
                )
            return _cube_index_of(chosen["partner"])

        plan = _plan_swap_partners_xhard(
            [slot[:2] for slot in slots], initiator_seq, partner_u, feasibility.feasible, nearest_k, resolve=resolve,
        )
        self._xhard_swap_partners = [step["partner"] for step in plan]
        # Read-only diagnostics (not in the spec): per-swap path length, candidate pool size, fallbacks, and the slot pairs checked
        self._xhard_swap_plan_info = {
            "slots": slots,
            "plan": plan,
            "checked_slot_pairs": len(feasibility.cache),
            "infeasible_slot_pairs": sorted(key for key, ok in feasibility.cache.items() if not ok),
        }
        return plan

    def _plan_swaps_newvalue_v6(self, plan_seed, button_obb):
        """V6 four-tier swap sequence reset planning (plan 2.5, M6(a) S5); called only in ``_load_cubes_newvalue``.

        Slot feasibility uses the same criteria as V5 (``_XhardSlotSweepFeasibility``: half size + ``sweep_margin_m`` margin, button base as a static obstacle),
        computing all C(n,2) slot pairs of the current tier at once to get the feasibility graph M; an isolated slot in M raises a real ``SceneGenerationError``. Run S5 on M
        (``swap_uniform.plan_balanced_swaps``, local stream seeded by ``plan_seed``), injecting each swap via ``value`` into
        ``actions.swap_pairs.<k>``; on replay use frozen values and re-check "slot pair feasible, no immediate swap-back" (N17, raising ``EpisodeSpecError`` on violation).
        ``record`` ``objects.swap_plan`` = {counts, range, tries, undo}. Returns ``[(initiator, partner), ...]``.
        """
        swap_cfg = self._sampling["decision"][self.difficulty]["swap_plan"]
        if swap_cfg["initiator_rule"] != "s5_balanced_pair" or swap_cfg["partner_rule"] != "s5_balanced_greedy":
            raise ValueError(f"VideoRepick {self.difficulty}: unimplemented swap rule {swap_cfg['initiator_rule']!r}/{swap_cfg['partner_rule']!r}")
        s5 = swap_uniform.parse_inner_swap_plan_cfg(swap_cfg["s5"])
        margin = float(swap_cfg["sweep_margin_m"])
        if not margin >= 0.0:
            raise ValueError(f"VideoRepick {self.difficulty} swap_plan.sweep_margin_m must not be negative, got {margin}")
        half = float(self.cube_half_size)
        slots = []
        for cube in self.spawned_cubes:
            c, axes, _h = cube_obb2d_exact(cube, half)
            slots.append((float(c[0]), float(c[1]), float(np.arctan2(axes[1, 0], axes[0, 0]))))
        statics = []
        if swap_cfg["button_obstacle"]:
            if button_obb is None:
                raise ValueError(f"VideoRepick {self.difficulty}: planning needs the button base as a static obstacle, but avoid has no button OBB")
            statics.append(button_base_state(
                "button_base", button_obb[0], scale=float(self._sampling["positions"]["button"]["scale"])
            ))
        feasibility = _XhardSlotSweepFeasibility(_xhard_slot_states(slots, half, margin), statics)
        graph = swap_uniform.slot_pair_graph(len(slots), feasibility.feasible)
        self._spec.record("layout.swap_graph", [[int(a), int(b)] for a, b in swap_uniform.graph_edges(graph)])
        isolated = swap_uniform.isolated_slots(graph)
        if isolated or (s5["require_connected_graph"] and not swap_uniform.graph_connected(graph)):
            raise _RealSceneGenerationError(
                f"{self.difficulty}: slot {isolated} has no sweep-feasible partner (feasible slot pairs {swap_uniform.graph_edges(graph)})"
            )
        local = torch.Generator()
        local.manual_seed(int(plan_seed))
        plan = swap_uniform.plan_balanced_swaps(graph, int(self.swap_times), local, score=s5["score"],
                                                budget=int(s5["range_retry_budget"]),
                                                accept_range=int(s5["accept_range"]), forbid_undo=True)
        if plan is None:
            raise _RealSceneGenerationError(f"{self.difficulty}: S5 cannot plan {self.swap_times} swaps")
        pairs = []
        for k, (a, b) in enumerate(plan.pairs):
            chosen = self._spec.value(
                f"actions.swap_pairs.{k}",
                {"initiator": f"bin_{a}", "partner": f"bin_{b}"},
                decision_key=f"{self.difficulty}.swap_plan",
            )
            try:
                pairs.append((_cube_index_of(chosen["initiator"]), _cube_index_of(chosen["partner"])))
            except (TypeError, KeyError, ValueError, IndexError) as exc:
                raise _EpisodeSpecError(f"{self.difficulty}: actions.swap_pairs.{k} has an invalid shape: {chosen!r}") from exc
        problems, stats = swap_uniform.verify_swap_sequence(graph, pairs, forbid_undo=True)
        if problems:
            raise _EpisodeSpecError(f"{self.difficulty}: swap sequence violates V6 S5 rules: " + "; ".join(problems))
        summary = stats.summary()
        summary["tries"] = int(plan.tries)
        self._spec.record("objects.swap_plan", summary)
        self._newvalue_swap_partners = [int(b) for _a, b in pairs]
        # Read-only diagnostics (not in the spec)
        self._newvalue_swap_plan_info = {
            "slots": slots,
            "plan": [{"initiator": a, "partner": b, "slot_a": sa, "slot_b": sb}
                     for (a, b), (sa, sb) in zip(pairs, stats.slot_pairs)],
            "graph_edges": swap_uniform.graph_edges(graph),
            "summary": summary,
            "checked_slot_pairs": len(feasibility.cache),
            "infeasible_slot_pairs": sorted(key for key, ok in feasibility.cache.items() if not ok),
        }
        return pairs

    def _newvalue_planned_partner(self, sweep_index, initiator):
        """Partner actor planned at reset for swap ``sweep_index``."""
        partners = getattr(self, "_newvalue_swap_partners", None)
        if partners is None or sweep_index >= len(partners):
            raise SpecBindingError(f"{self.difficulty}: swap {sweep_index} has no reset-planned partner")
        partner = self.spawned_cubes[partners[sweep_index]]
        if partner is initiator:
            raise SpecBindingError(f"{self.difficulty}: swap {sweep_index} planned partner is the same cube as the initiator")
        return partner

    def _sweep_checks_enabled(self):
        """D5 (H2): geometry checks are enabled only on "channel A" or the new-value tiers' channel B; original three tiers' channel B still does not check."""
        return self._episode_spec is not None or is_newvalue_difficulty(self.difficulty)



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            if self._sweep_checks_enabled():
                # Initialization re-check: read actual collision boxes and check cubes pairwise; runs once at construction and once at the formal reset
                # (V4 D5: enabled on channel A or xhard channel B; original three tiers' channel B still does not run it)
                gap, rejection = self._check_state_readonly("initial")
                self._runtime_checks.append(
                    {
                        "kind": "initial",
                        "min_g_m": None if rejection is not None else gap,
                        "rejection": None if rejection is None else rejection.as_dict(),
                    }
                )
                if rejection is not None:
                    raise BinCollisionError(rejection)
            # V4 xhard: static/swap segments use a wait function that only swallows AttributeError (see _solve_hold_obj_xhard),
            # otherwise the BinCollisionError raised by D5 would be swallowed by solve_hold_obj's bare except and loop forever; original three tiers still use the original function
            hold_fn = _solve_hold_obj_xhard if is_newvalue_difficulty(self.difficulty) else solve_hold_obj
            tasks = [
            {
                "func": (lambda: is_obj_pickup(self, obj=self.target_cube_1)),
                "name": f"pick up the cube",
                "subgoal_segment":f"pick up the cube at <>",
                "choice_label": "pick up the cube",
                "demonstration": True,
                "failure_func": lambda:None,
                "solve": lambda env, planner: [solve_pickup(env, planner, obj=self.target_cube_1)],
                'segment':self.target_cube_1,
            },{
                "func": (lambda: is_obj_dropped(self, obj=self.target_cube_1)),
                "name": "drop the cube on the table",
                "subgoal_segment":f"drop the cube on the table",
                "choice_label": "put it down",
                "demonstration": True,
                "failure_func": lambda: None,
                "solve": lambda env, planner: [solve_putdown_whenhold(env, planner,release_z=0.03)]
                }, 
            ]
            if self.swap_times>=1:
                tasks.append(   {
                                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=20),
                                "name": "static",
                                "subgoal_segment":"static",
                                "demonstration": True,
                                "failure_func": None,
                                "solve": lambda env, planner: [solve_reset(env,planner),hold_fn(env, planner, static_steps=20)],
                                },)
            if self.swap_times>=1:
                for count in range(self.swap_times):
                    if is_newvalue_difficulty(self.difficulty):
                        # V6 review fix N11 (user "n11 a"): completion check and wait solver of segment k are both pinned to that segment's absolute end step
                        # swap_schedule[k][3] (step() refreshes the schedule from start_step when the first segment task becomes current), see _solve_hold_until_step_xhard
                        tasks.append(   {
                                "func": lambda k=count: bool(is_static(self)) and int(self.elapsed_steps) >= int(self.swap_schedule[k][3]),
                                "name": "static",
                                "subgoal_segment":"static",
                                "demonstration": True,
                                "failure_func": None,
                                "specialflag":"swap",
                                "solve": lambda env, planner, k=count: [_solve_hold_until_step_xhard(env, planner, self.swap_schedule[k][3])],
                                },)
                        continue
                    tasks.append(   {
                                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=self.swap_schedule[-1][3]-self.swap_schedule[-1][2]),
                                "name": "static",
                                "subgoal_segment":"static",
                                "demonstration": True,
                                "failure_func": None,
                                "specialflag":"swap",
                                "solve": lambda env, planner: [hold_fn(env, planner, static_steps=self.swap_schedule[-1][3]-self.swap_schedule[-1][2])],
                                },)
                
            tasks.append(             {
                                "func": lambda:reset_check(self),
                                "name": "NO RECORD",
                                "subgoal_segment":"NO RECORD",
                                "demonstration": True,
                                "failure_func": None,
                                "solve": lambda env, planner: [ solve_strong_reset(env,planner)],
                                },)
            ordinal_words = [
                "first",
                "second",
                "third",
                "fourth",
                "fifth",
                "sixth",
                "seventh",
                "eighth",
                "ninth",
                "tenth",
            ]
            # V6 review fix M1 (user "m1 a"): the four new tiers' "pressing the button early = failure" window rolls with rounds -- each round's pick/place uses its own timer
            # (counted from when that task becomes current); the first round keeps the original [50,500] steps, later rounds start from step 0, the window covering up to the last place;
            # original three tiers still use the single timer 2/3 ([50,500] steps from the first pick, never reset).
            rolling_window = is_newvalue_difficulty(self.difficulty)
            for i in range(self.num_repeats):
                ordinal = ordinal_words[i] if i < len(ordinal_words) else f"{i+1}th"
                pick_timer = f"xhard_pick_{i}" if rolling_window else 2
                put_timer = f"xhard_put_{i}" if rolling_window else 3
                window_min = 50 if (not rolling_window or i == 0) else 0
                tasks.append(  {
                        "func": (lambda: is_obj_pickup(self, obj=self.target_cube_1)),
                        "name": f"pick up the correct cube for the {ordinal} time" ,
                        "subgoal_segment":f"pick up the correct cube at <> for the {ordinal} time" ,
                        "choice_label": "pick up the cube",
                        "demonstration": False,
                        "failure_func": lambda pick_timer=pick_timer, window_min=window_min: [
                            is_any_obj_pickup(self,[cube for cube in self.spawned_cubes if cube != self.target_cube_1]),
                            timewindow(self, lambda: is_button_pressed(self, obj=self.button_left),min_steps=window_min,max_steps=500,timewindow_timer=pick_timer,),],
                        "solve": lambda env, planner: [solve_pickup(env, planner, obj=self.target_cube_1)],
                        'segment':self.target_cube_1,
                    },)
                
                tasks.append({
                        "func": lambda: is_obj_dropped(self,obj=self.target_cube_1),
                    "name": "put it down",
                    "subgoal_segment":f"put it down",
                    "choice_label": "put it down",
                        "demonstration": False,
                        "failure_func": lambda put_timer=put_timer, window_min=window_min:[
                            is_any_obj_pickup(self,[cube for cube in self.spawned_cubes if cube != self.target_cube_1]),
                            timewindow(self, lambda: is_button_pressed(self, obj=self.button_left),min_steps=window_min,max_steps=500,timewindow_timer=put_timer,),],
                        "solve": lambda env, planner: solve_putdown_whenhold(env, planner,release_z=0.01)
                    })

            tasks.append({
                    "func": lambda: is_button_pressed(self, obj=self.button_left),
                    "name": "press the button to finish",
                    "subgoal_segment":f"press the button at <> to finish",
                    "choice_label": "press the button to finish",
                    "demonstration": False,
                    "failure_func":lambda: is_any_obj_pickup(self,[cube for cube in self.spawned_cubes]),
                    "solve": lambda env, planner: solve_button(env, planner, obj=self.button_left),
                    "segment":self.cap_link 
                })



        # Store task list for RecordWrapper use
        self.task_list = tasks

        # Record pickup related task indices and items for recovery
        self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
        if self.robomme_failure_recovery:
            # Only inject an intentional failed grasp when recovery mode is enabled
            self.fail_grasp_task_index = inject_fail_grasp(
                self.task_list,
                generator=self.generator,
                mode=self.robomme_failure_recovery_mode,
            )
        else:
            self.fail_grasp_task_index = None

    def _get_obs_extra(self, info: Dict):
        return dict()



    def evaluate(self,solve_complete_eval=False):
        previous_failure = getattr(self, "failureflag", None)
        self.successflag = torch.tensor([False])
        if previous_failure is not None and bool(previous_failure.item()):
            self.failureflag = previous_failure
        else:
            self.failureflag = torch.tensor([False])


        # Use encapsulated sequence task check function
        if(self.use_demonstrationwrapper==False):# change subgoal after planner ends during recording
            if solve_complete_eval==True:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False
        else:# during demonstration, video needs to call evaluate(solve_complete_eval), video ends and flag changes in demonstrationwrapper
            if solve_complete_eval==True or self.demonstration_record_traj==False:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False
        all_tasks_completed, current_task_name, task_failed,self.current_task_specialflag = sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

        # If task failed, mark as failed immediately
        if task_failed:
            self.failureflag = torch.tensor([True])
            logger.debug(f"Task failed: {current_task_name}")

        # If static_check succeeds or all tasks completed, set success flag
        if all_tasks_completed and not task_failed:
            self.successflag = torch.tensor([True])

        return {
            "success": self.successflag,
            "fail": self.failureflag,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.agent.tcp_pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward*0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    def _get_other_bins_for_pair(self, idx_a: int, idx_b: int):
        """Return bins that are not part of the provided pair indices."""
        if not hasattr(self, "spawned_bins"):
            return []

        total_bins = len(self.spawned_cubes)
        if idx_a >= total_bins or idx_b >= total_bins:
            return []

        # Prefer precomputed lists when available
        if hasattr(self, "otherbins") and idx_a < len(self.otherbins):
            other_candidates = [
                bin_actor
                for bin_actor in self.otherbins[idx_a]
                if bin_actor is not self.spawned_cubes[idx_b]
            ]
            return other_candidates

        return [
            bin_actor
            for i, bin_actor in enumerate(self.spawned_cubes)
            if i not in (idx_a, idx_b)
        ]

    def _get_actor_position(self, actor):
        """Return actor position as a numpy array."""
        if actor is None:
            return np.zeros(3, dtype=np.float32)

        pos = actor.pose.p if hasattr(actor, "pose") else actor.get_pose().p
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()

        pos = np.asarray(pos, dtype=np.float32).reshape(-1)
        if pos.size < 3:
            padded = np.zeros(3, dtype=np.float32)
            padded[: pos.size] = pos
            return padded
        return pos

    def _compute_dynamic_swap_candidates(self, positions):
        """Compute nearest-neighbour swap candidates using provided positions."""
        candidate_map = {}
        num_positions = len(positions)
        if num_positions <= 1:
            return candidate_map

        for idx, pos in enumerate(positions):
            distances = []
            for other_idx, other_pos in enumerate(positions):
                if other_idx == idx:
                    continue
                dist = np.linalg.norm(pos[:2] - other_pos[:2])
                distances.append((other_idx, dist))

            distances.sort(key=lambda item: item[1])
            candidate_map[idx] = [j for j, _ in distances[:2]]

        return candidate_map

    def _select_swap_pair_from_positions(self, positions, generator=None):
        """Select one swap pair given current planned positions."""
        num_bins = len(positions)
        if num_bins < 2:
            return None

        candidate_map = self._compute_dynamic_swap_candidates(positions)
        valid_indices = [idx for idx, cands in candidate_map.items() if cands]
        if not valid_indices:
            return None

        if generator is None:
            generator = self.generator

        # Swap partners are only "known when the event happens": frozen separately per event index (plan 8.2)
        event_index = getattr(self, "_native_swap_event_index", -1) + 1
        self._native_swap_event_index = event_index
        first_idx = valid_indices[
            self._spec.value(
                f"actions.swap_pairs.{event_index}.first_choice",
                int(torch.randint(0, len(valid_indices), (1,), generator=generator).item()),
            )
        ]
        candidates = candidate_map[first_idx]
        second_idx = candidates[
            self._spec.value(
                f"actions.swap_pairs.{event_index}.second_choice",
                int(torch.randint(0, len(candidates), (1,), generator=generator).item()),
            )
        ]

        distance = float(
            np.linalg.norm(positions[first_idx][:2] - positions[second_idx][:2])
        )

        return {"idx1": first_idx, "idx2": second_idx, "distance": distance}

    def _verify_swap_binding(self, sweep_index, initiator, resolved_partner):
        """Verify that the partner pre-written in the spec matches the actual nearest neighbor computed at execution; fail on mismatch, changing partners is forbidden."""
        pairs = self._episode_spec["actions"]["swap_pairs"]
        if sweep_index >= len(pairs):
            raise SpecBindingError(
                f"swap segment {sweep_index} has no entry in the spec (spec only has {len(pairs)} segments)"
            )
        expected = pairs[sweep_index]
        actual_initiator = self.spawned_cubes.index(initiator)
        actual_partner = self.spawned_cubes.index(resolved_partner)
        reference = self._get_actor_position(initiator)
        candidates = [
            (index, self._get_actor_position(actor))
            for index, actor in enumerate(self.spawned_cubes)
            if actor is not None and actor is not initiator
        ]
        recomputed, table = nearest_partner_index(reference, candidates)
        detail = {
            "sweep_index": sweep_index,
            "control_step": int(self.elapsed_steps),
            "expected_initiator": expected["initiator"],
            "expected_partner": expected["partner"],
            "actual_initiator": f"bin_{actual_initiator}",
            "actual_partner": f"bin_{actual_partner}",
            "recomputed_partner": f"bin_{recomputed}",
            "distances": [[f"bin_{index}", dist] for index, dist in table],
        }
        self._runtime_checks.append({"kind": "swap_binding", **detail})
        if expected["initiator"] != f"bin_{actual_initiator}":
            raise SpecBindingError(
                f"segment {sweep_index} initiator is bin_{actual_initiator}, spec pre-wrote {expected['initiator']}", detail
            )
        if expected["partner"] != f"bin_{actual_partner}":
            raise SpecBindingError(
                f"segment {sweep_index} actual nearest neighbor is bin_{actual_partner}, spec pre-wrote {expected['partner']}", detail
            )

    def _object_states_for_collision(self):
        """Read the three cubes on the table into the state used by the collision criterion; reads real collision boxes only."""
        return [
            object_state_from_actor(actor, f"bin_{index}")
            for index, actor in enumerate(self.spawned_cubes)
            if actor is not None
        ]

    def _check_swap_sweep_from_actual(self, sweep_index, initiator, partner):
        """Continuously check the whole swap path from actual poses; on a hit raise to abort the sample."""
        states = {}
        for index, actor in enumerate(self.spawned_cubes):
            if actor is None:
                continue
            states[index] = object_state_from_actor(actor, f"bin_{index}")
        a = self.spawned_cubes.index(initiator)
        b = self.spawned_cubes.index(partner)
        bystanders = [state for index, state in sorted(states.items()) if index not in (a, b)]
        gap, rejection = check_swap_sweep(states[a], states[b], bystanders, sweep_index=sweep_index, stage="sweep")
        self._runtime_checks.append(
            {
                "kind": "swap_sweep",
                "sweep_index": sweep_index,
                "control_step": int(self.elapsed_steps),
                "min_g_m": None if rejection is not None else gap,
                "rejection": None if rejection is None else rejection.as_dict(),
            }
        )
        if rejection is not None:
            raise BinCollisionError(rejection)

    def _check_state_readonly(self, stage):
        """Read-only re-check at one instant; returns rejection evidence without raising."""
        return check_bin_state(self._object_states_for_collision(), stage=stage)

    def _in_swap_window(self):
        """Whether the current control step falls within the time window of some swap segment.

        WARNING: substep checks run only inside the window. Outside the window bins/cubes are either static or pinned in place every frame by
        ``lift_and_drop_objects_back_to_original``, which the initial-state check already covers;
        substeps far outnumber control steps, so running a full SAT on every substep throughout would slow a single sample down more than tenfold
        (measured in smoke tests: VideoUnmaskSwap went from ~47 seconds to minutes).
        """
        step = int(self.elapsed_steps)
        return any(start <= step <= end for _a, _b, start, end in getattr(self, "swap_schedule", []))

    def _before_simulation_step(self):
        super()._before_simulation_step()
        if not self._sweep_checks_enabled() or not self._in_swap_window():
            return
        gap, rejection = self._check_state_readonly("substep_before")
        if rejection is not None:
            self._runtime_checks.append(
                {"kind": "substep_before", "control_step": int(self.elapsed_steps), "rejection": rejection.as_dict()}
            )
            raise BinCollisionError(rejection)

    def _after_simulation_step(self):
        super()._after_simulation_step()
        if not self._sweep_checks_enabled() or not self._in_swap_window():
            return
        gap, rejection = self._check_state_readonly("substep_after")
        if rejection is not None:
            self._runtime_checks.append(
                {"kind": "substep_after", "control_step": int(self.elapsed_steps), "rejection": rejection.as_dict()}
            )
            raise BinCollisionError(rejection)

    def _refresh_swap_schedule(self,start_step=400):
        # General formula: the k-th swap occupies [start_step+50k, start_step+50(k+1)]; identical to the original three branches for 1/2/3 swaps, nothing assigned for 0
        if self.swap_times < 1:
            return
        self.swap_schedule = [
            (getattr(self, f"swap_pair{k+1}_idx1"), getattr(self, f"swap_pair{k+1}_idx2"), start_step + 50 * k, start_step + 50 * (k + 1))
            for k in range(self.swap_times)
        ]

#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


       
        if self.current_task_specialflag=="swap":
            if self.static_flag==False:
                self.static_flag=True
                self.start_step=int(self.elapsed_steps.item())
                self._refresh_swap_schedule(self.start_step)
                logger.debug("tag!")
             
        if self.static_flag==True:
            for i in range(len(self.swap_schedule)):
                start = self.swap_schedule[i][2]
                end = self.swap_schedule[i][3]
                if self.elapsed_steps in range (start,end):
                    # Select corresponding swap pair based on index
                    pair_idx1 = getattr(self, f'swap_pair{i+1}_idx1')
                    pair_idx2 = getattr(self, f'swap_pair{i+1}_idx2')

                    if pair_idx2 is None and pair_idx1 is not None:
                        if is_newvalue_difficulty(getattr(self, "difficulty", None)):
                            # New-value tier partners use those planned at reset, no longer the nearest neighbor by actual XY;
                            # the D5 actual-pose sweep check below still runs as a runtime guard
                            closest_actor = self._newvalue_planned_partner(i, pair_idx1)
                        else:
                            reference_pos = self._get_actor_position(pair_idx1)
                            closest_actor = None
                            closest_dist = float("inf")
                            for candidate in self.spawned_cubes:
                                if candidate is None or candidate is pair_idx1:
                                    continue
                                candidate_pos = self._get_actor_position(candidate)
                                axes = self._sampling["parameters"]["swap_selection"]["partner"]["position_axes"]
                                dist = np.linalg.norm(reference_pos[axes] - candidate_pos[axes])
                                if dist < closest_dist:
                                    closest_dist = dist
                                    closest_actor = candidate
                        if closest_actor is not None:
                            # Two runtime checkpoints of new-value injection (plan section 5 step 0d): first verify partner identity,
                            # then run a continuous geometry check of the whole segment from the **actual** start state. Neither runs when disabled.
                            if self._episode_spec is not None:
                                self._verify_swap_binding(i, pair_idx1, closest_actor)
                            # V4 D5 (H2): the sweep check now runs on "channel A, or the new-value tiers' channel B"; original three tiers' channel B still does not run it.
                            # Only channel A's spec pre-writes partners for identity verification; channel B only records read-only (see below).
                            if self._sweep_checks_enabled():
                                self._check_swap_sweep_from_actual(i, pair_idx1, closest_actor)
                            if is_newvalue_difficulty(self.difficulty):
                                self._spec.record(
                                    f"actions.swap_pairs.{i}",
                                    {"initiator": f"bin_{self.spawned_cubes.index(pair_idx1)}",
                                     "partner": f"bin_{self.spawned_cubes.index(closest_actor)}"},
                                )
                                self._spec.record(
                                    f"actions.swap_windows.{i}", [int(start), int(end)]
                                )
                            setattr(self, f'swap_pair{i+1}_idx2', closest_actor)
                            self._refresh_swap_schedule(self.start_step)


            for idx_a, idx_b, start_step, end_step in self.swap_schedule:
                if idx_a is None or idx_b is None:
                    continue
                if self.elapsed_steps >= int(start_step) and self.elapsed_steps <= int(end_step):
                        
                    swap_flat_two_lane(
                                    self,
                                    cube_a=idx_a,
                                    cube_b=idx_b,
                                    start_step=start_step,
                                    end_step=end_step,
                                    cur_step=self.elapsed_steps,
                                    lane_offset=0.07,
                                    smooth=True,
                                    keep_upright=True,
                                    other_cube=[b for b in self.spawned_cubes if b not in (idx_a, idx_b)],  # Keep all other bins in place to prevent collision during swap
                                )



        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
