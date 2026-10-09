import copy
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

from .utils import *
from .utils.subgoal_evaluate_func import static_check
from .utils import subgoal_language
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils.episode_spec import SpecRecorder
from .utils.sampling_config import (
    SamplingConfigError, assert_native_decision, fill_missing_newvalue, split_sampling_config,
)
from .utils.SceneGenerationError import SceneGenerationError
from .utils import reset_panda
from .utils.difficulty import normalize_robomme_difficulty, is_newvalue_difficulty
from ..logging_utils import logger


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


# ── Original values of the decision/native blocks (newtaskRelease-v3 step 3, mapping in plan section 2.9) ────────
# decision: cube layout mode and sampling region, number of targets highlighted simultaneously and to be picked, total cubes on the table, per-cube color policy.
# native: per-cube color and pose drawing, highlight target selection (randperm), button position, recovery,
#        highlight window and pick order (all targets share one window, highlighted simultaneously).
NATIVE_SAMPLING = {
    "parameters": {
        "color_pool": [
            {"rgba": [1, 0, 0, 1], "name": "red"},
            {"rgba": [0, 0, 1, 1], "name": "blue"},
            {"rgba": [0, 1, 0, 1], "name": "green"},
        ],
        "color_draw": {"sampler": "torch.randint", "low": 0, "high_exclusive": "len(color_pool)"},
        "target_selection": {"sampler": "torch.randperm(len(all_cubes))[:pickup]"},
        "spawn_failure": "break on spawn failure, keep the actual count, no extra draws",
        "recovery": "keep the entry-provided fail recover mode and the original generator",
    },
    "positions": {
        "button": {"center_xy": [-0.2, 0], "scale": 1.5},
        "cubes": {
            "region_center": [-0.1, 0],
            "region_half_size": 0.2,
            "min_gap_factor": 2,
            "random_yaw": True,
            "include_existing": False,
            "include_goal": False,
        },
        "highlight_window": {"start_step": 10, "end_step": 100, "simultaneous": True},
    },
}


def native_blocks(cls):
    """Original ``(decision, native)`` blocks of this env; shared by external export and internal parsing."""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


# V4 xhard-specific decision entries (NEWTASK_RELEASE_V4_PLAN 2.12); placed under the subkey named ``xhard``,
# after the guard (assert_native_decision) strips xhard, the part visible to the original three tiers is verbatim identical to the original.
#   block_color_policy: "block color arbitrary" = each cube draws its color independently; per user decision 2026-09-22 the gamut has saturation/value floors
#     ("hsv_floor": any hue, S>=0.5, V>=0.4, see utils/xhard.py::HSV_FLOOR_COLOR; alpha fixed 1).
#   subgoal_color_suffix: arbitrary RGB has no color name; how to write the subgoal's ``, which is {color}`` suffix.
#     User 2026-09-22 decided "drop it entirely" ("omit"); only this one is implemented, other values are rejected.
from .utils.xhard import HSV_FLOOR_COLOR, cube_obb2d_exact, hsv_floor_rgb

XHARD_DECISION = {
    "block_color_policy": "hsv_floor",
    "block_color_hsv": copy.deepcopy(HSV_FLOOR_COLOR),
    "subgoal_color_suffix": "omit",
}

# V6: all four tiers keep arbitrary HSV colors, exact OBB and the color-free subgoal suffix; tier values only change pick count and total cubes.
NEWVALUE_DECISION = {
    "xhard4": XHARD_DECISION,
    "xhard1": copy.deepcopy(XHARD_DECISION),
    "xhard2": copy.deepcopy(XHARD_DECISION),
    "xhard3": copy.deepcopy(XHARD_DECISION),
}


def _native_decision(cls):
    """Slice the decision block per plan section 2.9 (equals the original in the original-value stage).

    ``highlight_count`` / ``spawn_count`` are expanded per tier from ``cls.configs``: integers for the original three tiers,
    closed interval ``[lo, hi]`` for xhard (V4 new value, admitted by the guard's xhard pass-through).
    """
    return {
        "layout_mode": "native_region",
        "cube_region": {"region_center": [-0.1, 0], "region_half_size": 0.2},
        "highlight_count": {difficulty: copy.deepcopy(cfg["pickup"]) for difficulty, cfg in cls.configs.items()},
        "spawn_count": {difficulty: copy.deepcopy(cfg["spawn"]) for difficulty, cfg in cls.configs.items()},
        "block_color_policy": "native_per_cube_uniform",
        # V6: xhard subtree original values unchanged, then append three same-structure subtrees xhard1/2/3 per tier
        **{tier: copy.deepcopy(entry) for tier, entry in NEWVALUE_DECISION.items()},
    }


def _closed_range(value, key):
    """Validate xhard's closed interval ``[lo, hi]`` into two integers; reject directly if the external config is malformed."""
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
            or value[0] < 1 or value[0] > value[1]):
        raise SamplingConfigError(
            f"PickHighlight: decision.{key} must be a closed interval [lo, hi] (integers with 1<=lo<=hi), got {value!r}"
        )
    return int(value[0]), int(value[1])


def _resolve_sampling_config(cls, override):
    """Split out this instance's private decision/native copies; draws no random numbers, must be called before the Generator."""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    # V6: old snapshots (V5 has no xhard1/2/3 subtrees) get missing new-value tiers filled from source declarations; existing ones untouched
    fill_missing_newvalue(decision, decision_default)
    native["decision"] = decision
    return native


@register_env("PickHighlight", override=True)
class PickHighlight(BaseEnv):

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

    config_hard = {
        'spawn': 6,
        "pickup": 3
    }

    config_easy = {
        'spawn': 3,
        "pickup": 1
    }

    config_medium = {
        'spawn': 4,
        "pickup": 2
    }

    # xhard4: finalized pick 7, total cubes 10; intervals degenerate to single values.
    config_xhard4 = {
        'spawn': [10, 10],
        "pickup": [7, 7]
    }

    # V6 new tiers take finalized values, keeping the spawn lower bound >= pick upper bound.
    config_xhard1 = {
        'spawn': [7, 7],
        "pickup": [4, 4]
    }

    config_xhard2 = {
        'spawn': [8, 8],
        "pickup": [5, 5]
    }

    config_xhard3 = {
        'spawn': [9, 9],
        "pickup": [6, 6]
    }

    # Combine into a dictionary
    configs = {
        'hard': config_hard,
        'easy': config_easy,
        'medium': config_medium,
        'xhard4': config_xhard4,
        'xhard1': config_xhard1,
        'xhard2': config_xhard2,
        'xhard3': config_xhard3,
    }


    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     sampling_config=None,
                     native_episode_spec=None,
                     **kwargs):
        # Must happen before any RNG call and before super().__init__()
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "PickHighlight", {"seed": seed},
                                  difficulty=kwargs.get("difficulty"))
        # Initialization index starts at -1; _initialize_episode increments it on each entry;
        # value points in _load_scene use index-free paths, so this is only a fallback.
        self._native_init_index = -1
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
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
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

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
        normalized_robomme_difficulty = normalize_robomme_difficulty(
            kwargs.pop("difficulty", None)
        )
        if normalized_robomme_difficulty is not None:
            self.difficulty = normalized_robomme_difficulty
        else:
            # Determine difficulty based on seed % 3
            seed_mod = seed % 3
            if seed_mod == 0:
                self.difficulty = "easy"
            elif seed_mod == 1:
                self.difficulty = "medium"
            else:  # seed_mod == 2
                self.difficulty = "hard"

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
        self.generator.manual_seed(self.seed)
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()


        button_cfg = self._sampling["positions"]["button"]
        cubes_cfg = self._sampling["positions"]["cubes"]
        decision_cfg = self._sampling["decision"]
        cube_region = decision_cfg["cube_region"]
        self._spec.identity.setdefault("difficulty", getattr(self, "difficulty", None))
        button_obb = build_button(
            self,
            center_xy=tuple(button_cfg["center_xy"]),
            scale=button_cfg["scale"],
            generator=self.generator,
            recorder=self._spec,
            spec_path="layout.button_xy",
        )
        avoid = [button_obb]

        self.all_cubes = []  # Save all cube objects
        self.all_cube_names = []
        self.all_cube_colors = []

        # List of available colors
        # Color pool from the snapshot (same order as the original literal: red, blue, green)
        available_colors = [
            {"color": tuple(entry["rgba"]), "name": entry["name"]}
            for entry in self._sampling["parameters"]["color_pool"]
        ]

        # V4 xhard branch (H2/N12: the original three tiers never go through here).
        # V6: the four new-value tiers share this branch, fetching subtree and intervals by this episode's tier
        xhard = is_newvalue_difficulty(self.difficulty)
        tier = self.difficulty
        if xhard:
            xhard_cfg = decision_cfg[tier]
            if xhard_cfg["block_color_policy"] != "hsv_floor":
                raise SamplingConfigError(
                    f"PickHighlight: decision.{tier}.block_color_policy only supports 'hsv_floor', "
                    f"got {xhard_cfg['block_color_policy']!r}"
                )
            if xhard_cfg["subgoal_color_suffix"] != "omit":
                raise SamplingConfigError(
                    f"PickHighlight: decision.{tier}.subgoal_color_suffix currently only implements 'omit', "
                    f"got {xhard_cfg['subgoal_color_suffix']!r}"
                )
            spawn_lo, spawn_hi = _closed_range(decision_cfg["spawn_count"][tier], f"spawn_count.{tier}")
            highlight_lo, highlight_hi = _closed_range(
                decision_cfg["highlight_count"][tier], f"highlight_count.{tier}"
            )
            # Hard assertion spawn>=highlight (plan 2.12): checked against the worst case of the effective intervals; malformed external narrowing is rejected on the spot
            if spawn_lo < highlight_hi:
                raise SamplingConfigError(
                    f"PickHighlight: {tier} requires spawn lower bound >= highlight upper bound, got spawn=[{spawn_lo},{spawn_hi}] "
                    f"highlight=[{highlight_lo},{highlight_hi}]"
                )
            # New value point: this episode's cube count (drawn only in xhard; it decides the length of the loop below, so it must precede the cube loop)
            num_cubes_to_spawn = int(self._spec.value(
                "objects.n_cubes",
                int(torch.randint(spawn_lo, spawn_hi + 1, (1,), generator=self.generator).item()),
                decision_key=f"spawn_count.{tier}",
            ))
        else:
            # Get number of cubes to spawn based on difficulty
            num_cubes_to_spawn = decision_cfg["spawn_count"][self.difficulty]

        # Spawn specified number of cubes, each with random color
        for cube_idx in range(num_cubes_to_spawn):
            if xhard:
                # Arbitrary colors: each cube draws its color independently (restricted HSV gamut; replaces the original three-color randint, only in xhard).
                # No color name => label set to None, actor name uses "rgb"; the subgoal suffix follows subgoal_color_suffix.
                rgba = self._spec.value(
                    f"objects.color_rgba.{cube_idx}",
                    hsv_floor_rgb(torch.rand(3, generator=self.generator).tolist(),
                                  xhard_cfg["block_color_hsv"]) + [1.0],
                    decision_key=f"{tier}.block_color_policy",
                )
                chosen_color = {"color": tuple(float(c) for c in rgba), "name": "rgb", "label": None}
            else:
                # Randomly select a color
                color_choice_idx = self._spec.value(
                    f"objects.color_choice.{cube_idx}",
                    torch.randint(
                        self._sampling["parameters"]["color_draw"]["low"],
                        len(available_colors), (1,), generator=self.generator,
                    ).item(),
                )
                chosen_color = available_colors[color_choice_idx]

            try:
                cube = spawn_random_cube(
                    self,
                    color=chosen_color["color"],
                    avoid=avoid,
                    include_existing=False,
                    include_goal=False,
                    region_center=list(cube_region["region_center"]),
                    region_half_size=cube_region["region_half_size"],
                    half_size=self.cube_half_size,
                    min_gap=self.cube_half_size*cubes_cfg["min_gap_factor"],
                    random_yaw=cubes_cfg["random_yaw"],
                    name_prefix=f"cube_{chosen_color['name']}_{cube_idx}",
                    generator=self.generator,
                    recorder=self._spec,
                    spec_path=f"layout.cubes.{cube_idx}",
                )

                cube_name = f"cube_{chosen_color['name']}_{cube_idx}"

                # Add cube immediately after successful creation
                self.all_cubes.append(cube)
                self.all_cube_names.append(cube_name)
                self.all_cube_colors.append(chosen_color.get("label", chosen_color["name"]))
                setattr(self, cube_name, cube)
                if xhard:
                    # V5 (L2 b, plan 2.16): placed cubes enter avoid as exact 2D obstacles. The actor path via
                    # _trimesh_box_to_obb2d degenerates into a segment ~2/3 of the time, voiding min_gap along its normal; take the
                    # initial_pose used to build the cube (does not depend on simulation being initialized), no random draws. Original three tiers still pass the actor, verbatim unchanged.
                    avoid.append(cube_obb2d_exact(cube.initial_pose, self.cube_half_size))
                else:
                    avoid.append(cube)

            except RuntimeError as e:
                if xhard:
                    # 2.2-4: xhard must not truncate silently; failing to place all cubes fails this episode's generation
                    raise SceneGenerationError(
                        f"PickHighlight xhard: cubes do not all fit, requested {num_cubes_to_spawn} actual {len(self.all_cubes)}"
                        f" (cube {cube_idx} failed: {e})"
                    ) from e
                logger.debug(f"Failed to spawn cube {cube_idx} ({chosen_color['name']}): {e}")
                break

        logger.debug(f"Generated {len(self.all_cubes)} cubes total")

        if xhard:
            # Requested vs actual (2.2-4); reaching here they must be equal
            self._spec.record("objects.n_cubes_spawned", len(self.all_cubes))
            # The original draw randperm(len(all_cubes)) stays in place; the highlight count is a new value point appended after randperm
            permutation = torch.randperm(len(self.all_cubes), generator=self.generator).tolist()
            highlight_count = int(self._spec.value(
                "objects.highlight_count",
                int(torch.randint(highlight_lo, highlight_hi + 1, (1,), generator=self.generator).item()),
                decision_key=f"highlight_count.{tier}",
            ))
            # randperm(len)[:k] silently truncates when k>len; explicitly blocked here
            if highlight_count > len(permutation):
                raise SceneGenerationError(
                    f"PickHighlight xhard: highlight count {highlight_count} exceeds actual cube count {len(permutation)}"
                )
            target_cube_indices = self._spec.value(
                "objects.highlight_ids", permutation[:highlight_count],
                decision_key=f"highlight_count.{tier}",
            )
        else:
            # Randomly select one cube from all available cubes as the target
            target_cube_indices = self._spec.value(
                "objects.highlight_ids",
                torch.randperm(len(self.all_cubes), generator=self.generator)[:decision_cfg["highlight_count"][self.difficulty]].tolist(),
            )

        self.target_cubes = [self.all_cubes[idx] for idx in target_cube_indices]
        self.target_cube_names = [self.all_cube_names[idx] for idx in target_cube_indices]
        self.target_cube_colors = [self.all_cube_colors[idx] for idx in target_cube_indices]
        self.target_labels = [
            (color or name or "target") 
            for color, name in zip(self.target_cube_colors, self.target_cube_names)
        ]
        # Record pick count for each target cube; all must be > 1 for success
        self.target_cube_pickup_counts = {name: 0 for name in self.target_cube_names}

        # Define task list, each task contains a dictionary with function, name, demonstration flag, and optional failure_func
        tasks = []
        target_label = getattr(self, "target_cube_color", None) or getattr(
            self, "target_cube_name", None
        ) or getattr(self, "target_label", None) or "target"
        self.target_label = target_label

        if xhard:
            # D4 fixed only in xhard (H2): the original three tiers pass the value evaluated at construction time here (not a callable),
            # so the criterion never took effect; xhard wraps it in a lambda: picking any cube before pressing the button is a failure.
            button_failure_func = lambda: is_any_obj_pickup(self, [cube for cube in self.all_cubes])
        else:
            button_failure_func = is_any_obj_pickup(self,[cube for cube in self.all_cubes])
        tasks.append({
            "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment":"press the button at <>",
                "choice_label": "press button",
                "demonstration": False,
                "failure_func":button_failure_func,
                "solve": lambda env, planner:solve_button(env, planner, obj=self.button),
                 "segment":self.cap_link,
            })
        # Pick each target cube once, lambda captures current cube explicitly to avoid closure issue
        num_targets = len(self.target_cubes)
        for cube_idx, cube in enumerate(self.target_cubes):
                # If only one target cube, do not show index
                if xhard:
                    # Arbitrary RGB has no color name: subgoal_color_suffix="omit" => drop the whole ", which is {color}" suffix
                    if num_targets == 1:
                        task_name = "pick up the highlighted cube"
                        task_subgoal = "pick up the highlighted cube at <>"
                    else:
                        task_name = subgoal_language.get_subgoal_with_index(cube_idx, "pick up the {idx} highlighted cube")
                        task_subgoal = subgoal_language.get_subgoal_with_index(cube_idx, "pick up the {idx} highlighted cube at <>")
                elif num_targets == 1:
                    task_name = f"pick up the highlighted cube, which is {self.target_labels[cube_idx]}"
                    task_subgoal = f"pick up the highlighted cube at <>, which is {self.target_labels[cube_idx]}"
                else:
                    task_name = subgoal_language.get_subgoal_with_index(cube_idx, "pick up the {idx} highlighted cube, which is {color}", color=self.target_labels[cube_idx])
                    task_subgoal = subgoal_language.get_subgoal_with_index(cube_idx, "pick up the {idx} highlighted cube at <>, which is {color}", color=self.target_labels[cube_idx])

                tasks.append({
                    "func": (lambda c=cube: is_any_obj_pickup_flag_currentpickup(self, objects=[c])),
                    "name": task_name,
                    "subgoal_segment": task_subgoal,
                    "choice_label": "pick up the highlighted cube",
                    "demonstration": False,
                    "failure_func": lambda idx=cube_idx:
                        [is_any_obj_pickup(self,[cube for cube in self.all_cubes if cube not in self.target_cubes] ),
                       ],
                    "solve": lambda env, planner, c=cube: solve_pickup(env, planner, obj=c),
                    "segment":cube,
                })
                if xhard or cube_idx!=num_targets-1:
                    # V6 review fix F1 (user K2 "agree to fix f1"): in the four new tiers the last cube is also put down after being picked, then finish with the final button; original three tiers stop once the last cube is picked
                    tasks.append({
                        "func": (lambda :is_obj_dropped_currentpickup(self,self.target_cubes)),
                        "name": f"place the cube onto the table",
                        "subgoal_segment":"place the cube onto the table",
                        "choice_label": "place the cube onto the table",
                        "demonstration": False,
                        "failure_func": lambda idx=cube_idx:
                        [ is_any_obj_pickup(self,[cube for cube in self.all_cubes if cube not in self.target_cubes] ),
                           ],
                        "solve": lambda env, planner, c=cube: [solve_putdown_whenhold(env, planner, release_z=0.01),
                                                        # solve_pickup(env, planner, obj=c),
                                                        # solve_putdown_whenhold(env, planner, obj=c,release_z=0.01)# For testing
                                                        ],
                        "segment":None,
                    })
        if xhard:
            # V6 review fix F1 (K2): the four new tiers append "press the button" at the end of the chain, consistent with the prompt "finally press the button to stop";
            # the success moment moves to the final button accordingly (see the xhard branch of evaluate). Original three tiers' task chain unchanged.
            tasks.append({
                "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment": "press the button at <>",
                "choice_label": "press button",
                "demonstration": False,
                "failure_func": None,
                "solve": lambda env, planner: solve_button(env, planner, obj=self.button),
                "segment": self.cap_link,
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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Each initialization records its own spec, never reusing the previous result
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)            


    def _get_obs_extra(self, info: Dict):
        return dict()


    def evaluate(self,solve_complete_eval=False):
        self.successflag=torch.tensor([False])
        # Keep previous failure state (once failed, always failed)
        if not hasattr(self, 'failureflag') or self.failureflag is None:
            self.failureflag = torch.tensor([False])
        previous_failure = bool(self.failureflag.detach().cpu().item()) if isinstance(self.failureflag, torch.Tensor) else False
        # If previously failed, do not reset, keep failed state; otherwise reset
        if previous_failure:
            # Keep failed state, do not reset
            pass
        else:
            self.failureflag = torch.tensor([False])



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
            
        # Use encapsulated sequence task check function
        all_tasks_completed, current_task_name, task_failed ,self.current_task_specialflag= sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

        if task_failed:
            self.failureflag = torch.tensor([True])
            logger.debug(f"Task failed: {current_task_name}")
        
        # If previously failed, keep failed state
        if previous_failure:
            self.failureflag = torch.tensor([True])


        ############# Rising edge detection must be placed before fail detection
        target_cubes = getattr(self, "target_cubes", [])
        target_cube_names = getattr(self, "target_cube_names", [])

        if target_cubes and not hasattr(self, "target_cube_pickup_counts"):
            self.target_cube_pickup_counts = {name: 0 for name in target_cube_names}
            self.target_cube_pickup_active = {name: False for name in target_cube_names}


        if target_cubes and not hasattr(self, "target_cube_pickup_active"):
            self.target_cube_pickup_active = {name: False for name in target_cube_names}

        # Only count when cube changes from "not picked" to "picked", avoid duplicate counting in multiple frames for same pick
        for cube, name in zip(target_cubes, target_cube_names):
            pickup_tensor = is_obj_pickup(self, cube)
            if isinstance(pickup_tensor, torch.Tensor):
                picked_now = bool(pickup_tensor.detach().cpu().any())
            else:
                picked_now = bool(pickup_tensor)

            was_picked = self.target_cube_pickup_active.get(name, False)
            if picked_now and not was_picked:
                self.target_cube_pickup_counts[name] = (
                    self.target_cube_pickup_counts.get(name, 0) + 1
                )
            self.target_cube_pickup_active[name] = picked_now

        pickup_counts = getattr(self, "target_cube_pickup_counts", {})
        counts_satisfied = (
            len(pickup_counts) > 0
            and all(count >= 1 for count in pickup_counts.values())
        )
        ############# Rising edge detection must be placed before fail detection


        # Success if all picked at least once (counting discrete pick events)
        # V6 review fix F1 (K2): the four new tiers also require the whole task chain (including the final button) to complete for success; original three tiers still succeed once every target has been picked
        if counts_satisfied and (all_tasks_completed or not is_newvalue_difficulty(self.difficulty)):
            self.successflag = torch.tensor([True])
       
       # Fail if planner finished but not successful
        if all_tasks_completed and not counts_satisfied:
            self.failureflag = torch.tensor([True])
            logger.debug(f"Pickup counts not satisfied: {pickup_counts}")

        if self.failureflag == torch.tensor([True]):
            pass
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


#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


      
        timestep = self.elapsed_steps
        target_cubes = getattr(self, "target_cubes", [])

      

        if is_newvalue_difficulty(self.difficulty):
            # xhard's highlight_count is an interval; this episode's actual highlight count is the number of targets already drawn
            highlight_count = len(target_cubes)
        else:
            highlight_count = min(self._sampling["decision"]["highlight_count"][self.difficulty], len(target_cubes))
        for i in range(highlight_count):
            highlight_obj(
                self,
                target_cubes[i],
                start_step=self._sampling["positions"]["highlight_window"]["start_step"],
                end_step=self._sampling["positions"]["highlight_window"]["end_step"],
                cur_step=timestep,
            )
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
