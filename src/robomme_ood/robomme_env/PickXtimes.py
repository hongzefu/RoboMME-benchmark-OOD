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
from .utils.sampling_config import assert_native_decision, fill_missing_newvalue, split_sampling_config
from .utils import reset_panda
from .utils.difficulty import normalize_robomme_difficulty, is_newvalue_difficulty
from .utils.SceneGenerationError import SceneGenerationError
from .utils.xhard import BLOCK_DISTRACTOR_COLORS, cube_obb2d_exact

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


# ── Original values of the decision/native blocks (newtaskRelease-v3 step 3, mapping in plan section 2.2) ────────
# decision: color count, range of repeated pick-and-place counts, separate position sampling regions for the target cube and the placement disk, extra distractors.
# native: color order and target selection, button position, cube/disk geometry and rejection conditions, recovery and action expansion rules.
# All values are taken from the literals at the call sites before the change; in the original-value stage both blocks equal the original (red line R7).
NATIVE_SAMPLING = {
    "parameters": {
        "cubes_per_color": 1,
        "color_pool": [
            {"rgba": [1, 0, 0, 1], "name": "red"},
            {"rgba": [0, 0, 1, 1], "name": "blue"},
            {"rgba": [0, 1, 0, 1], "name": "green"},
        ],
        "color_and_target_selection": {
            "shuffle": "torch.randperm(len(color_groups))",
            "target_color_idx": "torch.randint(0, len(color_groups), (1,))",
            "target_cube_idx": "torch.randint(0, len(all_cubes), (1,))",
            "note": "the earlier color draw is overridden by the later target cube selection; original draw count and order kept",
        },
        "recovery": "keep the entry-provided fail recover mode and the original inject_fail_grasp draw",
        "task_expansion": "repeatedly pick the same target_cube and place it on target, press the button at the end; non-targets are the complement of the candidates",
    },
    "positions": {
        "button": {
            "center_xy": [-0.2, 0],
            "scale": 1.5,
            "randomize_range_note": "the original call site did not pass randomize_range; keep build_button's parameter default",
        },
        "cube_pose": {
            "half_size": "self.cube_half_size",
            "min_gap": "self.cube_half_size",
            "random_yaw": True,
            "include_existing": False,
            "include_goal": False,
        },
        "target_pose": {
            "radius_factor": 2,
            "thickness": 0.005,
            "min_gap_factor": 2,
            "include_existing": False,
            "include_goal": False,
        },
    },
}


# ── xhard-specific decision (V4 plan 2.4 / 2.21: C1 / G1 / A5 / B2; V5 plan 2.13: L43-L46) ──────
# * target_cube_position_policy: sampling region of target candidate cubes (the three colored cubes). V5 L43 (c) removes the corner bias
#   (deletes the corner_bias key, overturning V4 J5), drawing uniformly in the region like the distractor cubes; V5 L46 half width 0.2 -> 0.25
#   (P1 probe: 10 cm three-cube clusters 33.5% -> 10.2%, demonstrations 13/13).
# * goal_position_policy: the placement disk has its own region parameters (C1: the disk may stay in the middle; values follow the original region, unchanged in V5).
# * distractor: four distractor cubes, one each of yellow/cyan/magenta/4th color (BLOCK_DISTRACTOR_COLORS, V7), placed uniformly in the cube region; V5 L46 half width also 0.25.
# * min_center_dist_m: V5 L44, minimum pairwise center distance (meters) of the 6 cubes (3 colored + 3 distractors), leaving a gap of one cube width.
XHARD_DECISION = {
    "target_cube_position_policy": {"region_center": [-0.1, 0], "region_half_size": 0.25},
    "goal_position_policy": {"region_center": [-0.1, 0], "region_half_size": 0.2},
    "distractor": {
        "colors": [entry["name"] for entry in BLOCK_DISTRACTOR_COLORS],
        "region_center": [-0.1, 0],
        "region_half_size": 0.25,
    },
    "min_center_dist_m": 0.08,
}


def _newvalue_decision(n_distractors):
    """V6 (plan 2.8): decision subtree of one new-value tier -- key structure identical to ``XHARD_DECISION``,
    only truncating distractor cube colors to the first k of ``BLOCK_DISTRACTOR_COLORS``; region, center distance and other fields follow xhard."""
    tree = copy.deepcopy(XHARD_DECISION)
    tree["distractor"]["colors"] = [entry["name"] for entry in BLOCK_DISTRACTOR_COLORS[:n_distractors]]
    return tree


# V6 new-value tier table: distractor count xhard1=1, xhard2=2, xhard3=3, xhard4=3.
NEWVALUE_DECISION = {
    "xhard1": _newvalue_decision(1),
    "xhard2": _newvalue_decision(2),
    "xhard3": _newvalue_decision(3),
    "xhard4": _newvalue_decision(4),
}

# V5 L45: per-cube rejection sampling budget for xhard cubes (colored + distractors) (original three tiers keep spawn_random_cube's default 256, unaffected).
# Plan 2.13 estimate: with the 8 cm center distance, 256 trials leave ~3.5% of episodes unplaceable, 1024 trials ~1%; S3f offline 3000 episodes measured 0 failures at 1024.
XHARD_CUBE_MAX_TRIALS = 1024


def _disk_avoid_obb(target, clearance):
    """Convert the placement disk into a prebuilt OBB ``(center, axes, half extents)`` usable by cube rejection sampling.

    The disk is a visual-only actor with ``add_collision=False``; ``get_actor_obb`` cannot get its mesh, so putting it directly into
    ``avoid`` is silently ignored by ``spawn_random_cube`` (measured 2026-09-22). xhard places the disk before the cubes (G1),
    so its bounding square must be given explicitly: half extent = disk radius + disk gap - the cube's own gap,
    making the axial criterion match the original "disk avoids cubes" circle-box distance criterion and more conservative diagonally.
    """
    p = target.pose.p
    if isinstance(p, torch.Tensor):
        p = p[0].detach().cpu().numpy()
    return (
        np.array(p[:2], dtype=np.float64),
        np.eye(2, dtype=np.float64),
        np.array([clearance, clearance], dtype=np.float64),
    )


def native_blocks(cls):
    """Original ``(decision, native)`` blocks of this env; shared by external export and internal parsing."""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


def _native_decision(cls):
    """Slice the decision block per plan section 2.2 (equals the original in the original-value stage)."""
    return {
        "color": {difficulty: cfg["color"] for difficulty, cfg in cls.configs.items()},
        "number_range": {
            difficulty: [cfg["number_min"], cfg["number_max"]]
            for difficulty, cfg in cls.configs.items()
        },
        # Separate position sampling regions for the target cube and the placement disk; originally both use the same region.
        "target_cube_position_policy": {"region_center": [-0.1, 0], "region_half_size": 0.2},
        "goal_position_policy": {"region_center": [-0.1, 0], "region_half_size": 0.2},
        # Section 2's "add distractors of other colors" is not enabled for the original three tiers (original value stays None).
        "distractor": None,
        # V4 xhard-specific (plan 2.4): key named xhard; the guard only lets this subtree take new values; the part visible to the original three tiers is unchanged.
        # V6 (plan 2.8): append three same-structure subtrees xhard1/2/3 after xhard (new-value family, likewise admitted by the guard).
        "xhard4": copy.deepcopy(XHARD_DECISION),
        **{tier: copy.deepcopy(NEWVALUE_DECISION[tier]) for tier in ("xhard1", "xhard2", "xhard3")},
    }


def _resolve_sampling_config(cls, override):
    """Split out this instance's private decision/native copies; draws no random numbers, must be called before the Generator."""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    # V6: old snapshots (e.g. V5 snapshots) missing xhard1/2/3 subtrees get them filled from source declarations
    fill_missing_newvalue(decision, decision_default)
    native["decision"] = decision
    return native


@register_env("PickXtimes", override=True)
class PickXtimes(BaseEnv):

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
    'color': 3, 
    'number_min': 4,
    'number_max':5,
    }

    config_easy = {
        'color': 1, 
    'number_min': 1,
    'number_max':3
    }

    config_medium = {
        'color': 3, 
    'number_min': 1,
    'number_max':3
    }

    # v8 fixed values (1001 plan 1 table 1 / 2.1): one fixed number per tier, pick-and-place counts 6/7/8/9, distractor cubes 1/2/3/4 (BLOCK_DISTRACTOR_COLORS).
    # xhard4=9 only preserves the four-key structure (v7-compatible configs and tests); not delivered, not evaluated (9 exceeds the 1600-step limit, must not be used for generation);
    # this env adds no xhard5; the disk region (XHARD_DECISION) is a non-gradient parameter, unchanged.
    config_xhard4 = {
        'color': 3,
        'number_min': 9,
        'number_max': 9,
    }

    config_xhard1 = {
        'color': 3,
        'number_min': 6,
        'number_max': 6,
    }

    config_xhard2 = {
        'color': 3,
        'number_min': 7,
        'number_max': 7,
    }

    config_xhard3 = {
        'color': 3,
        'number_min': 8,
        'number_max': 8,
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
        # Must happen before any RNG call and before super().__init__(): one extra or missing draw here would shift every later value
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        # Step 4: read-only export (no spec passed) or original-value re-injection (frozen spec passed)
        self._spec = SpecRecorder(native_episode_spec, "PickXtimes", {"seed": seed},
                                  difficulty=kwargs.get("difficulty"))
        # Initialization index starts at -1; _initialize_episode increments it on each entry;
        # value points in _load_scene use index-free paths, so this is only a fallback.
        self._native_init_index = -1
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
        self.seed = seed
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

        # Use seed to randomly determine number of repetitions (1-5)
        generator = torch.Generator()
        generator.manual_seed(seed)
        number_range = self._sampling["decision"]["number_range"][self.difficulty]
        self.num_repeats = self._spec.value(
            "objects.num_repeats",
            torch.randint(number_range[0], number_range[1]+1, (1,), generator=generator).item(),
            decision_key=f"number_range.{self.difficulty}",
        )
        self._spec.identity.setdefault("difficulty", self.difficulty)
        logger.debug(f"Task will repeat {self.num_repeats} times (pickup-drop cycles)")

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



        generator = torch.Generator()
        generator.manual_seed(self.seed)

        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()



        button_cfg = self._sampling["positions"]["button"]
        button_obb = build_button(
            self,
            center_xy=tuple(button_cfg["center_xy"]),
            scale=button_cfg["scale"],
            generator=generator,
            recorder=self._spec,
            spec_path="layout.button_xy",
        )
        avoid = [button_obb]

       

        # V4: xhard takes an independent generation path (G1 disk first, decoupled target candidate pool, D2 fix, colors backfilled per object),
        # original three tiers still take the original code (moved verbatim into _spawn_scene_objects_native, not a single line changed, H2/N12).
        if is_newvalue_difficulty(self.difficulty):
            self._spawn_scene_objects_xhard(generator, avoid)
        else:
            self._spawn_scene_objects_native(generator, avoid)

                # Dynamically generate task list for N pickup-drop cycles
        tasks = []
        for i in range(self.num_repeats):

            tasks.append({
                "func": (lambda i=i: is_obj_pickup(self, obj=self.target_cube)),
                "name": subgoal_language.get_subgoal_with_index(i, "pick up the {color} cube for the {idx} time", color=self.target_color_name),
                 "subgoal_segment": subgoal_language.get_subgoal_with_index(i, "pick up the {color} cube at <> for the {idx} time", color=self.target_color_name),
                "choice_label": "pick up the cube",
                "demonstration": False,
                "failure_func": lambda: [is_any_obj_pickup(self, self.non_target_cubes),is_button_pressed(self, obj=self.button)],
                "solve": lambda env, planner: solve_pickup(env,planner,self.target_cube),
                "segment":self.target_cube
            })
            tasks.append({
                "func": (lambda: is_obj_dropped_onto(self,obj=self.target_cube,target=self.target)),
                "name": f"place the {self.target_color_name} cube onto the target",
                "subgoal_segment": f"place the {self.target_color_name} cube onto the target at <>",
                "choice_label": "place the cube onto the target",
                "demonstration": False,
                "failure_func": lambda: [is_any_obj_pickup(self, self.non_target_cubes),is_button_pressed(self, obj=self.button)],
                "solve": lambda env, planner: solve_putonto_whenhold(env, planner, target=self.target),
                "segment":self.target
            })

        tasks.append( {
                "func": lambda:is_button_pressed(self, obj=self.button),
                "name": "press the button to stop",
                "subgoal_segment": "press the button at <> to stop",
                "choice_label": "press the button to stop",
                "demonstration": False,
                "failure_func":lambda:is_any_obj_pickup(self, self.all_cubes),
                "solve": lambda env, planner: solve_button(env, planner, obj=self.button),
                "segment":self.cap_link 
            })


        # Store task list for RecordWrapper use
        self.task_list = tasks

                # Record pickup related task indices and items for recovery
        self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
        if self.robomme_failure_recovery:
            # Only inject an intentional failed grasp when recovery mode is enabled
            # Choosing the recovery action is a real draw: the original draw happens as usual; re-injection mode uses the frozen index
            self.fail_grasp_task_index = self._spec.value(
                "actions.recovery.selected_action_index",
                inject_fail_grasp(
                self.task_list,
                generator=generator,
                mode=self.robomme_failure_recovery_mode,
            ),
            )
        else:
            self.fail_grasp_task_index = None

        # V4 xhard: distractor cubes are new random values appended after all existing value points of this function (including the recovery action draw) (N5).
        if is_newvalue_difficulty(self.difficulty):
            self._spawn_distractors_xhard(generator, avoid)

    def _spawn_scene_objects_native(self, generator, avoid):
        """Cube/disk/target cube generation for the original three tiers (the middle of the original ``_load_scene``, moved out verbatim, behavior unchanged)."""
        self.all_cubes = []  # Save all cube objects

        # Initialize storage for each color group
        self.red_cubes = []
        self.red_cube_names = []
        self.blue_cubes = []
        self.blue_cube_names = []
        self.green_cubes = []
        self.green_cube_names = []

        decision_cfg = self._sampling["decision"]
        cube_region = decision_cfg["target_cube_position_policy"]
        goal_region = decision_cfg["goal_position_policy"]
        cube_pose_cfg = self._sampling["positions"]["cube_pose"]
        target_pose_cfg = self._sampling["positions"]["target_pose"]
        cubes_per_color = self._sampling["parameters"]["cubes_per_color"]
        color_groups = [
            {"color": (1, 0, 0, 1), "name": "red", "list": self.red_cubes, "name_list": self.red_cube_names},
            {"color": (0, 0, 1, 1), "name": "blue", "list": self.blue_cubes, "name_list": self.blue_cube_names},
            {"color": (0, 1, 0, 1), "name": "green", "list": self.green_cubes, "name_list": self.green_cube_names}
        ]
        shuffle_indices = self._spec.value(
            "objects.color_order", torch.randperm(len(color_groups), generator=generator).tolist()
        )
        color_groups = [color_groups[i] for i in shuffle_indices]

        # Randomly select target color using generator
        target_color_idx = self._spec.value(
            "objects.target_color_idx",
            torch.randint(0, len(color_groups), (1,), generator=generator).item(),
        )
        self.target_color_name = color_groups[target_color_idx]["name"]
        logger.debug(f"Target color selected: {self.target_color_name}")

        # Generate 5 cubes for each color group
        for idx, group in enumerate(color_groups):
            if idx < decision_cfg["color"][self.difficulty]:
                for idx in range(cubes_per_color):
                    try:
                        cube = spawn_random_cube(
                            self,
                            color=group["color"],
                            avoid=avoid,
                            include_existing=False,
                            include_goal=False,
                            region_center=list(cube_region["region_center"]),
                            region_half_size=cube_region["region_half_size"],
                            half_size=self.cube_half_size,
                            min_gap=self.cube_half_size,
                            random_yaw=cube_pose_cfg["random_yaw"],
                            name_prefix=f"cube_{group['name']}_{idx}",
                            generator=generator,
                            recorder=self._spec,
                            spec_path=f"layout.cubes.{group['name']}_{idx}",
                        )
                    except RuntimeError as e:
                        logger.debug(f"Failed to generate {group['name']} cube {idx}: {e}")
                        break

                    self.all_cubes.append(cube)
                    group["list"].append(cube)
                    cube_name = f"cube_{group['name']}_{idx}"
                    group["name_list"].append(cube_name)
                    setattr(self, cube_name, cube)
                    avoid.append(cube)

            logger.debug(f"Generated {len(group['list'])} {group['name']} cubes")

        logger.debug(f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})")

        try:
            target = spawn_random_target(
                self,
                avoid=avoid,  # Use current avoidance list, containing all spawned cubes
                include_existing=False,  # Manually maintain list
                include_goal=False,  # Manually maintain list
                region_center=list(goal_region["region_center"]),
                region_half_size=goal_region["region_half_size"],
                radius=self.cube_half_size*target_pose_cfg["radius_factor"],  # Use radius instead of half_size
                thickness=target_pose_cfg["thickness"],  # target thickness
                min_gap=self.cube_half_size*target_pose_cfg["min_gap_factor"],  # Gap requirement same as cube
                name_prefix=f"target",
                generator=generator,
                recorder=self._spec,
                spec_path="layout.goal_xy",
            )
        except RuntimeError as e:
            logger.debug(f"Target sampling failed: {e}")


        # Assign target to self.target_0, self.target_1 etc. attributes
        setattr(self, f"target", target)
        # Add newly generated target to avoidance list
        avoid.append(target)


 # Randomly select one cube from all available cubes as the target
        if len(self.all_cubes) > 0:
            target_cube_idx = self._spec.value(
                "objects.target_cube_idx",
                torch.randint(0, len(self.all_cubes), (1,), generator=generator).item(),
            )
            self.target_cube = self.all_cubes[target_cube_idx]

            # Determine the color of the selected target cube
            if self.target_cube in self.red_cubes:
                self.target_color_name = "red"
            elif self.target_cube in self.blue_cubes:
                self.target_color_name = "blue"
            elif self.target_cube in self.green_cubes:
                self.target_color_name = "green"


            logger.debug(f"Target cube selected: {self.target_color_name} cube (index {target_cube_idx} in all_cubes)")
        else:
            self.target_cube = None
            self.target_color_name = None
            logger.debug("No cubes generated, no target cube selected")

        # Create list of non-target cubes for failure checking
        self.non_target_cubes = [cube for cube in self.all_cubes if cube != self.target_cube]
        logger.debug(f"Non-target cubes: {len(self.non_target_cubes)}")

    def _spawn_scene_objects_xhard(self, generator, avoid):
        """Cube/disk/target cube generation for V4 xhard (plan 2.4).

        Differences from the original path (all xhard-only, H2):
        * G1: **disk first, then cubes**; cubes explicitly avoid the disk via ``_disk_avoid_obb``;
        * colors come from ``NATIVE_SAMPLING.parameters.color_pool`` (the single source of truth for the xhard path, no hardcoded literals);
        * V5 (plan 2.13): the three colored cubes are drawn uniformly in the region (L43 removes V4's corner_bias); pairwise center distance of the 6 cubes
          >= ``min_center_dist_m`` (L44, via ``spawn_random_cube(min_center_dist=...)``); placed cubes enter ``avoid`` as
          ``cube_obb2d_exact`` exact OBBs (fixes the 2.0-1 obstacle box degeneracy); per-cube budget ``XHARD_CUBE_MAX_TRIALS`` (L45);
        * D2: if the disk or a cube cannot be placed, raise ``SceneGenerationError`` directly instead of silently truncating or hitting an unbound variable;
        * the target cube is drawn from the explicit candidate list ``self.target_candidates`` (decoupled from ``all_cubes``; distractors are never drawn);
        * ``target_color_name`` is backfilled by per-object lookup instead of the three-color if chain (which would leave a stale value for new colors).
        Relative order of random calls: color_order -> target_color_idx -> disk -> cubes -> target_cube_idx,
        followed by the recovery action and distractor cubes (N5).
        """
        xcfg = self._sampling["decision"][self.difficulty]
        cube_region = xcfg["target_cube_position_policy"]
        goal_region = xcfg["goal_position_policy"]
        min_center_dist = float(xcfg["min_center_dist_m"])
        cube_pose_cfg = self._sampling["positions"]["cube_pose"]
        target_pose_cfg = self._sampling["positions"]["target_pose"]
        cubes_per_color = self._sampling["parameters"]["cubes_per_color"]
        n_colors = self._sampling["decision"]["color"][self.difficulty]
        self._spec.record("layout.cube_min_center_dist", min_center_dist)
        # V5: exact OBBs of placed cubes (colored + distractors share one table); both reference points for the center distance rule and obstacles in avoid
        self._xhard_cube_obbs = []

        self.all_cubes = []
        self.distractor_cubes = []
        self._cube_color_of = []  # [(actor, color name)], used to backfill target_color_name per object
        color_groups = []
        for entry in self._sampling["parameters"]["color_pool"]:
            name = entry["name"]
            cube_list, name_list = [], []
            # Keep attribute names like red_cubes / red_cube_names so downstream code fetching lists by color still works
            setattr(self, f"{name}_cubes", cube_list)
            setattr(self, f"{name}_cube_names", name_list)
            color_groups.append({"color": tuple(entry["rgba"]), "name": name,
                                 "list": cube_list, "name_list": name_list})

        shuffle_indices = self._spec.value(
            "objects.color_order", torch.randperm(len(color_groups), generator=generator).tolist()
        )
        color_groups = [color_groups[i] for i in shuffle_indices]
        # Same draw as the original path (its value is then overridden by the target cube's color), kept to align the set of value points
        target_color_idx = self._spec.value(
            "objects.target_color_idx",
            torch.randint(0, len(color_groups), (1,), generator=generator).item(),
        )
        self.target_color_name = color_groups[target_color_idx]["name"]

        # G1: place the disk first. At this point avoid only contains the button.
        disk_radius = self.cube_half_size * target_pose_cfg["radius_factor"]
        disk_gap = self.cube_half_size * target_pose_cfg["min_gap_factor"]
        try:
            target = spawn_random_target(
                self,
                avoid=avoid,
                include_existing=False,
                include_goal=False,
                region_center=list(goal_region["region_center"]),
                region_half_size=goal_region["region_half_size"],
                radius=disk_radius,
                thickness=target_pose_cfg["thickness"],
                min_gap=disk_gap,
                name_prefix="target",
                generator=generator,
                recorder=self._spec,
                spec_path="layout.goal_xy",
            )
        except RuntimeError as exc:
            # D2 (xhard-only fix): after failure the original path falls through to an unbound target and raises UnboundLocalError
            raise SceneGenerationError(f"PickXtimes xhard: placement disk sampling failed: {exc}") from exc
        self.target = target
        avoid.append(target)
        # The disk itself has no OBB (visual-only actor); cubes avoid it via this prebuilt bounding square
        avoid.append(_disk_avoid_obb(target, disk_radius + disk_gap - self.cube_half_size))

        requested = min(n_colors, len(color_groups)) * cubes_per_color
        for group in color_groups[:n_colors]:
            for cube_idx in range(cubes_per_color):
                cube_name = f"cube_{group['name']}_{cube_idx}"
                try:
                    cube = spawn_random_cube(
                        self,
                        color=group["color"],
                        avoid=avoid,
                        include_existing=False,
                        include_goal=False,
                        region_center=list(cube_region["region_center"]),
                        region_half_size=cube_region["region_half_size"],
                        half_size=self.cube_half_size,
                        min_gap=self.cube_half_size,
                        random_yaw=cube_pose_cfg["random_yaw"],
                        name_prefix=cube_name,
                        generator=generator,
                        recorder=self._spec,
                        spec_path=f"layout.cubes.{group['name']}_{cube_idx}",
                        max_trials=XHARD_CUBE_MAX_TRIALS,
                        min_center_dist=(min_center_dist, self._xhard_cube_obbs),
                    )
                except RuntimeError as exc:
                    raise SceneGenerationError(f"PickXtimes xhard: cube {cube_name} does not fit: {exc}") from exc
                self.all_cubes.append(cube)
                group["list"].append(cube)
                group["name_list"].append(cube_name)
                self._cube_color_of.append((cube, group["name"]))
                setattr(self, cube_name, cube)
                self._append_cube_obstacle_xhard(cube, avoid)
        # 2.2-4: requested vs actual; mismatch fails this episode (already guaranteed by the raise above, recorded and checked explicitly here)
        self._spec.record("objects.cube_count", {"requested": requested, "actual": len(self.all_cubes)})
        if len(self.all_cubes) != requested or requested == 0:
            raise SceneGenerationError(
                f"PickXtimes xhard: colored cubes requested {requested} actual {len(self.all_cubes)}"
            )

        # Target candidate pool decoupled from all_cubes: only colored cubes, so distractor cubes appended later can never be drawn as the target
        self.target_candidates = list(self.all_cubes)
        self._spec.record(
            "objects.target_candidates", [self._color_name_of(cube) for cube in self.target_candidates]
        )
        target_cube_idx = self._spec.value(
            "objects.target_cube_idx",
            torch.randint(0, len(self.target_candidates), (1,), generator=generator).item(),
        )
        self.target_cube = self.target_candidates[target_cube_idx]
        self.target_color_name = self._color_name_of(self.target_cube)
        self.non_target_cubes = [cube for cube in self.all_cubes if cube is not self.target_cube]

    def _color_name_of(self, cube):
        """Look up the color name by object (xhard path); a miss means registration was skipped, so raise instead of leaving a stale value."""
        for actor, name in self._cube_color_of:
            if actor is cube:
                return name
        raise SceneGenerationError("PickXtimes xhard: target cube is not in the color registry")

    def _append_cube_obstacle_xhard(self, cube, avoid):
        """V5 (plan 2.0-1 / 2.13): register the cube just placed as an exact OBB, serving as an obstacle and center distance reference for later cubes.

        The actor itself is no longer put into ``avoid``: the actor path via ``_trimesh_box_to_obb2d`` degenerates into a segment for ~2/3 of
        cube poses, voiding ``min_gap`` along its normal. Pure geometry, no random draws.
        """
        obb = cube_obb2d_exact(cube, self.cube_half_size)
        self._xhard_cube_obbs.append(obb)
        avoid.append(obb)

    def _spawn_distractors_xhard(self, generator, avoid):
        """V4 xhard: place three "other color" distractor cubes (A5/B2: one each of yellow/cyan/magenta).

        Distractor cubes go into ``all_cubes`` and ``non_target_cubes`` (picking one triggers failure_func),
        not into ``target_candidates``. If one cannot be placed raise ``SceneGenerationError`` directly (2.2-4, no silent truncation).
        V5: share the center distance rule, exact OBB obstacles and rejection budget with colored cubes (plan 2.13).
        """
        min_center_dist = float(self._sampling["decision"][self.difficulty]["min_center_dist_m"])
        dcfg = self._sampling["decision"][self.difficulty]["distractor"]
        palette = {entry["name"]: entry["rgba"] for entry in BLOCK_DISTRACTOR_COLORS}
        names = list(dcfg["colors"])
        unknown = [name for name in names if name not in palette]
        if unknown:
            raise SceneGenerationError(f"PickXtimes xhard: distractor color not in BLOCK_DISTRACTOR_COLORS: {unknown}")
        self._spec.record("objects.distractors", [{"name": f"cube_{n}_0", "color": n} for n in names])
        for name in names:
            cube_name = f"cube_{name}_0"
            try:
                cube = spawn_random_cube(
                    self,
                    color=tuple(palette[name]),
                    avoid=avoid,
                    include_existing=False,
                    include_goal=False,
                    region_center=list(dcfg["region_center"]),
                    region_half_size=dcfg["region_half_size"],
                    half_size=self.cube_half_size,
                    min_gap=self.cube_half_size,
                    random_yaw=self._sampling["positions"]["cube_pose"]["random_yaw"],
                    name_prefix=cube_name,
                    generator=generator,
                    recorder=self._spec,
                    spec_path=f"layout.distractors.{name}_0",
                    max_trials=XHARD_CUBE_MAX_TRIALS,
                    min_center_dist=(min_center_dist, self._xhard_cube_obbs),
                )
            except RuntimeError as exc:
                raise SceneGenerationError(f"PickXtimes xhard: distractor cube {cube_name} does not fit: {exc}") from exc
            self.all_cubes.append(cube)
            self.distractor_cubes.append(cube)
            self._cube_color_of.append((cube, name))
            setattr(self, f"{name}_cubes", [cube])
            setattr(self, f"{name}_cube_names", [cube_name])
            setattr(self, cube_name, cube)
            self._append_cube_obstacle_xhard(cube, avoid)
        self._spec.record("objects.distractor_count",
                          {"requested": len(names), "actual": len(self.distractor_cubes)})
        if len(self.distractor_cubes) != len(names):
            raise SceneGenerationError(
                f"PickXtimes xhard: distractor cubes requested {len(names)} actual {len(self.distractor_cubes)}"
            )
        # failure_func reads self.non_target_cubes only when called; rebuilding it here lets distractor cubes take part in failure checks
        self.non_target_cubes = [cube for cube in self.all_cubes if cube is not self.target_cube]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Each initialization records its own spec, never reusing the previous result
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            logger.debug(self.agent.robot.qpos)

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
        all_tasks_completed, current_task_name, task_failed,self.current_task_specialflag= sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

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


#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


        
        # highlight_obj(self,self.target_cube, start_step=0, end_step=30, cur_step=timestep)
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
