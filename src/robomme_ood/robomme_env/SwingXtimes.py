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

from .utils.SceneGenerationError import SceneGenerationError
from .utils import *
# V5 L3 (following VideoPlaceOrder's K2 fix): the `from .utils import *` line above lets the same-named submodule
# `utils.SceneGenerationError` shadow the name `SceneGenerationError` (confirmed by import introspection), so in the original three tiers
# raise / except become TypeError (original three tiers kept as is per H2). xhard uses the alias below to get the real exception class.
from .utils.SceneGenerationError import SceneGenerationError as _RealSceneGenerationError
from .utils.subgoal_evaluate_func import static_check, too_many_swings
from .utils import subgoal_language
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils import reset_panda
from .utils.difficulty import normalize_robomme_difficulty, is_newvalue_difficulty
from .utils.episode_spec import SpecRecorder
from .utils.sampling_config import assert_native_decision, fill_missing_newvalue, split_sampling_config
from .utils.xhard import BLOCK_DISTRACTOR_COLORS, cube_obb2d_exact

from ..logging_utils import logger


def _scene_gen_error(difficulty):
    """V5 L3: select the scene-generation exception class by tier.

    xhard returns the real ``SceneGenerationError`` (retryable task failure); original three tiers return this module's
    shadowed name ``SceneGenerationError`` as is (a submodule, so raise / except still give TypeError; behavior verbatim unchanged).
    Usage: ``raise _scene_gen_error(self.difficulty)("message")``, ``except _scene_gen_error(self.difficulty):``;
    code executed only on the xhard path uses ``_RealSceneGenerationError`` directly.
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


# ── Original values of the decision/native blocks (newtaskRelease-v3 step 3, mapping in plan section 2.3) ────────
# decision: swing round range, color count, extra other-color distractors (not enabled this round).
# native: color order and target selection, regions and geometry of the cube / two disks / button, left-right order and success threshold, recovery rules.
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
        },
        "side_order": {"first": "right", "second": "left", "max_swings": "2 * num_repeats"},
        "swing_thresholds": {"distance": 0.03, "z": 0.12, "height": 0.1},
        "recovery": "keep the entry-provided fail recover mode and the original generator",
    },
    "positions": {
        "button": {"center_xy": [-0.2, 0], "scale": 1.5},
        "cubes": {
            "region_center": [-0.1, 0],
            "region_half_size": 0.25,
            "random_yaw": True,
            "min_gap": "self.cube_half_size",
        },
        "targets": [
            {"region_center": [-0.1, -0.2], "region_half_size": 0.1, "name": "temp_target_0"},
            {"region_center": [-0.1, 0.2], "region_half_size": 0.1, "name": "temp_target_1"},
        ],
        "target_geometry": {"radius_factor": 2, "thickness": 0.005, "min_gap_factor": 1, "style": "gray"},
    },
}


# ── V4 xhard-specific decision (plan 2.5, A5 / B2) ─────────────────────────────────
# distractor: four "other color" distractor cubes, one each of yellow/cyan/magenta/4th color (BLOCK_DISTRACTOR_COLORS, V7),
# region follows the original cube region (center [-0.1,0], half size 0.25, generous capacity).
# Distractor colors are **not merged** into native.color_pool: merging would change the length of randperm(len(color_groups)) and shift the original three tiers' random stream.
# min_center_dist_m: V5 L44 (plan 2.14), minimum pairwise center distance (meters) of the 6 cubes (3 colored + 3 distractors); this env has no corner_bias.
XHARD_DECISION = {
    "distractor": {
        "colors": [entry["name"] for entry in BLOCK_DISTRACTOR_COLORS],
        "region_center": [-0.1, 0],
        "region_half_size": 0.25,
    },
    "min_center_dist_m": 0.08,
}


def _newvalue_decision(n_distractors):
    """V6 (plan 2.8): decision subtree of one new-value tier -- key structure identical to ``XHARD_DECISION``,
    only truncating distractor cube colors to the first k of ``BLOCK_DISTRACTOR_COLORS``; region and center distance follow xhard."""
    tree = copy.deepcopy(XHARD_DECISION)
    tree["distractor"]["colors"] = [entry["name"] for entry in BLOCK_DISTRACTOR_COLORS[:n_distractors]]
    return tree


# V6 new-value tier table: distractor count xhard1=1, xhard2=2, xhard3=3, xhard4=3.
NEWVALUE_DECISION = {
    "xhard1": _newvalue_decision(1),
    "xhard2": _newvalue_decision(2),
    "xhard3": _newvalue_decision(3),
    "xhard4": _newvalue_decision(4),
    # v8 (1001 plan 1 table 1): xhard5 takes 4 distractor cubes -- BLOCK_DISTRACTOR_COLORS only has 4 colors: yellow/cyan/magenta/orange.
    "xhard5": _newvalue_decision(4),
}


def _disk_avoid_obb(target, clearance):
    """Convert a disk into a prebuilt OBB ``(center, axes, half extents)`` usable by cube rejection sampling.

    The disk is a visual-only actor with ``add_collision=False``; ``get_actor_obb`` cannot get its mesh, so putting it directly into
    ``avoid`` is silently ignored by ``spawn_random_cube`` (measured 2026-09-22). Distractor cubes are placed after the disks,
    so the bounding square must be given explicitly: half extent = disk radius + disk gap - the cube's own gap.
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
    """Slice the decision block per plan section 2.3 (equals the original in the original-value stage)."""
    return {
        # One round = one right and one left; original values from the class attributes number_min/number_max.
        "number_range": {
            difficulty: [cfg["number_min"], cfg["number_max"]]
            for difficulty, cfg in cls.configs.items()
        },
        "color": {difficulty: cfg["color"] for difficulty, cfg in cls.configs.items()},
        "distractor": None,
        # V4 xhard-specific (plan 2.5): key named xhard; the guard only lets this subtree take new values; the part visible to the original three tiers is unchanged.
        "xhard4": copy.deepcopy(XHARD_DECISION),
        # V6 (plan 2.8): append three same-structure subtrees xhard1/2/3 after xhard (new-value family, likewise admitted by the guard).
        # v8 (1001 plan 2.1): additionally append xhard5 (only this env and StopCube have this tier, R8).
        **{tier: copy.deepcopy(NEWVALUE_DECISION[tier]) for tier in ("xhard1", "xhard2", "xhard3", "xhard5")},
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


@register_env("SwingXtimes", override=True)
class SwingXtimes(BaseEnv):

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
    'number_min': 3,
    'number_max':3,
    }

    config_easy = {
        'color': 1, 
    'number_min': 1,
    'number_max':3
    }

    config_medium = {
        'color': 3, 
    'number_min': 1,
    'number_max':2
    }

    # v8 fixed values (1001 plan 1 table 1 / 2.1): one fixed number per tier, swing rounds 4/5/6/7/8, distractor cubes 1/2/3/4/4 (BLOCK_DISTRACTOR_COLORS).
    config_xhard4 = {
        'color': 3,
        'number_min': 7,
        'number_max': 7,
    }

    config_xhard1 = {
        'color': 3,
        'number_min': 4,
        'number_max': 4,
    }

    config_xhard2 = {
        'color': 3,
        'number_min': 5,
        'number_max': 5,
    }

    config_xhard3 = {
        'color': 3,
        'number_min': 6,
        'number_max': 6,
    }

    # New in v8: xhard5 swings 8 rounds (4 distractor cubes, see NEWVALUE_DECISION).
    config_xhard5 = {
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
        # v8: xhard5 appended at the end (tests assert the order of the leading keys is unchanged)
        'xhard5': config_xhard5,
    }


    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     sampling_config=None,
                     native_episode_spec=None,
                     **kwargs):
        # Must happen before any RNG call and before super().__init__()
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "SwingXtimes", {"seed": seed},
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

        try:
            self.table_scene = TableSceneBuilder(
                self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            self.table_scene.build()

            button_cfg = self._sampling["positions"]["button"]
            cubes_cfg = self._sampling["positions"]["cubes"]
            targets_cfg = self._sampling["positions"]["targets"]
            target_geom = self._sampling["positions"]["target_geometry"]
            button_obb = build_button(
                self,
                center_xy=tuple(button_cfg["center_xy"]),
                scale=button_cfg["scale"],
                generator=generator,
                recorder=self._spec,
                spec_path="layout.button_xy",
            )
            avoid = [button_obb]

            self.all_cubes = []  # Save all cube objects

            # Initialize storage for each color group
            self.red_cubes = []
            self.red_cube_names = []
            self.blue_cubes = []
            self.blue_cube_names = []
            self.green_cubes = []
            self.green_cube_names = []

            cubes_per_color = self._sampling["parameters"]["cubes_per_color"]
            _color_lists = {
                "red": (self.red_cubes, self.red_cube_names),
                "blue": (self.blue_cubes, self.blue_cube_names),
                "green": (self.green_cubes, self.green_cube_names),
            }
            # V4 (plan 2.5): when the color pool contains names other than red/blue/green, build the table dynamically by name instead of KeyError;
            # the original snapshot only has red/blue/green, so this block is never entered; original three tiers' behavior unchanged.
            for entry in self._sampling["parameters"]["color_pool"]:
                if entry["name"] not in _color_lists:
                    extra_cubes, extra_names = [], []
                    setattr(self, f"{entry['name']}_cubes", extra_cubes)
                    setattr(self, f"{entry['name']}_cube_names", extra_names)
                    _color_lists[entry["name"]] = (extra_cubes, extra_names)
            # Used to backfill color names per object (read on the xhard path; original three tiers only write, never read)
            self._cube_color_of = []
            # Color pool from the snapshot (same order as the original literal: red, blue, green)
            color_groups = [
                {
                    "color": tuple(entry["rgba"]),
                    "name": entry["name"],
                    "list": _color_lists[entry["name"]][0],
                    "name_list": _color_lists[entry["name"]][1],
                }
                for entry in self._sampling["parameters"]["color_pool"]
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

            if is_newvalue_difficulty(self.difficulty):
                # V5 (plan 2.14): xhard's colored cubes take a separate branch -- pairwise center distance rule and exact OBB obstacles;
                # value points, draw count cap and color/name registration are the same as the original code below.
                self._spawn_colored_cubes_xhard(generator, avoid, color_groups)
            else:
                # Original three tiers: the whole block below keeps the original code verbatim (only indented one level), behavior unchanged (H2).
                # Generate cubes for each color group
                for idx, group in enumerate(color_groups):
                    if idx < self._sampling["decision"]["color"][self.difficulty]:
                        for cube_idx in range(cubes_per_color):
                            try:
                                cube = spawn_random_cube(
                                    self,
                                    color=group["color"],
                                    avoid=avoid,
                                    include_existing=False,
                                    include_goal=False,
                                    region_center=list(cubes_cfg["region_center"]),
                                    region_half_size=cubes_cfg["region_half_size"],
                                    half_size=self.cube_half_size,
                                    min_gap=self.cube_half_size,
                                    random_yaw=cubes_cfg["random_yaw"],
                                    name_prefix=f"cube_{group['name']}_{cube_idx}",
                                    generator=generator,
                                    recorder=self._spec,
                                    spec_path=f"layout.cubes.{group['name']}_{cube_idx}",
                                )
                            except RuntimeError as e:
                                raise _scene_gen_error(self.difficulty)(
                                    f"Failed to generate {group['name']} cube {cube_idx}: {e}"
                                ) from e

                            self.all_cubes.append(cube)
                            group["list"].append(cube)
                            cube_name = f"cube_{group['name']}_{cube_idx}"
                            group["name_list"].append(cube_name)
                            self._cube_color_of.append((cube, group["name"]))
                            setattr(self, cube_name, cube)
                            avoid.append(cube)

                    logger.debug(f"Generated {len(group['list'])} {group['name']} cubes")

            logger.debug(f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})")

            # Generate first target
            try:
                temp_target_0 = spawn_random_target(
                    self,
                    avoid=avoid,  # Use current avoidance list, containing all spawned cubes
                    include_existing=False,  # Manually maintain list
                    include_goal=False,  # Manually maintain list
                    region_center=list(targets_cfg[0]["region_center"]),
                    region_half_size=targets_cfg[0]["region_half_size"],
                    radius=self.cube_half_size*target_geom["radius_factor"],  # Use radius instead of half_size
                    thickness=target_geom["thickness"],  # target thickness
                    min_gap=self.cube_half_size*target_geom["min_gap_factor"],  # Gap requirement same as cube
                    name_prefix=f"temp_target_0",
                    generator=generator,
                    target_style="gray",
                    recorder=self._spec,
                    spec_path="layout.targets.0",
                )
                avoid.append(temp_target_0)
                logger.debug(f"Generated first target")
            except RuntimeError as e:
                raise _scene_gen_error(self.difficulty)("First target sampling failed") from e

            # Generate second target
            try:
                temp_target_1 = spawn_random_target(
                    self,
                    avoid=avoid,  # Use current avoidance list, containing all spawned cubes and first target
                    include_existing=False,  # Manually maintain list
                    include_goal=False,  # Manually maintain list
                    region_center=list(targets_cfg[1]["region_center"]),
                    region_half_size=targets_cfg[1]["region_half_size"],
                    radius=self.cube_half_size*target_geom["radius_factor"],  # Use radius instead of half_size
                    thickness=target_geom["thickness"],  # target thickness
                    min_gap=self.cube_half_size*target_geom["min_gap_factor"],  # Gap requirement same as cube
                    name_prefix=f"temp_target_1",
                    generator=generator,
                    target_style="gray",
                    recorder=self._spec,
                    spec_path="layout.targets.1"
                )
                avoid.append(temp_target_1)
                logger.debug(f"Generated second target")
            except RuntimeError as e:
                raise _scene_gen_error(self.difficulty)("Second target sampling failed") from e

            # Swap names if necessary to ensure target_0.y < target_1.y
            temp_0_y = temp_target_0.pose.p[0, 1].item()  # Get y coordinate
            temp_1_y = temp_target_1.pose.p[0, 1].item()  # Get y coordinate

            if temp_0_y < temp_1_y:
                # No swap needed
                self.target_right = temp_target_0
                self.target_left = temp_target_1
                logger.debug(f"target_0 y={temp_0_y:.3f}, target_1 y={temp_1_y:.3f} (no swap needed)")
            else:
                # Swap the assignments
                self.target_right = temp_target_1
                self.target_left = temp_target_0
                logger.debug(f"Swapped: target_0 y={temp_1_y:.3f}, target_1 y={temp_0_y:.3f} (swapped to ensure target_0.y < target_1.y)")

            if is_newvalue_difficulty(self.difficulty):
                # V4 xhard: target candidate pool decoupled from all_cubes, colors backfilled per object (plan 2.5, same structure as PickXtimes)
                self._select_target_xhard(generator)
            # Randomly select one cube from all available cubes as the target
            elif len(self.all_cubes) > 0:
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



        except _scene_gen_error(self.difficulty):  # V5 L3: xhard uses the real class; original three tiers still use the shadowed original name
            raise
        except Exception as exc:
            raise _scene_gen_error(self.difficulty)(
                f"Failed to load SwingXtimes scene for seed {self.seed}"
            ) from exc
        

        tasks = []
        tasks.append({
                "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                "name": f"pick up the {self.target_color_name} cube",
                "subgoal_segment":f"pick up the {self.target_color_name} cube at <>",
                "choice_label": "pick up the cube",
                "demonstration": False,
                "failure_func": lambda: [is_any_obj_pickup(self, self.non_target_cubes),is_button_pressed(self, obj=self.button),too_many_swings(self)],
                "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
                'segment':self.target_cube,
            })

        # Swing success threshold and lift height taken from the snapshot (original values distance 0.03 / z 0.12 / height 0.1)
        _swing_cfg = self._sampling["parameters"]["swing_thresholds"]
        for i in range(self.num_repeats):
            # V6 review fix N10 (user "n10 fix a"): ordinals now use the shared ordinal table subgoal_language._ordinal_word --
            # the first ten entries are verbatim identical to the original local list (text of the original three tiers and xhard1-3 unchanged); from round 11 it gives eleventh...twentieth instead of 11th
            ordinal = subgoal_language._ordinal_word(i)
            tasks.append({
                "func": (lambda: is_obj_swing_onto(self,obj=self.target_cube,target=self.target_right,distance_threshold=_swing_cfg["distance"],z_threshold=_swing_cfg["z"])),
                "name": f"move to the top of the right-side target for the {ordinal} time",
                "subgoal_segment":f"move to the top of the right-side target at <> for the {ordinal} time",
                "choice_label": "move to the top of the target",
                "demonstration": False,
                "failure_func": lambda:  [is_any_obj_pickup(self, self.non_target_cubes),is_button_pressed(self, obj=self.button),too_many_swings(self)],
                # "solve": lambda env, planner: [solve_swingonto_whenhold(env, planner,target=self.target_right,height=_swing_cfg["height"]),
                #                             ],
                "solve": lambda env, planner: [solve_swingonto_whenhold(env, planner,target=self.target_right,height=_swing_cfg["height"]),
                                                # solve_swingonto_whenhold(env, planner,target=self.target_right,height=0.15),
                                                # solve_swingonto_whenhold(env, planner,target=self.target_right,height=_swing_cfg["height"]),
                                            ],
                'segment':self.target_right,
            })
            tasks.append({
                "func": (lambda: is_obj_swing_onto(self,obj=self.target_cube,target=self.target_left,distance_threshold=_swing_cfg["distance"],z_threshold=_swing_cfg["z"])),
                "name": f"move to the top of the left-side target for the {ordinal} time",
                "subgoal_segment":f"move to the top of the left-side target at <> for the {ordinal} time",
                "choice_label": "move to the top of the target",
                "demonstration": False,
                "failure_func": lambda:  [is_any_obj_pickup(self, self.non_target_cubes),is_button_pressed(self, obj=self.button),too_many_swings(self)],
                "solve": lambda env, planner: [solve_swingonto_whenhold(env, planner, target=self.target_left,height=_swing_cfg["height"]),
                                            ],
                'segment':self.target_left,
            })


        tasks.append({
                "func": (lambda: is_obj_dropped(self, obj=self.target_cube)),
                "name": f"put the {self.target_color_name} cube on the table",
                "subgoal_segment":f"put the {self.target_color_name} cube on the table",
                "choice_label": "put the cube on the table",
                "demonstration": False,
                "failure_func":  lambda: [is_any_obj_pickup(self, self.non_target_cubes),is_button_pressed(self, obj=self.button),too_many_swings(self)],
                "solve": lambda env, planner: solve_putdown_whenhold(env, planner,),
            })
        tasks.append({
                "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment":"press the button at <>",
                "choice_label": "press the button",
                "demonstration": False,
                "failure_func":lambda:[is_any_obj_pickup(self, self.non_target_cubes),too_many_swings(self)],
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

    def _color_name_of(self, cube):
        """Look up the color name by object (xhard path); a miss means registration was skipped, so raise instead of leaving a stale value."""
        for actor, name in self._cube_color_of:
            if actor is cube:
                return name
        raise _RealSceneGenerationError("SwingXtimes xhard: target cube is not in the color registry")

    def _select_target_xhard(self, generator):
        """V4 xhard target cube selection: draw from the explicit candidate list, colors backfilled per object.

        The draw position is the same as the original path's ``objects.target_cube_idx`` (one randint), only the upper bound is the candidate pool length;
        the candidate pool only has colored cubes, so distractor cubes appended later can never be drawn as the target.
        """
        self._spec.record("objects.cube_count", {
            "requested": min(self._sampling["decision"]["color"][self.difficulty],
                             len(self._sampling["parameters"]["color_pool"]))
            * self._sampling["parameters"]["cubes_per_color"],
            "actual": len(self.all_cubes),
        })
        self.target_candidates = list(self.all_cubes)
        if not self.target_candidates:
            raise _RealSceneGenerationError("SwingXtimes xhard: no target candidate cube available")
        self._spec.record(
            "objects.target_candidates", [self._color_name_of(cube) for cube in self.target_candidates]
        )
        target_cube_idx = self._spec.value(
            "objects.target_cube_idx",
            torch.randint(0, len(self.target_candidates), (1,), generator=generator).item(),
        )
        self.target_cube = self.target_candidates[target_cube_idx]
        self.target_color_name = self._color_name_of(self.target_cube)
        self.distractor_cubes = []
        self.non_target_cubes = [cube for cube in self.all_cubes if cube is not self.target_cube]

    def _append_cube_obstacle_xhard(self, cube, avoid):
        """V5 (plan 2.0-1 / 2.14): register the cube just placed as an exact OBB, serving as an obstacle and center distance reference for later objects.

        The actor itself is no longer put into ``avoid``: the actor path via ``_trimesh_box_to_obb2d`` degenerates into a segment for ~2/3 of
        cube poses, voiding ``min_gap`` along its normal. Pure geometry, no random draws.
        """
        obb = cube_obb2d_exact(cube, self.cube_half_size)
        self._xhard_cube_obbs.append(obb)
        avoid.append(obb)

    def _spawn_colored_cubes_xhard(self, generator, avoid, color_groups):
        """V5 xhard (plan 2.14): place the three colored cubes (target candidates).

        Only two differences from the loop shared with the original three tiers: candidate centers keep pairwise distance >= ``min_center_dist_m`` to placed cubes (L44, via
        ``spawn_random_cube(min_center_dist=...)``, which draws no random numbers itself); placed cubes enter ``avoid`` as ``cube_obb2d_exact``
        exact OBBs (the two disks and distractor cubes after them avoid accordingly). Region, gap, yaw, value point paths and
        per-cube rejection budget (default 256) are the same as the original loop; on failure raise a real ``SceneGenerationError`` (L3).
        """
        cubes_cfg = self._sampling["positions"]["cubes"]
        cubes_per_color = self._sampling["parameters"]["cubes_per_color"]
        min_center_dist = float(self._sampling["decision"][self.difficulty]["min_center_dist_m"])
        self._spec.record("layout.cube_min_center_dist", min_center_dist)
        # Exact OBBs of placed cubes (colored + distractors share one table); both reference points for the center distance rule and obstacles in avoid
        self._xhard_cube_obbs = []
        for idx, group in enumerate(color_groups):
            if idx < self._sampling["decision"]["color"][self.difficulty]:
                for cube_idx in range(cubes_per_color):
                    cube_name = f"cube_{group['name']}_{cube_idx}"
                    try:
                        cube = spawn_random_cube(
                            self,
                            color=group["color"],
                            avoid=avoid,
                            include_existing=False,
                            include_goal=False,
                            region_center=list(cubes_cfg["region_center"]),
                            region_half_size=cubes_cfg["region_half_size"],
                            half_size=self.cube_half_size,
                            min_gap=self.cube_half_size,
                            random_yaw=cubes_cfg["random_yaw"],
                            name_prefix=cube_name,
                            generator=generator,
                            recorder=self._spec,
                            spec_path=f"layout.cubes.{group['name']}_{cube_idx}",
                            min_center_dist=(min_center_dist, self._xhard_cube_obbs),
                        )
                    except RuntimeError as exc:
                        raise _RealSceneGenerationError(
                            f"SwingXtimes xhard: cube {cube_name} does not fit: {exc}"
                        ) from exc

                    self.all_cubes.append(cube)
                    group["list"].append(cube)
                    group["name_list"].append(cube_name)
                    self._cube_color_of.append((cube, group["name"]))
                    setattr(self, cube_name, cube)
                    self._append_cube_obstacle_xhard(cube, avoid)

                logger.debug(f"Generated {len(group['list'])} {group['name']} cubes")

    def _spawn_distractors_xhard(self, generator, avoid):
        """V4 xhard: place three "other color" distractor cubes (A5/B2: one each of yellow/cyan/magenta).

        Distractor cubes go into ``all_cubes`` and ``non_target_cubes`` (picking one triggers failure_func),
        not into ``target_candidates``; the two disks are explicitly avoided via ``_disk_avoid_obb``.
        If one cannot be placed raise ``SceneGenerationError`` directly (2.2-4, no silent truncation).
        V5: share the center distance rule and exact OBB obstacles with colored cubes (plan 2.14).
        """
        min_center_dist = float(self._sampling["decision"][self.difficulty]["min_center_dist_m"])
        dcfg = self._sampling["decision"][self.difficulty]["distractor"]
        palette = {entry["name"]: entry["rgba"] for entry in BLOCK_DISTRACTOR_COLORS}
        names = list(dcfg["colors"])
        unknown = [name for name in names if name not in palette]
        if unknown:
            raise _RealSceneGenerationError(f"SwingXtimes xhard: distractor color not in BLOCK_DISTRACTOR_COLORS: {unknown}")
        target_geom = self._sampling["positions"]["target_geometry"]
        clearance = self.cube_half_size * (target_geom["radius_factor"] + target_geom["min_gap_factor"]) \
            - self.cube_half_size
        avoid = list(avoid) + [_disk_avoid_obb(disk, clearance) for disk in (self.target_right, self.target_left)]
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
                    random_yaw=self._sampling["positions"]["cubes"]["random_yaw"],
                    name_prefix=cube_name,
                    generator=generator,
                    recorder=self._spec,
                    spec_path=f"layout.distractors.{name}_0",
                    min_center_dist=(min_center_dist, self._xhard_cube_obbs),
                )
            except RuntimeError as exc:
                raise _RealSceneGenerationError(f"SwingXtimes xhard: distractor cube {cube_name} does not fit: {exc}") from exc
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
            raise _RealSceneGenerationError(
                f"SwingXtimes xhard: distractor cubes requested {len(names)} actual {len(self.distractor_cubes)}"
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
            self.highlight_right_start = None
            self.highlight_left_start = None
            # Swing count initialization:
            # swing_count     Record cumulative swing/landing counts (left and right each count as one)
            # swing_over_limit Mark whether allowed count is exceeded, if True then failed
            # _was_on_right/_was_on_left Used for "edge detection" to prevent duplicate counting of same landing across multiple frames
            self.swing_count = 0
            self.swing_over_limit = False
            self._was_on_right = False
            self._was_on_left = False
            # Expected max swing count (left and right each self.num_repeats times)
            self.max_swings = self.num_repeats * 2

    def _get_obs_extra(self, info: Dict):
        return dict()




    def evaluate(self,solve_complete_eval=False):
        previous_failure = getattr(self, "failureflag", None)
        self.successflag = torch.tensor([False])
        if previous_failure is not None and bool(previous_failure.item()):
            self.failureflag = previous_failure
        else:
            self.failureflag = torch.tensor([False])

        # To test "exceed swing limit" scenario, forcibly lower limit (e.g. 1 time)
        # This way second landing triggers too_many_swings failure logic, for easy verification
        #self.max_swings = 2



       
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


#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


        obs, reward, terminated, truncated, info = super().step(action)
        # First check if current frame "lands" on left/right targets, for highlight and counting
        # Note: When policy jitters in z-axis, is_obj_swing_onto's z_threshold check might cause on/off flipping,
        # triggering multiple "False->True" edges and duplicate counting. Here use enter/exit hysteresis thresholds to mitigate jitter.
        # To prevent z-axis jitter near threshold from causing on/off flipping and duplicate counting:
        # Use enter/exit two sets of thresholds (hysteresis).
        # - enter strictly: entering target region counts as one-time landing
        # - exit loosely: small jitter within target region won't be misjudged as leaving
        swing_enter_distance_threshold = 0.03
        swing_exit_distance_threshold = 0.04  # >= enter
        swing_enter_z_threshold = 0.12
        swing_exit_z_threshold = 0.3  # >= enter
        
        if self._was_on_right:
            on_right = is_obj_swing_onto(
                self,
                obj=self.target_cube,
                target=self.target_right,
                distance_threshold=swing_exit_distance_threshold,
                z_threshold=swing_exit_z_threshold,
            )
        else:
            on_right = is_obj_swing_onto(
                self,
                obj=self.target_cube,
                target=self.target_right,
                distance_threshold=swing_enter_distance_threshold,
                z_threshold=swing_enter_z_threshold,
            )

        if self._was_on_left:
            on_left = is_obj_swing_onto(
                self,
                obj=self.target_cube,
                target=self.target_left,
                distance_threshold=swing_exit_distance_threshold,
                z_threshold=swing_exit_z_threshold,
            )
        else:
            on_left = is_obj_swing_onto(
                self,
                obj=self.target_cube,
                target=self.target_left,
                distance_threshold=swing_enter_distance_threshold,
                z_threshold=swing_enter_z_threshold,
            )
        if on_right:
             self.highlight_right_start=int(self.elapsed_steps[0].item())
             # Only accumulate swing count on "first" landing, avoid duplicate counting across continuous frames
             if not self._was_on_right:
                self.swing_count += 1
        if on_left:
             self.highlight_left_start=int(self.elapsed_steps[0].item())
             # Only accumulate swing count on "first" landing, avoid duplicate counting across continuous frames
             if not self._was_on_left:
                self.swing_count += 1

        # Update status after recording edge detection
        self._was_on_right = on_right
        self._was_on_left = on_left

        if self.swing_count > self.max_swings:
            if not self.swing_over_limit:
                # Print only once, warn swing count exceeded limit
                logger.debug(f"Swing count exceeded: {self.swing_count}>{self.max_swings}")
            self.swing_over_limit = True
             
        if self.highlight_right_start is not None:
            cur_step = int(self.elapsed_steps[0].item())

            highlight_obj(
                    self,
                    self.target_right,
                    start_step=self.highlight_right_start,
                    end_step=self.highlight_right_start+20,
                    cur_step=cur_step,
                    disk_radius=self.cube_half_size*2*1.003,
                    disk_half_length=0.005*2*1.4,
                    use_target_style=True,
                    highlight_color=[1.0, 0.0, 0.0, 1.0],
                )
        if self.highlight_left_start is not None:
            cur_step = int(self.elapsed_steps[0].item())

            highlight_obj(
                    self,
                    self.target_left,
                    start_step=self.highlight_left_start,
                    end_step=self.highlight_left_start+20,
                    cur_step=cur_step,
                    disk_radius=self.cube_half_size*2*1.003,
                    disk_half_length=0.005*2*1.4,
                    use_target_style=True,
                    highlight_color=[1.0, 0.0, 0.0, 1.0],
                )
        return obs, reward, terminated, truncated, info
