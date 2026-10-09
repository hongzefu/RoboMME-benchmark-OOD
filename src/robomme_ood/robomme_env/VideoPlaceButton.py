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
from .utils.SceneGenerationError import SceneGenerationError
from .utils.subgoal_evaluate_func import static_check
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils import reset_panda
from .utils import difficulty as difficulty_utils
from .utils.difficulty import normalize_robomme_difficulty
from .utils.episode_spec import SpecRecorder
from .utils.sampling_config import assert_native_decision, split_sampling_config
from .utils.xhard_home_site import (
    RETURN_TO_ORIGIN,
    build_goal_drop_sites,
    goal_drop_obstacles,
    build_home_sites,
    home_pose_record,
    returned_mask,
    validate_demo_plan,
    validate_place_sequence,
)
from .utils.xhard import cube_obb2d_exact

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


# ── Original values of the decision/native blocks (newtaskRelease-v3 step 3, mapping in plan section 2.11) ────────
NATIVE_SAMPLING = {
    "parameters": {
        "color_pool": [
            {"rgba": [1, 0, 0, 1], "name": "red"},
            {"rgba": [0, 0, 1, 1], "name": "blue"},
            {"rgba": [0, 1, 0, 1], "name": "green"},
        ],
        "color_order": {"sampler": "torch.randperm(3)"},
        "target_selection": {"sampler": "torch.randint(0, len(all_cubes))"},
        "task_mapping": "before->target_0, after->target_1; demonstration target_0->button->target_1->goal_site",
        "swap_duration_steps": 50,
        "goal_site_z_override": -0.05,
        "recovery": "keep the entry-provided fail recover mode and the original generator",
        "target_slots": 4,
    },
    "positions": {
        "goal": {"region_center": [-0.1, 0], "region_half_size": 0.1,
                  "radius_factor": 3, "thickness": 0.005},
        "button": {"center_xy": [0.1, 0], "scale": 1.5, "randomize_range": [0.05, 0.3]},
        "cubes": {"region_center": [0, 0], "region_half_size": 0.2, "random_yaw": True},
        "targets": {"region_center": [0, 0], "region_half_size": 0.2,
                     "radius_factor": 2, "thickness": 0.005, "min_gap_factor": 1},
    },
}


def native_blocks(cls, *, release="newtask-v6"):
    """Original ``(decision, native)`` blocks of this env; shared by external export and internal parsing."""
    # newtask-v7: same parsing path as v6, using current class constants (i.e. V7 fixed values; 0928 plan R3)
    if release not in {"newtask-v4", "newtask-v5", "newtask-v6", "newtask-v7"}:
        raise ValueError(f"unknown sampling_config release: {release}")
    native = copy.deepcopy(NATIVE_SAMPLING)
    # Plan section 2 lists color under native ("rule unchanged, only the per-episode value is generated externally"), not decision:
    # this round does not require changing the color count; it just takes the original value per difficulty.
    native["parameters"]["color"] = _tier_config_values(cls, "color", release)
    return _native_decision(cls, release=release), native


# V4 xhard demonstration decision (plan 2.14): the two original keys demo_object_count=1 / native_random_goal_site stay unchanged,
# xhard values live under the subkey named xhard (after the guard strips xhard, the part visible to the original three tiers is verbatim identical to the original).
XHARD_DEMO_DECISION = {"demo_object_count": 2, "demo_return_policy": "return_to_origin"}
V6_DEMO_DECISIONS = {
    # Total target placements = before/after twice per demonstration cube + the specified extra segments; the return-home steps do not count toward the gradient.
    # Extra segments are fixed to before/after the button and use non-answer tables.
    "xhard1": {"demo_object_count": 1, "demo_return_policy": "return_to_origin",
               "extra_place_before": 0, "extra_place_after": 1},
    "xhard2": {"demo_object_count": 1, "demo_return_policy": "return_to_origin",
               "extra_place_before": 1, "extra_place_after": 1},
    "xhard3": {"demo_object_count": 2, "demo_return_policy": "return_to_origin",
               "extra_place_before": 1, "extra_place_after": 0},
    "xhard4": {"demo_object_count": 2, "demo_return_policy": "return_to_origin",
               "extra_place_before": 1, "extra_place_after": 1},
}
_NATIVE_TIERS = ("easy", "medium", "hard")


def _is_newvalue_difficulty(value):
    """Prefer the unified difficulty family check; before the VP copy is merged, use a local compatible set to support targeted tests."""
    predicate = getattr(difficulty_utils, "is_newvalue_difficulty", None)
    if predicate is not None:
        return bool(predicate(value))
    return isinstance(value, str) and value.strip().lower() in {"xhard1", "xhard2", "xhard3", "xhard4"}


def _tier_config_values(cls, field, release):
    if release in {"newtask-v4", "newtask-v5"}:
        values = {}
        for name, cfg in cls.configs.items():
            if name in _NATIVE_TIERS:
                values[name] = cfg[field]
            elif name in {"xhard", "xhard4"}:
                # V4/V5 xhard uses the V5 baseline config_xhard (targets 4); V6 N2 changed config_xhard4's targets to 5,
                # which must not flow back into the V5 snapshot's xhard values
                values["xhard"] = getattr(cls, "config_xhard", cfg)[field]
        return values
    values = {}
    for tier in (*_NATIVE_TIERS, "xhard1", "xhard2", "xhard3", "xhard4"):
        source = getattr(cls, f"config_{tier}", None)
        if source is None and tier.startswith("xhard"):
            source = getattr(cls, "config_xhard4", None) or getattr(cls, "config_xhard", None)
        if source is not None:
            values[tier] = source[field]
    return values


def vpb_target_placement_count(tier_cfg):
    """The V6 gradient counts only segments placing onto a target; the final return home does not count."""
    return (2 * int(tier_cfg["demo_object_count"])
            + int(tier_cfg.get("extra_place_before", 0))
            + int(tier_cfg.get("extra_place_after", 0)))


def _extra_place_owners(demo_count, before_count, after_count, generator):
    sides = ["before"] * int(before_count) + ["after"] * int(after_count)
    if not sides:
        return sides, []
    if demo_count == 1:
        return sides, [0] * len(sides)
    if demo_count != 2 or len(sides) > 2:
        raise SceneGenerationError("VPB V6 extra tables only support 1-2 demonstration cubes and at most 2 extra segments")
    if len(sides) == 1:
        owner_ids = [int(torch.randint(0, demo_count, (1,), generator=generator).item())]
    elif len(sides) == 2:
        owner_ids = torch.randperm(demo_count, generator=generator).tolist()
    else:
        owner_ids = []
    return sides, [int(owner) for owner in owner_ids]


def _native_decision(cls, *, release="newtask-v6"):
    """Slice the decision block per plan section 2.11 (equals the original in the original-value stage)."""
    decision = {
        # How many cubes are manipulated in the demonstration video: original value 1 (the current single target_cube); section 2's 2 is not enabled this round.
        "demo_object_count": 1,
        # Where each cube goes after its demonstration: original value = finally placed on the same random goal_site.
        "demo_return_policy": "native_random_goal_site",
        "targets": _tier_config_values(cls, "targets", release),
        "swap": _tier_config_values(cls, "swap", release),
        "additional_place": _tier_config_values(cls, "additional_place", release),
    }
    if release in {"newtask-v4", "newtask-v5"}:
        decision["xhard"] = dict(XHARD_DEMO_DECISION)
    else:
        decision.update(copy.deepcopy(V6_DEMO_DECISIONS))
    return decision


def _resolve_sampling_config(cls, override):
    """Split out this instance's private decision/native copies; draws no random numbers, must be called before the Generator."""
    incoming_decision = override.get("decision", {}) if isinstance(override, dict) else {}
    legacy_v5 = isinstance(incoming_decision, dict) and "xhard" in incoming_decision and not any(
        tier in incoming_decision for tier in V6_DEMO_DECISIONS
    )
    decision_default, native_default = native_blocks(
        cls, release="newtask-v5" if legacy_v5 else "newtask-v6"
    )
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    native["decision"] = decision
    return native


@register_env("VideoPlaceButton", override=True)
class VideoPlaceButton(BaseEnv):
   
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
        'color': 1, 
        "additional_place":False,
        "swap":False,
        "targets":3
    }
    config_medium= {
        'color': 3, 
        "additional_place":False,
        "swap":False,
        "targets":4
    }
    config_hard = {
        'color': 3, 
        "additional_place":False,
        "swap":True,
        "targets":4
    }


    # V4 xhard (derived from hard, plan 2.14): color 3, targets 4, swap True all unchanged;
    # what changes is "how many cubes are demonstrated and where they go afterward", see decision's xhard subkey (XHARD_DEMO_DECISION).
    config_xhard = {
        'color': 3,
        "additional_place":False,
        "swap":True,
        "targets":4
    }
    config_xhard1 = copy.deepcopy(config_xhard)
    config_xhard2 = copy.deepcopy(config_xhard)
    # V6 review fix N2 (user "n2 agree to change to 5 tables"): xhard3/4 demonstrate 2 cubes, each with before/after tables, 4 in total;
    # with 4 tables the extra-table candidates always equal the after-table set (the post-button extra placement becomes an in-place no-op), so tables 4->5; xhard1/2 still 4 tables.
    config_xhard3 = copy.deepcopy(config_xhard)
    config_xhard3["targets"] = 5
    config_xhard4 = copy.deepcopy(config_xhard)
    config_xhard4["targets"] = 5

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
                     native_episode_spec=None,
                     **kwargs):
        # Must happen before any RNG call and before super().__init__()
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "VideoPlaceButton", {"seed": seed},
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
        self._episode_rng = torch.Generator()
        self._episode_rng.manual_seed(seed)

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

        self.onto_goalsite=False
        self.start_step=99999
        self.end_step=99999
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
            try:
                self.goal_site = spawn_random_target(
                    self,
                    avoid=None,  # Use current avoidance list, containing all spawned cubes
                    include_existing=False,  # Manually maintain list
                    include_goal=False,  # Manually maintain list
                    region_center=list(self._sampling["positions"]["goal"]["region_center"]),
                    region_half_size=self._sampling["positions"]["goal"]["region_half_size"],
                    radius=self.cube_half_size * self._sampling["positions"]["goal"]["radius_factor"],  # Use radius instead of half_size
                    thickness=0.005,  # target thickness
                    min_gap=self.cube_half_size * 1,  # Gap requirement same as cube
                    name_prefix=f"goal_site",
                    recorder=self._spec,
                    spec_path="layout.goal_xy",
                    generator=generator,
                )
            except RuntimeError as exc:
                raise SceneGenerationError("goal_site sampling failed") from exc
            avoid = [self.goal_site]

            button_cfg = self._sampling["positions"]["button"]
            cubes_cfg = self._sampling["positions"]["cubes"]
            targets_cfg = self._sampling["positions"]["targets"]
            decision_cfg = self._sampling["decision"]
            button_obb = build_button(
                self,
                center_xy=tuple(button_cfg["center_xy"]),
                scale=button_cfg["scale"],
                generator=generator,
                randomize_range=tuple(button_cfg["randomize_range"]),
                recorder=self._spec,
                spec_path="layout.button_xy",
            )
            avoid.append(button_obb)

            self.all_cubes = []  # Save all cube objects

            # Initialize storage for each color group
            self.red_cubes = []
            self.red_cube_names = []
            self.blue_cubes = []
            self.blue_cube_names = []
            self.green_cubes = []
            self.green_cube_names = []

            cubes_per_color = 1
            color_groups = [
                {"color": (1, 0, 0, 1), "name": "red", "list": self.red_cubes, "name_list": self.red_cube_names},
                {"color": (0, 0, 1, 1), "name": "blue", "list": self.blue_cubes, "name_list": self.blue_cube_names},
                {"color": (0, 1, 0, 1), "name": "green", "list": self.green_cubes, "name_list": self.green_cube_names},
            ]
            self._spec.identity.setdefault("difficulty", getattr(self, "difficulty", None))
            shuffle_indices = self._spec.value(
                "objects.color_order", torch.randperm(len(color_groups), generator=generator).tolist()
            )
            color_groups = [color_groups[i] for i in shuffle_indices]

            self.target_color_name = color_groups[0]["name"]
            logger.debug(f"Target color selected: {self.target_color_name}")

            # Generate cubes for each color group
            for idx, group in enumerate(color_groups):
                if idx < self._sampling["parameters"]["color"][self.difficulty]:
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
                        except RuntimeError as exc:
                            raise SceneGenerationError(
                                f"Failed to generate {group['name']} cube {cube_idx}: {exc}"
                            ) from exc

                        self.all_cubes.append(cube)
                        group["list"].append(cube)
                        cube_name = f"cube_{group['name']}_{cube_idx}"
                        group["name_list"].append(cube_name)
                        setattr(self, cube_name, cube)
                        if self.difficulty == "xhard" or _is_newvalue_difficulty(self.difficulty):
                            # V5 (L2 b, plan 2.16): placed cubes enter avoid as exact 2D obstacles; later cubes and target tables
                            # (both kinds of spawn calls) avoid accordingly. The actor path via _trimesh_box_to_obb2d degenerates into a segment ~2/3 of the time,
                            # voiding min_gap along its normal; take initial_pose (does not depend on simulation being initialized), no random draws.
                            # Original three tiers still pass the actor; the spawn calls themselves are verbatim unchanged.
                            avoid.append(cube_obb2d_exact(cube.initial_pose, self.cube_half_size))
                        else:
                            avoid.append(cube)

                logger.debug(f"Generated {len(group['list'])} {group['name']} cubes")

            logger.debug(
                f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})"
            )

            self.targets = []
            target_slots = int(self._sampling["parameters"]["target_slots"])
            if _is_newvalue_difficulty(self.difficulty):
                # V6 N2: new-value tier table count (5 for xhard3/4) can exceed the native 4 slots, so the loop cap is relaxed per decision.targets; original three tiers still read the original value
                target_slots = max(target_slots, int(decision_cfg["targets"][self.difficulty]))
            for i in range(target_slots):
                if i < decision_cfg["targets"][self.difficulty]:
                    try:
                        target = spawn_random_target(
                            self,
                            avoid=avoid,  # Use current avoidance list, containing all spawned cubes
                            include_existing=False,  # Manually maintain list
                            include_goal=False,  # Manually maintain list
                            region_center=list(targets_cfg["region_center"]),
                            region_half_size=targets_cfg["region_half_size"],
                            radius=self.cube_half_size * targets_cfg["radius_factor"],  # Use radius instead of half_size
                            thickness=targets_cfg["thickness"],  # target thickness
                            min_gap=self.cube_half_size * targets_cfg["min_gap_factor"],  # Gap requirement same as cube
                            name_prefix=f"target_{i}",
                            generator=generator,
                            recorder=self._spec,
                            spec_path=f"layout.targets.{i}",
                        )
                    except RuntimeError as exc:
                        raise SceneGenerationError(f"Target {i + 1} sampling failed: {exc}") from exc

                    self.targets.append(target)
                    setattr(self, f"target_{i}", target)
                    avoid.append(target)

            if self.difficulty == "xhard" or _is_newvalue_difficulty(self.difficulty):
                # V6 new tiers and V5 old specs share the demonstration template; the code below for the original three tiers is not changed by a single line
                self._load_scene_xhard_tail(generator)
                return
            # Original three tiers: demonstrate 1 cube, place on a random goal_site (the guard already ensures original values); read once here as the real consumption point, no random draws
            validate_demo_plan(decision_cfg["demo_object_count"], decision_cfg["demo_return_policy"],
                               self.difficulty, len(self.all_cubes))

            if len(self.all_cubes) > 0:
                target_cube_idx = self._spec.value(
                    "objects.target_cube_idx",
                    torch.randint(0, len(self.all_cubes), (1,), generator=generator).item(),
                )
                self.target_cube = self.all_cubes[target_cube_idx]

                if self.target_cube in self.red_cubes:
                    self.target_color_name = "red"
                elif self.target_cube in self.blue_cubes:
                    self.target_color_name = "blue"
                elif self.target_cube in self.green_cubes:
                    self.target_color_name = "green"

                logger.debug(
                    f"Target cube selected: {self.target_color_name} cube (index {target_cube_idx} in all_cubes)"
                )
            else:
                self.target_cube = None
                self.target_color_name = None
                logger.debug("No cubes generated, no target cube selected")

            self.non_target_cubes = [cube for cube in self.all_cubes if cube != self.target_cube]
            logger.debug(f"Non-target cubes: {len(self.non_target_cubes)}")

            self.swap_target_a = None
            self.swap_target_b = None
            self.swap_target_other = []

            if self._sampling["decision"]["swap"][self.difficulty] == True:
                if len(self.targets) >= 2:
                    perm = torch.randperm(len(self.targets), generator=generator)
                    swap_idx_a = perm[0].item()
                    swap_idx_b = perm[1].item()
                    self.swap_target_a = self.targets[swap_idx_a]
                    self.swap_target_b = self.targets[swap_idx_b]
                    self.swap_target_other = [
                        target
                        for idx, target in enumerate(self.targets)
                        if idx not in (swap_idx_a, swap_idx_b)
                    ]
                    logger.debug(
                        f"Swap targets selected: target_{swap_idx_a} <-> target_{swap_idx_b}"
                    )

            if self._sampling["decision"]["additional_place"][self.difficulty] == True:
                self.pre_flag = torch.rand(1, generator=generator).item() < 0.5
                self.post_flag = torch.rand(1, generator=generator).item() < 0.5
            else:
                self.pre_flag = 0
                self.post_flag = 0
            self.task_flag = torch.rand(1, generator=generator).item() < 0.5

            if self.task_flag == 1:
                self.target_target = self.target_0
                self.target_target_language = "before"
            else:
                self.target_target = self.target_1
                self.target_target_language = "after"

            self.targets_not_true = [
                t for i, t in enumerate(self.targets) if self.targets[i] != self.target_target
            ]

            tasks = []
            if self.pre_flag == True:
                tasks.append(
                    {
                        "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                        "name": f"pick up the cube",
                        "subgoal_segment": f"pick up the cube at <>",
                        "choice_label": "pick up the cube",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_pickup(
                            env, planner, obj=self.target_cube
                        ),
                        "segment": self.target_cube,
                    }
                )
                tasks.append(
                    {
                        "func": (
                            lambda: is_obj_dropped_onto(
                                self, obj=self.target_cube, target=self.target_2
                            )
                        ),
                        "name": "drop the cube onto target",
                        "subgoal_segment": f"drop the cube onto target at <>",
                        "choice_label": "drop onto",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_putonto_whenhold(
                            env, planner, target=self.target_2
                        ),
                        "segment": self.target_2,
                    }
                )

            tasks.append(
                {
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                    "name": f"pick up the cube",
                    "subgoal_segment": f"pick up the cube at <>",
                    "choice_label": "pick up the cube",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_pickup(
                        env, planner, obj=self.target_cube
                    ),
                    "segment": self.target_cube,
                }
            )
            tasks.append(
                {
                    "func": (
                        lambda: is_obj_dropped_onto(
                            self, obj=self.target_cube, target=self.target_0
                        )
                    ),
                    "name": "drop the cube onto target",
                    "subgoal_segment": f"drop the cube onto target at <>",
                    "choice_label": "drop onto",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_putonto_whenhold(
                        env, planner, target=self.target_0
                    ),
                    "segment": self.target_0,
                }
            )

            tasks.append(
                {
                    "func": (lambda: is_button_pressed(self, obj=self.button)),
                    "name": "press the button",
                    "subgoal_segment": f"press the button at <>",
                    "choice_label": "press the button",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_button(env, planner, self.button),
                    "segment": self.cap_link,
                }
            )

            if self.post_flag == True:
                tasks.append(
                    {
                        "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                        "name": f"pick up the cube",
                        "subgoal_segment": f"pick up the cube at <>",
                        "choice_label": "pick up the cube",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_pickup(
                            env, planner, obj=self.target_cube
                        ),
                        "segment": self.target_cube,
                    }
                )
                tasks.append(
                    {
                        "func": (
                            lambda: is_obj_dropped_onto(
                                self, obj=self.target_cube, target=self.target_3
                            )
                        ),
                        "name": "drop the cube onto target",
                        "subgoal_segment": f"drop the cube onto target at <>",
                        "choice_label": "drop onto",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_putonto_whenhold(
                            env, planner, target=self.target_3
                        ),
                        "segment": self.target_3,
                    }
                )

            tasks.append(
                {
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                    "name": f"pick up the cube",
                    "subgoal_segment": f"pick up the cube at <>",
                    "choice_label": "pick up the cube",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_pickup(
                        env, planner, obj=self.target_cube
                    ),
                    "segment": self.target_cube,
                }
            )
            tasks.append(
                {
                    "func": (
                        lambda: is_obj_dropped_onto(
                            self, obj=self.target_cube, target=self.target_1
                        )
                    ),
                    "name": "drop the cube onto target",
                    "subgoal_segment": f"drop the cube onto target at <>",
                    "choice_label": "drop onto",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_putonto_whenhold(
                        env, planner, target=self.target_1
                    ),
                    "segment": self.target_1,
                }
            )

            tasks.append(
                {
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                    "name": f"pick up the cube",
                    "subgoal_segment": f"pick up the cube at <>",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_pickup(
                        env, planner, obj=self.target_cube
                    ),
                    "segment": self.target_cube,
                }
            )
            tasks.append(
                {
                    "func": (
                        lambda: is_obj_dropped_onto(
                            self, obj=self.target_cube, target=self.goal_site
                        )
                    ),
                    "name": "drop the cube onto table",
                    "subgoal_segment": f"drop the cube onto table",
                    "choice_label": "drop onto",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [
                        solve_putonto_whenhold(env, planner, target=self.goal_site,height=0.01),
                    ],
                }
            )

            tasks.append(
                {
                    "func": lambda: static_check(
                        self, timestep=int(self.elapsed_steps), static_steps=20
                    ),
                    "name": "static",
                    "subgoal_segment": f"static",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [
                        solve_reset(env, planner),
                        solve_hold_obj(env, planner, static_steps=20),
                    ],
                },
            )

            tasks.append(
                {
                    "func": lambda: static_check(
                        self, timestep=int(self.elapsed_steps), static_steps=60
                    ),
                    "name": "static",
                    "subgoal_segment": f"static",
                    "specialflag": "swap",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [
                        solve_hold_obj(env, planner, static_steps=60)
                    ],
                },
            )

            tasks.append(
                {
                    "func": lambda: reset_check(self),
                    "name": "NO RECORD",
                    "subgoal_segment": f"NO RECORD",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [solve_strong_reset(env, planner)],
                },
            )

            tasks.append(
                {
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                    "name": f"pick up the cube",
                    "subgoal_segment": f"pick up the cube at <>",
                    "choice_label": "pick up the cube",
                    "demonstration": False,
                    "failure_func": lambda: is_any_obj_pickup(self, self.non_target_cubes),
                    "solve": lambda env, planner: [
                        solve_pickup(env, planner, obj=self.target_cube)
                    ],
                    "segment": self.target_cube,
                }
            )
            tasks.append(
                {
                    "func": (
                        lambda: is_obj_dropped_onto(
                            self, obj=self.target_cube, target=self.target_target
                        )
                    ),
                    "name": "place the cube onto the correct target",
                    "subgoal_segment": f"place the cube onto the correct target at <>",
                    "choice_label": "drop onto",
                    "demonstration": False,
                    "failure_func": (
                        lambda: is_obj_dropped_onto_any(
                            self, obj=self.target_cube, target=self.targets_not_true
                        )
                    ),
                    "solve": lambda env, planner: [
                        solve_putonto_whenhold(env, planner, target=self.target_target),
                    ],
                    "segment": self.target_target,
                }
            )

            self.task_list = tasks
            self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(
                self.task_list
            )
            if self.robomme_failure_recovery:
                self.fail_grasp_task_index = inject_fail_grasp(
                    self.task_list,
                    generator=generator,
                    mode=self.robomme_failure_recovery_mode,
                )
            else:
                self.fail_grasp_task_index = None

        except SceneGenerationError:
            raise
        except Exception as exc:
            raise SceneGenerationError(
                f"Failed to load VideoPlaceButton scene for seed {self.seed}"
            ) from exc

    # ------------------------------------------------------------------
    # V4 xhard (plan 2.14): demonstrate 2 cubes, each returned to its origin, everything else unchanged
    # ------------------------------------------------------------------
    def _load_scene_xhard_tail(self, generator):
        """xhard only: takes over the second half of _load_scene from "select target cube" onward (called only for xhard).

        Demonstration template (looping over objects; the original three tiers' inline sequence stays verbatim in _load_scene)::

            for each demonstration cube k: pick(k) -> drop(targets[2k])        <- before the button
            press button
            for each demonstration cube k: pick(k) -> drop(targets[2k+1])      <- after the button
            for each demonstration cube k: pick(k) -> drop(home_k)             <- return to origin
            static20 -> static60(swap) -> NO RECORD -> pick(answer cube) -> drop(target_target)

        Answer to "before / after the button" (2-object generalization of task_mapping; WARNING: not settled by the plan, may be overturned by the user):
        first draw task_flag by the original rule to decide before/after, then newly draw "which demonstration cube is asked";
        before -> the table that cube was placed on before the button (targets[2k]), after -> the table placed on after the button (targets[2k+1]).

        Random stream (N5, only shifts xhard itself): randperm[:count] replaces the original randint target selection -> swap randperm
        (as is) -> task_flag (as is) -> new answer_demo_index; landing-site actors draw no random numbers and come after all spawns.
        """
        decision_cfg = self._sampling["decision"]
        tier_key = "xhard" if self.difficulty == "xhard" else self.difficulty
        xhard_cfg = decision_cfg[tier_key]
        demo_count, return_policy = validate_demo_plan(
            xhard_cfg["demo_object_count"], xhard_cfg["demo_return_policy"],
            self.difficulty, len(self.all_cubes),
        )
        is_v6_tier = _is_newvalue_difficulty(self.difficulty)
        if not is_v6_tier and decision_cfg["additional_place"][self.difficulty] == True:
            # The original extra placements on target_2/target_3 are mutually exclusive with "two tables per cube"; xhard declares False, passing True is rejected
            raise SceneGenerationError("VideoPlaceButton xhard does not support additional_place=True")
        extra_before = int(xhard_cfg.get("extra_place_before", 0)) if is_v6_tier else 0
        extra_after = int(xhard_cfg.get("extra_place_after", 0)) if is_v6_tier else 0
        target_placements = vpb_target_placement_count(xhard_cfg) if is_v6_tier else 2 * demo_count
        expected_placements = {"xhard1": 3, "xhard2": 4, "xhard3": 5, "xhard4": 6}
        if is_v6_tier and target_placements != expected_placements[self.difficulty]:
            raise SceneGenerationError(
                f"VideoPlaceButton {self.difficulty} requires target placement count "
                f"{expected_placements[self.difficulty]}, config gives {target_placements}"
            )
        self.target_placement_count = target_placements
        if len(self.targets) < 2 * demo_count:
            raise SceneGenerationError(
                f"VideoPlaceButton xhard requires targets >= 2*demo_object_count={2 * demo_count}, actual {len(self.targets)}"
            )

        demo_ids = self._spec.value(
            "objects.demo_ids",
            torch.randperm(len(self.all_cubes), generator=generator)[:demo_count].tolist(),
            decision_key=f"{tier_key}.demo_object_count",
        )
        demo_ids = [int(i) for i in demo_ids]
        if len(demo_ids) != demo_count:
            raise SceneGenerationError(f"demonstration cubes requested {demo_count}, actual {len(demo_ids)} cubes")
        self.demo_cubes = [self.all_cubes[i] for i in demo_ids]

        self.swap_target_a = None
        self.swap_target_b = None
        self.swap_target_other = []
        if decision_cfg["swap"][self.difficulty] == True and len(self.targets) >= 2:
            perm = self._spec.value(
                "objects.swap_pair_ids",
                torch.randperm(len(self.targets), generator=generator).tolist(),
                decision_key=f"swap.{self.difficulty}",
            )
            swap_idx_a, swap_idx_b = int(perm[0]), int(perm[1])
            self.swap_target_a = self.targets[swap_idx_a]
            self.swap_target_b = self.targets[swap_idx_b]
            self.swap_target_other = [
                target for idx, target in enumerate(self.targets) if idx not in (swap_idx_a, swap_idx_b)
            ]
        self.pre_flag = 0
        self.post_flag = 0
        self.task_flag = bool(self._spec.value(
            "objects.task_flag", torch.rand(1, generator=generator).item() < 0.5,
        ))
        answer_index = int(self._spec.value(
            "objects.answer_demo_index",
            torch.randint(0, demo_count, (1,), generator=generator).item(),
            decision_key=f"{tier_key}.demo_object_count",
        ))

        # Each demonstration cube: before the button place on targets[2k], after the button on targets[2k+1]
        self.demo_before_targets = [self.targets[2 * k] for k in range(demo_count)]
        self.demo_after_targets = [self.targets[2 * k + 1] for k in range(demo_count)]
        self.target_cube = self.demo_cubes[answer_index]
        for color_name, group in (("red", self.red_cubes), ("blue", self.blue_cubes), ("green", self.green_cubes)):
            if self.target_cube in group:
                self.target_color_name = color_name
        self.target_target_language = "before" if self.task_flag == 1 else "after"
        self.non_target_cubes = [cube for cube in self.all_cubes if cube != self.target_cube]

        # V6 review fix F6 / N2 / Q-C (user K10 "fix f6", K14 "vpb issue must be fixed too"):
        # extra tables track per-table occupancy along the **full task sequence** (formal before -> extra before -> button -> formal after -> extra after),
        # before-side candidates additionally exclude all after tables (otherwise the extra cube stays on another cube's after table and two cubes share a table after the button);
        # before-question answer = the table where the answer cube was **last** put down before the button (the extra table if the extra before's owner is the answer cube),
        # after-question answer = the formal after table (extra after and return-to-origin come later). Empty candidates fail this episode's generation, no silent downgrade.
        # Random consumption same as before the change (one for owner + one randint per extra segment); only the candidate set and answer binding changed.
        n_targets = len(self.targets)
        before_ids = [2 * k for k in range(demo_count)]
        after_ids = [2 * k + 1 for k in range(demo_count)]
        occupancy = {i: None for i in range(n_targets)}   # table -> index of the demonstration cube currently on it
        location = {}                                       # demonstration cube index -> its current table
        for k in range(demo_count):
            occupancy[before_ids[k]] = k
            location[k] = before_ids[k]
        last_before_target = dict(enumerate(before_ids))    # the table where each demonstration cube was last placed before the button
        self.demo_extra_place_before = []
        self.demo_extra_place_after = []

        def _apply_formal_after():
            for k in range(demo_count):
                holder = occupancy[after_ids[k]]
                if holder is not None and holder != k:
                    raise SceneGenerationError(
                        f"VideoPlaceButton {self.difficulty} after-button table {after_ids[k]} is still occupied by cube {holder} (two cubes on one table)"
                    )
                occupancy[location[k]] = None
                occupancy[after_ids[k]] = k
                location[k] = after_ids[k]

        def _answer_target_idx():
            if self.task_flag == 1:
                return int(last_before_target[answer_index])
            return int(after_ids[answer_index])

        answer_target_idx = None
        if is_v6_tier:
            sides, owner_ids_drawn = _extra_place_owners(
                demo_count, extra_before, extra_after, generator
            )
            owner_ids = self._spec.value(
                "objects.extra_place_owner_ids", owner_ids_drawn,
                decision_key=f"{tier_key}.extra_place_before/after",
            )
            before_ids_drawn, after_ids_drawn = [], []
            after_phase_applied = False
            for side, owner in zip(sides, owner_ids):
                owner = int(owner)
                if not 0 <= owner < demo_count:
                    raise SceneGenerationError(f"VPB extra table owner out of range: {owner}")
                if side == "after" and not after_phase_applied:
                    answer_target_idx = _answer_target_idx()
                    _apply_formal_after()
                    after_phase_applied = True
                if side == "before":
                    excluded = set(after_ids) | {before_ids[answer_index]}
                else:
                    excluded = {answer_target_idx}
                candidates = [idx for idx in range(n_targets) if occupancy[idx] is None and idx not in excluded]
                if not candidates:
                    raise SceneGenerationError(
                        f"VideoPlaceButton {self.difficulty} {side} extra table: no free target that is neither the answer nor an after table"
                    )
                target_idx_drawn = candidates[
                    int(torch.randint(0, len(candidates), (1,), generator=generator).item())
                ]
                target_idx = int(self._spec.value(
                    f"objects.extra_place_target_ids.{side}.{len(before_ids_drawn if side == 'before' else after_ids_drawn)}",
                    target_idx_drawn,
                    decision_key=f"{tier_key}.extra_place_{side}",
                ))
                if target_idx not in candidates:
                    raise SceneGenerationError(
                        f"VPB re-injected {side} extra target={target_idx} is currently occupied, is an after table, or is the answer target"
                    )
                occupancy[location[owner]] = None
                occupancy[target_idx] = owner
                location[owner] = target_idx
                item = (self.demo_cubes[owner], self.targets[target_idx])
                if side == "before":
                    self.demo_extra_place_before.append(item)
                    before_ids_drawn.append(target_idx)
                    last_before_target[owner] = target_idx
                else:
                    self.demo_extra_place_after.append(item)
                    after_ids_drawn.append(target_idx)
            if not after_phase_applied:
                answer_target_idx = _answer_target_idx()
                _apply_formal_after()
            self._spec.record("actions.target_placement_count", target_placements)
            # F6 guard: replay the occupancy table over the whole placement sequence once; two cubes on one table or an in-place no-op fails generation
            place_sequence = [(k, before_ids[k]) for k in range(demo_count)]
            place_sequence += [(self.demo_cubes.index(c), self.targets.index(t)) for c, t in self.demo_extra_place_before]
            place_sequence += [(k, after_ids[k]) for k in range(demo_count)]
            place_sequence += [(self.demo_cubes.index(c), self.targets.index(t)) for c, t in self.demo_extra_place_after]
            validate_place_sequence(place_sequence, n_targets)
            self._spec.record("actions.place_sequence", [[int(c), int(t)] for c, t in place_sequence])
        else:
            answer_target_idx = _answer_target_idx()
        self.target_target = self.targets[answer_target_idx]
        self.targets_not_true = [t for t in self.targets if t != self.target_target]
        self._spec.record("actions.target_target_id", int(answer_target_idx))

        # Landing sites for returning to origin: after all spawns, call the target builder directly with each cube's initial pose (not spawn_random_target)
        self._build_xhard_final_sites(return_policy, generator)

        tasks = []
        for cube, target in zip(self.demo_cubes, self.demo_before_targets):
            tasks.extend(self._xhard_pick_place(cube, target))
        for cube, target in self.demo_extra_place_before:
            tasks.extend(self._xhard_pick_place(cube, target))
        tasks.append(
            {
                "func": (lambda: is_button_pressed(self, obj=self.button)),
                "name": "press the button",
                "subgoal_segment": f"press the button at <>",
                "choice_label": "press the button",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: solve_button(env, planner, self.button),
                "segment": self.cap_link,
            }
        )
        for cube, target in zip(self.demo_cubes, self.demo_after_targets):
            tasks.extend(self._xhard_pick_place(cube, target))
        for cube, target in self.demo_extra_place_after:
            tasks.extend(self._xhard_pick_place(cube, target))
        for cube, site, kind in self._xhard_final_sites:
            tasks.extend(self._xhard_pick_place(cube, site, home=kind))
        tasks.extend(self._xhard_closing_tasks())

        self.task_list = tasks
        self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
        if self.robomme_failure_recovery:
            self.fail_grasp_task_index = inject_fail_grasp(
                self.task_list,
                generator=generator,
                mode=self.robomme_failure_recovery_mode,
            )
        else:
            self.fail_grasp_task_index = None

    def _build_xhard_final_sites(self, return_policy, generator):
        """Final landing sites of demonstration cubes (V6 plan 2.10): build home sites for return-to-origin, goal_site region sites otherwise.

        ``return_to_origin`` follows the same path as V5 xhard bit for bit (the same ``build_home_sites`` call and record);
        other policies are reached only with new-value configs; neither kind of site draws random numbers. Results are stored in demonstration order as
        ``self._xhard_final_sites = [(cube, site, "home" | "goal"), ...]``.
        """
        if return_policy == RETURN_TO_ORIGIN:
            self.xhard_home_sites, self._xhard_home_checks = build_home_sites(self, self.demo_cubes, generator)
            self._spec.record("actions.return_pose_by_object_id", home_pose_record(self.demo_cubes, self.xhard_home_sites))
            self.xhard_goal_drop_sites = []
            self._xhard_final_sites = [(c, h, "home") for c, h in zip(self.demo_cubes, self.xhard_home_sites)]
            return
        mask = returned_mask(return_policy, len(self.demo_cubes))
        home_cubes = [c for c, r in zip(self.demo_cubes, mask) if r]
        drop_cubes = [c for c, r in zip(self.demo_cubes, mask) if not r]
        self.xhard_home_sites, self._xhard_home_checks = build_home_sites(self, home_cubes, generator)
        self.xhard_goal_drop_sites, self._xhard_goal_drop_checks = build_goal_drop_sites(
            self, drop_cubes, self.goal_site, generator,
            obstacles=goal_drop_obstacles(self, self._spec.to_dict()["layout"]["button_xy"]),
        )
        self._spec.record("actions.return_pose_by_object_id", home_pose_record(home_cubes, self.xhard_home_sites))
        self._spec.record("actions.goal_drop_pose_by_object_id",
                          home_pose_record(drop_cubes, self.xhard_goal_drop_sites))
        homes = dict(zip([c.name for c in home_cubes], self.xhard_home_sites))
        drops = dict(zip([c.name for c in drop_cubes], self.xhard_goal_drop_sites))
        self._xhard_final_sites = [
            (c, homes[c.name], "home") if r else (c, drops[c.name], "goal")
            for c, r in zip(self.demo_cubes, mask)
        ]

    def _xhard_pick_place(self, cube, target, home=False):
        """One pick + drop pair of the demonstration segment (closures bind the current cube and site via default arguments).

        ``home``: False / "target" = place on the target table; True / "home" = return to origin; "goal" = do not return, place on the table surface
        (goal_site region; text follows the original three tiers' "drop the cube onto table").
        """
        if home == "goal":
            name = "drop the cube onto table"
            segment_text = "drop the cube onto table"
        elif home is True or home == "home":
            name = "put the cube back to its original position"
            # V6 review fix N5 (user "n5 no more coordinates"): return-to-origin sites are hidden by _hidden_objects, so coordinates can never be filled in;
            # the four new tiers' template drops ``at <>``; V5 xhard specs keep the old text.
            segment_text = (name if _is_newvalue_difficulty(self.difficulty)
                            else "put the cube back to its original position at <>")
        else:
            name = "drop the cube onto target"
            segment_text = "drop the cube onto target at <>"
        return [
            {
                "func": (lambda cube=cube: is_obj_pickup(self, obj=cube)),
                "name": f"pick up the cube",
                "subgoal_segment": f"pick up the cube at <>",
                "choice_label": "pick up the cube",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner, cube=cube: solve_pickup(env, planner, obj=cube),
                "segment": cube,
            },
            {
                "func": (lambda cube=cube, target=target: is_obj_dropped_onto(self, obj=cube, target=target)),
                "name": name,
                "subgoal_segment": segment_text,
                "choice_label": "drop onto",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner, target=target: solve_putonto_whenhold(env, planner, target=target),
                "segment": target,
            },
        ]

    def _xhard_closing_tasks(self):
        """Demonstration ending + execution segment: same structure as the original three tiers (static20 -> swap -> NO RECORD -> take the answer cube to the correct table)."""
        return [
            {
                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=20),
                "name": "static",
                "subgoal_segment": f"static",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [
                    solve_reset(env, planner),
                    solve_hold_obj(env, planner, static_steps=20),
                ],
            },
            {
                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=60),
                "name": "static",
                "subgoal_segment": f"static",
                "specialflag": "swap",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [solve_hold_obj(env, planner, static_steps=60)],
            },
            {
                "func": lambda: reset_check(self),
                "name": "NO RECORD",
                "subgoal_segment": f"NO RECORD",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [solve_strong_reset(env, planner)],
            },
            {
                "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                "name": f"pick up the cube",
                "subgoal_segment": f"pick up the cube at <>",
                "choice_label": "pick up the cube",
                "demonstration": False,
                "failure_func": lambda: is_any_obj_pickup(self, self.non_target_cubes),
                "solve": lambda env, planner: [solve_pickup(env, planner, obj=self.target_cube)],
                "segment": self.target_cube,
            },
            {
                "func": (lambda: is_obj_dropped_onto(self, obj=self.target_cube, target=self.target_target)),
                "name": "place the cube onto the correct target",
                "subgoal_segment": f"place the cube onto the correct target at <>",
                "choice_label": "drop onto",
                "demonstration": False,
                "failure_func": (
                    lambda: is_obj_dropped_onto_any(self, obj=self.target_cube, target=self.targets_not_true)
                ),
                "solve": lambda env, planner: [solve_putonto_whenhold(env, planner, target=self.target_target)],
                "segment": self.target_target,
            },
        ]


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Each initialization records its own spec, never reusing the previous result
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            pose_p=self.goal_site.pose.p.tolist()[0]
            pose_q=self.goal_site.pose.q.tolist()[0]
            pose_p[2]=-0.05
            self.goal_site.set_pose(sapien.Pose(p=pose_p,q=pose_q))  
            #print(self.goal_site.pose.p)  

    def _get_obs_extra(self, info: Dict):
        return dict()


 
    def evaluate(self,solve_complete_eval=False):
        self.successflag=torch.tensor([False])
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


#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):



        # highlight_obj(self,self.target_cube, start_step=0, end_step=100, cur_step=timestep)
        
        if self.current_task_specialflag=="swap":
            if self.onto_goalsite==False:
                self.onto_goalsite=True
                self.start_step=int(self.elapsed_steps.item())
                self.end_step=int(self.elapsed_steps.item())+50



        if self.swap_target_a is not None and self.swap_target_b is not None:
            other_bins = self.swap_target_other if self.swap_target_other else None
            swap_flat_two_lane(
                self,
                cube_a=self.swap_target_a,
                cube_b=self.swap_target_b,
                start_step=self.start_step,
                end_step=self.end_step,
                cur_step=self.elapsed_steps,
                lane_offset=0.1,
                smooth=True,
                keep_upright=True,
                other_cube=other_bins,
            )
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
