
from pathlib import Path
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

from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

from .utils.SceneGenerationError import SceneGenerationError
from .utils import *
# V4 (user 2026-09-23 "fix xhard"): the `from .utils import *` line above lets the same-named submodule
# `utils.SceneGenerationError` shadow the name `SceneGenerationError`, so layout failures of the original three tiers become TypeError
# (original three tiers kept as is per H2). The xhard branch uses the alias below to get the real exception class.
from .utils.SceneGenerationError import SceneGenerationError as _RealSceneGenerationError
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


# ── Original values of the decision/native blocks (newtaskRelease-v3 step 3, mapping in plan section 2.12) ────────
NATIVE_SAMPLING = {
    "parameters": {
        "color_pool": [
            {"rgba": [1, 0, 0, 1], "name": "red"},
            {"rgba": [0, 0, 1, 1], "name": "blue"},
            {"rgba": [0, 1, 0, 1], "name": "green"},
        ],
        "color_order": {"sampler": "torch.randperm(3)"},
        "target_selection": {"sampler": "torch.randint(0, len(all_cubes))"},
        "visit_selection": {
            "count_sampler": "torch.randint(2, len(targets) + 1)",
            "order_sampler": "torch.randperm(len(targets))[:count]",
        },
        "answer_selection": {"sampler": "torch.randint(1, len(subset) + 1)",
                              "mapping": "target_target = visit_ids[which_in_subset - 1]"},
        "button_insertion": {"sampler": "torch.randint(0, len(subset))",
                              "mapping": "button_task_index = k * 2 + 2"},
        "swap_selection": {"sampler": "torch.randperm(len(targets))[:2]"},
        "swap_duration_steps": 50,
        "goal_site_z_override": -0.05,
        "recovery": "keep the entry-provided fail recover mode and the original generator",
        "target_slots": 4,
    },
    "positions": {
        "goal": {"region_center": [-0.1, 0], "region_half_size": 0.1,
                  "radius_factor": 5, "thickness": 0.005},
        "button": {"center_xy": [0.1, 0], "scale": 1.5, "randomize_range": [0.05, 0.3]},
        "cubes": {"region_center": [0, 0], "region_half_size": 0.2},
        "targets": {"region_center": [0, 0], "region_half_size": 0.2,
                     "radius_factor": 2, "thickness": 0.005},
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


# V4 xhard demonstration decision (plan 2.15): the two original keys demo_object_count=1 / native_random_goal_site stay unchanged,
# xhard values live under the subkey named xhard (after the guard strips xhard, the part visible to the original three tiers is verbatim identical to the original).
XHARD_DEMO_DECISION = {"demo_object_count": 2, "demo_return_policy": "return_to_origin"}
V6_DEMO_DECISIONS = {
    "xhard1": {"demo_object_count": 2, "demo_return_policy": "return_to_origin", "visit_counts": [2, 3]},
    "xhard2": {"demo_object_count": 2, "demo_return_policy": "return_to_origin", "visit_counts": [3, 3]},
    "xhard3": {"demo_object_count": 2, "demo_return_policy": "return_to_origin", "visit_counts": [3, 4]},
    "xhard4": {"demo_object_count": 2, "demo_return_policy": "return_to_origin", "visit_counts": [4, 4]},
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
                values["xhard"] = cfg[field]
        return values
    values = {}
    for tier in (*_NATIVE_TIERS, "xhard1", "xhard2", "xhard3", "xhard4"):
        source = getattr(cls, f"config_{tier}", None)
        if source is None and tier.startswith("xhard"):
            source = getattr(cls, "config_xhard4", None) or getattr(cls, "config_xhard", None)
        if source is not None:
            values[tier] = source[field]
    return values


def vpo_target_placement_count(tier_cfg):
    """The VPO gradient counts only placements onto a target; return_to_origin is a shared ending segment and does not count."""
    return sum(int(value) for value in tier_cfg["visit_counts"])


def _native_decision(cls, *, release="newtask-v6"):
    """Slice the decision block per plan section 2.12 (equals the original in the original-value stage)."""
    decision = {
        # How many cubes are manipulated in the demonstration video: original value 1; section 2's 2 is not enabled this round.
        "demo_object_count": 1,
        # Where each cube goes after its demonstration: original value = placed on a random goal_site at the end of the demonstration.
        "demo_return_policy": "native_random_goal_site",
        "targets": _tier_config_values(cls, "targets", release),
        "swap": _tier_config_values(cls, "swap", release),
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


@register_env("VideoPlaceOrder", override=True)
class VideoPlaceOrder(BaseEnv):

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
        #"place":2,
        "swap":False,
        "targets":4
    }
    config_medium= {
        'color': 3, 
       # "place":2,
        "swap":False,
        "targets":4
    }
    config_hard = {
        'color': 3, 
       # "place":4,
        "swap":True,
        "targets":4
    }


    # V4 xhard (derived from hard, plan 2.15): color 3, targets 4, swap True all unchanged;
    # what changes is "how many cubes are demonstrated and where they go afterward", see decision's xhard subkey (XHARD_DEMO_DECISION).
    config_xhard = {
        'color': 3,
        "swap":True,
        "targets":4
    }
    config_xhard4 = copy.deepcopy(config_xhard)
    config_xhard1 = copy.deepcopy(config_xhard4)
    config_xhard2 = copy.deepcopy(config_xhard4)
    config_xhard3 = copy.deepcopy(config_xhard4)

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
        self._spec = SpecRecorder(native_episode_spec, "VideoPlaceOrder", {"seed": seed},
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
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

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
            # keep the difficulty selected by the seed instead of forcing it to easy

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
        # V4 H2: xhard uses the real exception class (failures count as retryable task failures); original three tiers still use the shadowed name, behavior verbatim unchanged
        is_newvalue = _is_newvalue_difficulty(self.difficulty)
        _SceneGenError = _RealSceneGenerationError if self.difficulty == "xhard" or is_newvalue else SceneGenerationError

        try:
            self.table_scene = TableSceneBuilder(
                self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            self.table_scene.build()

            yaw = 0
            rotate = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])  # Quaternion for z-axis rotation
            angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))  # (3,)
            rotate = matrix_to_quaternion(
                euler_angles_to_matrix(angles, convention="XYZ")
            )
            try:
                self.goal_site = spawn_random_target(
                    self,
                    avoid=None,  # Use current avoidance list, containing all spawned cubes
                    include_existing=False,  # Manually maintain list
                    include_goal=False,  # Manually maintain list
                    region_center=list(self._sampling["positions"]["goal"]["region_center"]),
                    region_half_size=self._sampling["positions"]["goal"]["region_half_size"],
                    radius=self.cube_half_size * self._sampling["positions"]["goal"]["radius_factor"],  # Use radius instead of half_size
                    thickness=self._sampling["positions"]["goal"]["thickness"],  # target thickness
                    min_gap=self.cube_half_size * 1,  # Gap requirement same as cube
                    name_prefix=f"goal_site",
                    generator=self.generator,
                    recorder=self._spec,
                    spec_path="layout.goal_xy",
                )
            except RuntimeError as exc:
                raise _SceneGenError("goal_site sampling failed") from exc
            avoid = []
            avoid.append(self.goal_site)
            button_cfg = self._sampling["positions"]["button"]
            cubes_cfg = self._sampling["positions"]["cubes"]
            targets_cfg = self._sampling["positions"]["targets"]
            decision_cfg = self._sampling["decision"]
            button_obb = build_button(
                self,
                center_xy=tuple(button_cfg["center_xy"]),
                scale=button_cfg["scale"],
                generator=self.generator,
                randomize_range=tuple(button_cfg["randomize_range"]),
                recorder=self._spec,
                spec_path="layout.button_xy",
            )
            avoid.append(button_obb)

            self.all_cubes = []  # Save all cube objects

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
                "objects.color_order",
                torch.randperm(len(color_groups), generator=self.generator).tolist(),
            )
            color_groups = [color_groups[i] for i in shuffle_indices]

            self.target_color_name = color_groups[0]["name"]
            logger.debug(f"Target color selected: {self.target_color_name}")

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
                                random_yaw=True,
                                name_prefix=f"cube_{group['name']}_{cube_idx}",
                                generator=self.generator,
                                # Only V4 xhard records cube poses into the spec (the recorder only freezes accepted values, no extra random draws)
                                **self._xhard_spec_kwargs(f"layout.cubes.{group['name']}_{cube_idx}"),
                            )
                        except RuntimeError as exc:
                            raise _SceneGenError(
                                f"Failed to generate {group['name']} cube {cube_idx}: {exc}"
                            ) from exc

                        self.all_cubes.append(cube)
                        group["list"].append(cube)
                        cube_name = f"cube_{group['name']}_{cube_idx}"
                        group["name_list"].append(cube_name)
                        setattr(self, cube_name, cube)
                        if self.difficulty == "xhard" or is_newvalue:
                            # V5 (L2 b, plan 2.16): placed cubes enter avoid as exact 2D obstacles; later cubes and target tables
                            # (both kinds of spawn calls) avoid accordingly. The actor path via _trimesh_box_to_obb2d degenerates into a segment ~2/3 of the time,
                            # voiding min_gap along its normal; take initial_pose (does not depend on simulation being initialized), no random draws.
                            # Original three tiers still pass the actor; the spawn calls themselves are verbatim unchanged.
                            avoid.append(cube_obb2d_exact(cube.initial_pose, self.cube_half_size))
                        else:
                            avoid.append(cube)

                logger.debug(f"Generated {len(group['list'])} {group['name']} cubes")

            logger.debug(f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})")

            self.targets = []
            for i in range(4):
                if i < decision_cfg["targets"][self.difficulty]:
                    try:
                        target = spawn_random_target(
                            self,
                            avoid=avoid,  # Use current avoidance list, containing all spawned cubes
                            include_existing=False,  # Manually maintain list
                            include_goal=False,  # Manually maintain list
                            region_center=list(targets_cfg["region_center"]),
                            region_half_size=targets_cfg["region_half_size"],
                            radius=self.cube_half_size*targets_cfg["radius_factor"],  # Use radius instead of half_size
                            thickness=targets_cfg["thickness"],  # target thickness
                            min_gap=self.cube_half_size*1,  # Gap requirement same as cube
                            name_prefix=f"target_{i}",
                            generator=self.generator,
                            **self._xhard_spec_kwargs(f"layout.targets.{i}"),
                        )
                    except RuntimeError as exc:
                        raise _SceneGenError(f"Target {i + 1} sampling failed: {exc}") from exc

                    self.targets.append(target)
                    setattr(self, f"target_{i}", target)
                    avoid.append(target)

            if self.difficulty == "xhard" or is_newvalue:
                # V6 new tiers and V5 old specs share the demonstration path; the code below for the original three tiers is not changed by a single line
                self._load_scene_xhard_tail()
                return
            # Original three tiers: demonstrate 1 cube, place on a random goal_site (the guard already ensures original values); read once here as the real consumption point, no random draws
            validate_demo_plan(decision_cfg["demo_object_count"], decision_cfg["demo_return_policy"],
                               self.difficulty, len(self.all_cubes))

            if len(self.all_cubes) > 0:
                target_cube_idx = self._spec.value(
                    "objects.target_cube_idx",
                    torch.randint(0, len(self.all_cubes), (1,), generator=self.generator).item(),
                )
                self.target_cube = self.all_cubes[target_cube_idx]

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

            self.non_target_cubes = [cube for cube in self.all_cubes if cube != self.target_cube]
            logger.debug(f"Non-target cubes: {len(self.non_target_cubes)}")

            self.swap_target_a = None
            self.swap_target_b = None
            self.swap_target_other = []

            if decision_cfg["swap"][self.difficulty]==True:
                if len(self.targets) >= 2:
                    perm = self._spec.value(
                        "objects.swap_pair_ids",
                        torch.randperm(len(self.targets), generator=self.generator).tolist(),
                    )
                    # perm is now an integer list from the spec (originally a tensor); value semantics unchanged
                    swap_idx_a = int(perm[0])
                    swap_idx_b = int(perm[1])
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
            num_targets_to_pick = self._spec.value(
                "objects.num_targets_to_pick",
                torch.randint(2, len(self.targets) + 1, (1,), generator=self.generator).item(),
            )

            indices = self._spec.value(
                "objects.visit_ids",
                torch.randperm(len(self.targets), generator=self.generator)[:num_targets_to_pick].tolist(),
            )

            self.which_targets_to_pick = [self.targets[i] for i in indices]

            self.which_in_subset=self._spec.value(
                "objects.which_in_subset",
                torch.randint(1,len(self.which_targets_to_pick)+1,(1,),generator=self.generator).item(),
            )

            logger.debug("self.which_in_subset: %s", self.which_in_subset)
            self.target_target=self.which_targets_to_pick[self.which_in_subset-1]

            self.targets_not_true = [t for i, t in enumerate(self.targets) if self.targets[i]!=self.target_target]

            if len(self.which_targets_to_pick) > 0:
                k = self._spec.value(
                    "objects.button_after_pair_index",
                    torch.randint(0, len(self.which_targets_to_pick), (1,), generator=self.generator).item(),
                )
                self.button_task_index = k * 2 + 2  # each pair contributes pickup + drop
            else:
                self.button_task_index = 0

        except _SceneGenError:
            raise
        except Exception as exc:
            raise _SceneGenError(
                f"Failed to load VideoPlaceOrder scene for seed {self.seed}"
            ) from exc

    # ------------------------------------------------------------------
    # V4 xhard (plan 2.15): demonstrate 2 cubes, each returned to its origin, everything else unchanged
    # ------------------------------------------------------------------
    def _xhard_spec_kwargs(self, spec_path):
        """Only xhard passes a recorder to spawn_*; original three tiers return an empty dict (call form verbatim equivalent to the original)."""
        if self.difficulty != "xhard" and not _is_newvalue_difficulty(self.difficulty):
            return {}
        return {"recorder": self._spec, "spec_path": spec_path}

    @staticmethod
    def xhard_button_task_index(visit_counts, button_after_visit):
        """Index at which the button is inserted into the demonstration sequence (2-object generalization, replacing the original ``k*2+2``).

        The xhard demonstration sequence = per object in turn ``[that object's visits pick+drop ..., return-to-origin pick+drop]``,
        each unit being 2 tasks. The button is inserted after the "global visit placement number ``button_after_visit + 1``",
        i.e. between two units, never between a pick and its drop:

            index = 2 * (button_after_visit + 1 + number of completed "return to origin" units before it)

        When an object's last visit is exactly the insertion point, the button goes **before** its return to origin.
        With ``visit_counts=[n]`` this degenerates to the original formula ``k*2+2``.
        """
        placed = int(button_after_visit) + 1
        if not 1 <= placed <= sum(visit_counts):
            raise ValueError(f"button_after_visit={button_after_visit} exceeds total visits {sum(visit_counts)}")
        homes_before = 0
        cumulative = 0
        for count in visit_counts:
            cumulative += int(count)
            if cumulative < placed:
                homes_before += 1
        return 2 * (placed + homes_before)

    def _load_scene_xhard_tail(self):
        """xhard only: takes over the second half of _load_scene from "select target cube" onward (called only for xhard).

        Each demonstration cube goes through the original rule of "visit several target tables" and is returned to origin right after, before the next cube
        (sequential execution ensures tables are not occupied by the previous cube). Answer (2-object generalization of task_mapping; WARNING: not settled by the plan,
        may be overturned by the user): newly draw "which demonstration cube is asked"; which_in_subset is drawn by the original rule within that cube's own visit sequence,
        and the instruction text is still "put the {that cube's color} cube on the table it was placed on the n-th time".

        Random stream (N5, only shifts xhard itself): randperm[:count] replaces the original randint target selection -> swap randperm (as is)
        -> per object [visit count randint, visit order randperm] -> answer_demo_index -> which_in_subset -> button insertion point;
        landing-site actors draw no random numbers and come after all spawns.
        """
        decision_cfg = self._sampling["decision"]
        tier_key = "xhard" if self.difficulty == "xhard" else self.difficulty
        xhard_cfg = decision_cfg[tier_key]
        is_v6_tier = _is_newvalue_difficulty(self.difficulty)
        demo_count, return_policy = validate_demo_plan(
            xhard_cfg["demo_object_count"], xhard_cfg["demo_return_policy"],
            self.difficulty, len(self.all_cubes),
        )
        if is_v6_tier:
            configured_counts = [int(value) for value in xhard_cfg["visit_counts"]]
            expected_sums = {"xhard1": 5, "xhard2": 6, "xhard3": 7, "xhard4": 8}
            if (len(configured_counts) != demo_count or any(value < 2 or value > len(self.targets)
                    for value in configured_counts) or sum(configured_counts) != expected_sums[self.difficulty]):
                raise _RealSceneGenerationError(
                    f"VideoPlaceOrder {self.difficulty} visit_counts={configured_counts} is invalid; "
                    f"requires {demo_count} counts summing to {expected_sums[self.difficulty]}"
                )
            count_order = (torch.randperm(demo_count, generator=self.generator).tolist()
                           if len(set(configured_counts)) > 1 else list(range(demo_count)))
            counts_drawn = [configured_counts[index] for index in count_order]
            visit_counts = [int(value) for value in self._spec.value(
                "objects.visit_counts_by_object", counts_drawn,
                decision_key=f"{tier_key}.visit_counts",
            )]
            if sorted(visit_counts) != sorted(configured_counts):
                raise _RealSceneGenerationError(
                    f"VideoPlaceOrder re-injected visit_counts={visit_counts} and the configured multiset {configured_counts} do not match"
                )
        else:
            # V5 snapshots have no visit_count_range field; the default range keeps the original randint(2, len(targets)+1) bit for bit.
            visit_lo, visit_hi = (int(v) for v in xhard_cfg.get("visit_count_range", [2, len(self.targets)]))
            if visit_lo != 2 or not visit_lo <= visit_hi <= len(self.targets):
                raise _RealSceneGenerationError(
                    f"VideoPlaceOrder xhard visit_count_range={[visit_lo, visit_hi]} is invalid (lower bound must be 2, upper bound <= {len(self.targets)})"
                )
            visit_counts = []
        if len(self.targets) < 2:
            raise _RealSceneGenerationError(f"VideoPlaceOrder xhard requires at least 2 target tables, actual {len(self.targets)}")

        demo_ids = self._spec.value(
            "objects.demo_ids",
            torch.randperm(len(self.all_cubes), generator=self.generator)[:demo_count].tolist(),
            decision_key=f"{tier_key}.demo_object_count",
        )
        demo_ids = [int(i) for i in demo_ids]
        if len(demo_ids) != demo_count:
            raise _RealSceneGenerationError(f"demonstration cubes requested {demo_count}, actual {len(demo_ids)} cubes")
        self.demo_cubes = [self.all_cubes[i] for i in demo_ids]

        self.swap_target_a = None
        self.swap_target_b = None
        self.swap_target_other = []
        if decision_cfg["swap"][self.difficulty] == True:
            perm = self._spec.value(
                "objects.swap_pair_ids",
                torch.randperm(len(self.targets), generator=self.generator).tolist(),
                decision_key=f"swap.{self.difficulty}",
            )
            swap_idx_a, swap_idx_b = int(perm[0]), int(perm[1])
            self.swap_target_a = self.targets[swap_idx_a]
            self.swap_target_b = self.targets[swap_idx_b]
            self.swap_target_other = [
                target for idx, target in enumerate(self.targets) if idx not in (swap_idx_a, swap_idx_b)
            ]

        # Each demonstration cube draws its own visit count and visit order by the original rule
        self.demo_visit_targets = []
        for k in range(demo_count):
            if is_v6_tier:
                count = visit_counts[k]
            else:
                count = int(self._spec.value(
                    f"objects.num_targets_by_object.{k}",
                    torch.randint(visit_lo, visit_hi + 1, (1,), generator=self.generator).item(),
                    decision_key="xhard.demo_object_count",
                ))
            ids = self._spec.value(
                f"objects.visit_ids_by_object.{k}",
                torch.randperm(len(self.targets), generator=self.generator)[:count].tolist(),
                decision_key=f"{tier_key}.visit_counts" if is_v6_tier else "xhard.demo_object_count",
            )
            self.demo_visit_targets.append([self.targets[int(i)] for i in ids])
            if not is_v6_tier:
                visit_counts.append(len(ids))

        answer_index = int(self._spec.value(
            "objects.answer_demo_index",
            torch.randint(0, demo_count, (1,), generator=self.generator).item(),
            decision_key=f"{tier_key}.demo_object_count",
        ))
        self.target_cube = self.demo_cubes[answer_index]
        for color_name, group in (("red", self.red_cubes), ("blue", self.blue_cubes), ("green", self.green_cubes)):
            if self.target_cube in group:
                self.target_color_name = color_name
        self.non_target_cubes = [cube for cube in self.all_cubes if cube != self.target_cube]
        # Attributes with the same names as in the original three tiers point at the answer cube's visit sequence, so downstream (task_goal etc.) can read them as before
        self.which_targets_to_pick = self.demo_visit_targets[answer_index]
        self.which_in_subset = int(self._spec.value(
            "objects.which_in_subset",
            torch.randint(1, len(self.which_targets_to_pick) + 1, (1,), generator=self.generator).item(),
            decision_key=f"{tier_key}.visit_counts" if is_v6_tier else "xhard.demo_object_count",
        ))
        self.target_target = self.which_targets_to_pick[self.which_in_subset - 1]
        self.targets_not_true = [t for t in self.targets if t != self.target_target]
        self._spec.record("actions.target_target_id", self.targets.index(self.target_target))

        button_after_visit = int(self._spec.value(
            "objects.button_after_visit_index",
            torch.randint(0, sum(visit_counts), (1,), generator=self.generator).item(),
            decision_key=f"{tier_key}.visit_counts" if is_v6_tier else "xhard.demo_object_count",
        ))
        self.button_task_index = self.xhard_button_task_index(visit_counts, button_after_visit)
        self._spec.record("actions.button_task_index", self.button_task_index)
        if is_v6_tier:
            self.target_placement_count = sum(visit_counts)
            self._spec.record("actions.target_placement_count", sum(visit_counts))

        # Landing sites for returning to origin: after all spawns, call the target builder directly with each cube's initial pose (not spawn_random_target)
        self._build_xhard_final_sites(return_policy)

    def _build_xhard_final_sites(self, return_policy):
        """Final landing sites of demonstration cubes (V6 plan 2.10): build home sites for return-to-origin, goal_site region sites otherwise.

        ``return_to_origin`` follows the same path as V5 xhard bit for bit; other policies are reached only with new-value configs; neither kind of site draws random numbers.
        Cubes not returned must leave the target tables: VPO demonstrates objects sequentially, and the next cube may visit a table where the previous cube is resting.
        """
        if return_policy == RETURN_TO_ORIGIN:
            self.xhard_home_sites, self._xhard_home_checks = build_home_sites(self, self.demo_cubes, self.generator)
            self._spec.record("actions.return_pose_by_object_id", home_pose_record(self.demo_cubes, self.xhard_home_sites))
            self.xhard_goal_drop_sites = []
            self._xhard_final_sites = [(c, h, "home") for c, h in zip(self.demo_cubes, self.xhard_home_sites)]
            return
        mask = returned_mask(return_policy, len(self.demo_cubes))
        home_cubes = [c for c, r in zip(self.demo_cubes, mask) if r]
        drop_cubes = [c for c, r in zip(self.demo_cubes, mask) if not r]
        self.xhard_home_sites, self._xhard_home_checks = build_home_sites(self, home_cubes, self.generator)
        self.xhard_goal_drop_sites, self._xhard_goal_drop_checks = build_goal_drop_sites(
            self, drop_cubes, self.goal_site, self.generator,
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

    def _build_xhard_task_list(self):
        """xhard task table (rebuilt on every _initialize_episode, at the same time as the original three tiers)."""
        pair_tasks = []
        for (cube, site, kind), visits in zip(self._xhard_final_sites, self.demo_visit_targets):
            for target in visits:
                pair_tasks.extend(self._xhard_pick_place(cube, target))
            pair_tasks.extend(self._xhard_pick_place(cube, site, home=kind))

        button_task = {
            "func": (lambda: is_button_pressed(self, obj=self.button)),
            "name": "press the button",
            "subgoal_segment": f"press the button at <>",
            "choice_label": "press the button",
            "demonstration": True,
            "failure_func": None,
            "solve": lambda env, planner: solve_button(env, planner, self.button),
            "segment": self.cap_link,
        }
        tasks = pair_tasks[: self.button_task_index] + [button_task] + pair_tasks[self.button_task_index:]
        tasks.extend([
            {
                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=20),
                "name": "static",
                "subgoal_segment": f"static",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [solve_reset(env, planner), solve_hold_obj(env, planner, static_steps=20)],
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
        ])

        self.task_list = tasks
        self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
        if self.robomme_failure_recovery:
            self.fail_grasp_task_index = inject_fail_grasp(
                self.task_list,
                generator=self.generator,
                mode=self.robomme_failure_recovery_mode,
            )
        else:
            self.fail_grasp_task_index = None

    def _xhard_pick_place(self, cube, target, home=False):
        """One pick + drop pair of the demonstration segment (closures bind the current cube and site via default arguments).

        ``home``: False = place on the target table; True / "home" = return to origin; "goal" = do not return, place on the table surface
        (goal_site region; text follows the original three tiers' "drop the cube onto table").
        """
        if home == "goal":
            name = "drop the cube onto table"
            segment_text = "drop the cube onto table"
        elif home is True or home == "home":
            name = "put the cube back to its original position"
            # V6 review fix N5 (user "n5 no more coordinates"): return-to-origin sites are hidden and coordinates can never be filled in, so the four new tiers' template drops ``at <>``; V5 xhard keeps the old text
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

            if self.difficulty == "xhard" or _is_newvalue_difficulty(self.difficulty):
                # V6 new-value tiers and V5 old specs share the demonstration template; the inline sequence below for the original three tiers is not changed by a single line
                self._build_xhard_task_list()
                return


            tasks = []

            # 1) Generate all "pick + place" combinations into temporary list first
            pair_tasks = []
            for i in self.which_targets_to_pick:
                # 1.1 Pick up
                pair_tasks.append({
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                "name": f"pick up the cube",
                "subgoal_segment":f"pick up the cube at <>",
                "choice_label": "pick up the cube",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
                    "segment":self.target_cube, 
                })

                # 1.2 Place (note using i=i to bind current target)
                pair_tasks.append({
                    "func": (lambda i=i: is_obj_dropped_onto(self, obj=self.target_cube, target=i)),
                    "name": "drop the cube onto target",
                    "subgoal_segment":f"drop the cube onto target at <>",
                    "choice_label": "drop onto",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner, i=i: solve_putonto_whenhold(env, planner,  target=i),
                    "segment":i,
                })

            # 2) Define "press button" task
            button_task = {
                "func": (lambda: is_button_pressed(self, obj=self.button)),
                "name": "press the button",
                "subgoal_segment":f"press the button at <>",
                "choice_label": "press the button",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: solve_button(env, planner, self.button),
                "segment":self.cap_link 
            }

            # 4) Assemble final tasks
            tasks = pair_tasks[: self.button_task_index] + [button_task] + pair_tasks[self.button_task_index :]
    ############
            tasks.append({
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                        "name": f"pick up the cube",
                        "subgoal_segment":f"pick up the cube at <>",
                        "choice_label": "pick up the cube",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
                    "segment":self.target_cube,
                })
            tasks.append({
                    "func": (lambda: is_obj_dropped_onto(self,obj=self.target_cube,target=self.goal_site)),
                "name": "drop the cube onto table",
                "subgoal_segment":f"drop the cube onto table",
                "choice_label": "drop onto",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [solve_putonto_whenhold(env, planner,target=self.goal_site,height=0.01)],
             
            })
            tasks.append(       {
                                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=20),
                                "name": "static",
                                "subgoal_segment":f"static",

                                "demonstration": True,
                                "failure_func": None,

                                "solve": lambda env, planner: [solve_reset(env,planner), solve_hold_obj(env, planner, static_steps=20)],
                                },)

            tasks.append(       {
                                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=60),
                                "name": "static",
                                "subgoal_segment":f"static",
                                "specialflag":"swap",
                                "demonstration": True,
                                "failure_func": None,

                                "solve": lambda env, planner: [solve_hold_obj(env, planner, static_steps=60)],
                                },)

            tasks.append({
                                "func": lambda:reset_check(self),
                                "name": "NO RECORD",
                                "subgoal_segment":f"NO RECORD",
                                "demonstration": True,
                                "failure_func": None,
                                "solve": lambda env, planner: [ solve_strong_reset(env,planner)],
                                },)


            tasks.append({
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                    "name": f"pick up the cube",
                    "subgoal_segment":f"pick up the cube at <>",
                    "choice_label": "pick up the cube",
                    "demonstration": False,
                    "failure_func":lambda: is_any_obj_pickup(self, self.non_target_cubes),
                    "solve": lambda env, planner: [solve_pickup(env, planner, obj=self.target_cube)],
                    "segment":self.target_cube,
                })
            tasks.append({
                    "func": (lambda: is_obj_dropped_onto(self,obj=self.target_cube,target=self.target_target)),
                    "name": "place the cube onto the correct target",
                    "subgoal_segment":f"place the cube onto the correct target at <>",
                    "choice_label": "drop onto",
                    "demonstration": False,
                    "failure_func": (lambda: is_obj_dropped_onto_any(self,obj=self.target_cube,target=self.targets_not_true)),
                    "solve": lambda env, planner: [solve_putonto_whenhold(env, planner,target=self.target_target),
                                                ],
                    "segment":self.target_target
            
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
        all_tasks_completed, current_task_name, task_failed ,self.current_task_specialflags= sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

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
