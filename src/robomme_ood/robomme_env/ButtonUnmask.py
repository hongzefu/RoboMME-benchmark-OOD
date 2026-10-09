import copy
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from .utils.episode_spec import SpecRecorder
from .utils.sampling_config import assert_native_decision, fill_missing_newvalue, split_sampling_config
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
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils import reset_panda
from .utils.difficulty import normalize_robomme_difficulty, is_newvalue_difficulty
from .utils.SceneGenerationError import SceneGenerationError
from .utils.unmask_distractors import add_distractor_misgrasp_failure
# V5 xhard (L13 / L14): unified distractor sampler and independent parking points; called only in the xhard branch, original three tiers do not enter this module
from .utils.unmask_distractor_sampler import (
    V5_DISTRACTOR_PRESETS,
    reveal_actors_parked,
    reveal_distractor_bins_parked,
    spawn_distractor_layout,
)
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


# ── Original values of the decision/native blocks (newtaskRelease-v3 step 3, mapping in plan section 2.6) ────────
NATIVE_SAMPLING = {
    "parameters": {
        "color_pool": [
            {"rgba": [1, 0, 0, 1], "name": "red"},
            {"rgba": [0, 1, 0, 1], "name": "green"},
            {"rgba": [0, 0, 1, 1], "name": "blue"},
        ],
        "color_order": {"sampler": "torch.randperm(3)"},
        "constructor_rng": {
            "sampler": "torch.randint",
            "low": 1,
            "high_exclusive": 6,
            "note": "this constructor draw does not decide the pick count but decides later random stream position; must be kept (red line R8)",
        },
        "hidden_rule": "first 3 bins each hide one cube, the rest are empty",
        "pick_rule": "after pressing the button pick bin_0 first, then bin_1 if pick>1",
        "recovery": "the constructor's self.generator is used for recovery; the scene has its own local generator with the same seed; the two streams are separate",
        "step_bin_scan": 15,
    },
    "positions": {
        "button": {
            "center_xy": [-0.2, 0],
            "scale": 1.5,
            "randomize": True,
            "randomize_range": [0.1, 0.1],
        },
        "bins": {"min_gap_factor": 2, "max_trials": 256, "yaw_expression": "u * 90 degrees"},
        "hidden_cube": {"half_size_divisor": 1.2, "yaw": 0.0, "dynamic": True},
        "reveal_window": {"start_step": 0, "end_step": 64},
    },
}


# ── V4 xhard-specific decision entries (NEWTASK_RELEASE_V4_PLAN 2.7 / 2.9; G2, B3, B13 are user-fixed values) ──
# Original three tiers do not read these keys; the guard only checks structure and admits values for subkeys named xhard (sampling_config.assert_native_decision).
XHARD_BIN_LAYOUT = {
    # G2 (2026-09-22): region unchanged, spacing factor 2 -> 0.75, 8 bins; if not all fit, the episode fails
    "min_gap_factor": 0.75,
}
# V5 (NEWTASK_RELEASE_V5_PLAN 2.4; L6-L13): tight ring band [0.2425, 0.3289], count set by inner density, half contain a cube,
# three-color balanced rotation, 1024 trials; unified 7-key schema (count / ring_max_abs_xy / cube_count_range / color_pool /
# color_rule / min_gap_factor / max_trials), values in unmask_distractor_sampler.V5_DISTRACTOR_PRESETS.
XHARD_DISTRACTOR = copy.deepcopy(V5_DISTRACTOR_PRESETS["ButtonUnmask"])


# ── V6 (NEWTASK_RELEASE_V6_PLAN 2.3) new-value tier table: xhard1/2/3 keep all xhard mechanisms (8 inner-ring bins + tight ring band +
# three-color rotation + independent parking), only changing per tier the distractor count and cube-containing count cube_count_range; ring width, spacing,
# trial count etc. all follow xhard (with fewer distractors they are placed more sparsely in the same band; band width is not re-derived).
# The "xhard" key is the original XHARD_* constant itself, values bit-identical.
def _newvalue_distractor(count, cube_count_range):
    """Based on xhard's distractor config, replace only count and cube_count_range."""
    cfg = copy.deepcopy(XHARD_DISTRACTOR)
    cfg["count"] = count
    cfg["cube_count_range"] = list(cube_count_range)
    return cfg


# V7 fixed values (0928 plan 3.2.2): tight-band distractors 0/4/8/12, cube-containing always half; inner ring fixed at 8 => total table bins 8/12/16/20.
# v8 (1001 plan 1 table 1 / 2.1): xhard1 distractors 0 -> 4, cube-containing 0 -> 2 (pick_count still 2), same distractors as xhard2 but one fewer pick;
# xhard2-4 unchanged => distractors 4/4/8/12, total table bins 12/12/16/20.
# XHARD_DISTRACTOR (V5 preset of 14) itself unchanged; xhard4 uses 12 instead (top tier lowered).
NEWVALUE_DISTRACTOR = {
    "xhard4": _newvalue_distractor(12, [6, 6]),
    "xhard1": _newvalue_distractor(4, [2, 2]),
    "xhard2": _newvalue_distractor(4, [2, 2]),
    "xhard3": _newvalue_distractor(8, [4, 4]),
}
# Inner-ring bin placement (spacing factor 0.75) identical across the four tiers
NEWVALUE_BIN_LAYOUT = {tier: XHARD_BIN_LAYOUT for tier in NEWVALUE_DISTRACTOR}


def native_blocks(cls):
    """Original ``(decision, native)`` blocks of this env; shared by external export and internal parsing."""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


def _native_decision(cls):
    """Slice the decision block per plan section 2.6 (equals the original in the original-value stage)."""
    return {
        "pick_count": {difficulty: cfg["pick"] for difficulty, cfg in cls.configs.items()},
        "bin_layout_policy": {
            "count": {difficulty: cfg["bin"] for difficulty, cfg in cls.configs.items()},
            "region_center": [0, 0],
            "region_half_size": 0.2,
            # xhard first (same key order as V5), V6's xhard1/2/3 appended after it
            **{tier: copy.deepcopy(layout) for tier, layout in NEWVALUE_BIN_LAYOUT.items()},
        },
        # Part visible to the original three tiers stays None; V4 distractor bins live under the xhard subkey
        "distractor": None,
        **{tier: {"distractor": copy.deepcopy(dist)} for tier, dist in NEWVALUE_DISTRACTOR.items()},
    }


def _resolve_sampling_config(cls, override):
    """Split out this instance's private decision/native copies; draws no random numbers, must be called before the Generator."""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    # V6: old snapshots (V5 has no xhard1/2/3 subtrees) get them filled from source declarations
    fill_missing_newvalue(decision, decision_default)
    native["decision"] = decision
    return native


@register_env("ButtonUnmask", override=True)
class ButtonUnmask(BaseEnv):

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
    'bin':15,
    "pick":2,
    }

    config_easy = {
    'bin':3,
    "pick":1,
    }

    config_medium = {
    'bin':5,
    "pick":1,
    }

    # V4 xhard (derived from hard, plan 2.9): pick 2 -> 3; bins 15 -> 8 (G2: with min_gap_factor 0.75,
    # see XHARD_BIN_LAYOUT). There are also tight-band distractor bins (V5: XHARD_DISTRACTOR, VU 15 / BU 14), not counted in bin.
    config_xhard4 = {
    'bin':8,
    "pick":3,
    }

    # V6 new-value family: inner-ring bin count fixed at 8, pick per tier 2/3/3/3.
    config_xhard1 = {
    'bin':8,
    "pick":2,
    }

    config_xhard2 = {
    'bin':8,
    "pick":3,
    }

    config_xhard3 = {
    'bin':8,
    "pick":3,
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
        self._spec = SpecRecorder(native_episode_spec, "ButtonUnmask", {"seed": seed},
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
        normalized_robomme_difficulty = normalize_robomme_difficulty(
            kwargs.pop("difficulty", None)
        )
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

        # Use seed to randomly determine number of repetitions (1-5) arbitrarily
        generator = torch.Generator()
        generator.manual_seed(seed)
        ctor_cfg = self._sampling["parameters"]["constructor_rng"]
        # This draw does not decide the pick count but decides random stream position; recorded in sampling_trace to prove it still happens
        self.num_repeats = self._spec.value(
            "actions.sampling_trace.constructor_draw",
            torch.randint(ctor_cfg["low"], ctor_cfg["high_exclusive"], (1,), generator=generator).item(),
        )
        logger.debug(f"Task will repeat {self.num_repeats} times (pickup-drop cycles)")
        self.generator = generator  

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
        button_obb_1 = build_button(
            self,
            center_xy=tuple(button_cfg["center_xy"]),
            scale=button_cfg["scale"],
            generator=generator,
            name="button",
            randomize=button_cfg["randomize"],
            randomize_range=tuple(button_cfg["randomize_range"]),
            recorder=self._spec,
            spec_path="layout.button_xy",
        )
        # Store first button before building second one
        self.button_left = self.button
        self.button_joint_1 = self.button_joint

        avoid = [button_obb_1]


             # Generate 3 bins
        self.spawned_bins = []
        decision_cfg = self._sampling["decision"]
        bin_layout = decision_cfg["bin_layout_policy"]
        bins_cfg = self._sampling["positions"]["bins"]
        hidden_cfg = self._sampling["positions"]["hidden_cube"]
        # V6: the new-value family (xhard1/2/3/xhard) uses the xhard mechanism; values are looked up by this episode's tier
        xhard = is_newvalue_difficulty(self.difficulty)
        # V4 xhard: spacing factor from decision's xhard entry (G2 0.75); original three tiers still read native's original value, expression unchanged
        gap_factor = (bin_layout[self.difficulty]["min_gap_factor"] if xhard
                      else bins_cfg["min_gap_factor"])
        if xhard:
            requested_bins = bin_layout["count"][self.difficulty]
            # The reveal animation only scans bin_0..bin_{scan-1}: bins beyond that are never revealed (plan 2.8/2.9)
            if self._sampling["parameters"]["step_bin_scan"] < requested_bins:
                raise ValueError(
                    f"step_bin_scan={self._sampling['parameters']['step_bin_scan']} is smaller than bin count {requested_bins}"
                )
            self._spec.record("layout.bin_count.requested", requested_bins)
        for i in range(bin_layout["count"][self.difficulty]):
            try:
                bin_actor = spawn_random_bin(
                    self,
                    avoid=avoid,  # Use current avoidance list, containing all spawned objects
                    region_center=list(bin_layout["region_center"]),
                    region_half_size=bin_layout["region_half_size"],
                    min_gap=self.cube_half_size*gap_factor,  # bins need larger gap, increased to 6x to avoid collision
                    name_prefix=f"bin_{i}",
                    max_trials=bins_cfg["max_trials"],
                    generator=generator,
                    recorder=self._spec,
                    spec_path=f"layout.bins.{i}",
                )
            except RuntimeError as e:
                if xhard:
                    # 2.2-4: under xhard, failing to place all bins fails the episode; silent truncation not allowed
                    self._spec.record("layout.bin_count.placed", len(self.spawned_bins))
                    raise SceneGenerationError(
                        f"ButtonUnmask xhard bins do not fit: requested {requested_bins}, placed only {len(self.spawned_bins)} bins"
                    ) from e
                break

            self.spawned_bins.append(bin_actor)
            # Assign bin to self.bin_0, self.bin_1 etc. attributes
            setattr(self, f"bin_{i}", bin_actor)
            # Add newly generated bin to avoidance list
            avoid.append(bin_actor)
        if xhard:
            self._spec.record("layout.bin_count.placed", len(self.spawned_bins))


        # Generate 3 dynamic cubes under each bin (using fixed position, colors red, green, blue)
        spawned_dynamic_cubes = []
        cube_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # Red, Green, Blue
        color_names = ["red", "green", "blue"]

        # Use seed to randomly shuffle color order

        self._spec.identity.setdefault("difficulty", getattr(self, "difficulty", None))
        shuffle_indices = self._spec.value(
            "objects.color_order", torch.randperm(len(cube_colors), generator=generator).tolist()
        )
        cube_colors = [cube_colors[i] for i in shuffle_indices]
        color_names = [color_names[i] for i in shuffle_indices]

        # Store color_names for RecordWrapper access
        self.color_names = color_names

        # Generate cubes only for first 3 bins
        for i, bin_actor in enumerate(self.spawned_bins[:3]):
            # Get bin position
            bin_pos = bin_actor.pose.p
            if isinstance(bin_pos, torch.Tensor):
                bin_pos = bin_pos[0].detach().cpu().numpy()

            cube_position = [bin_pos[0], bin_pos[1]]
            # Generate cube using fixed position, colors red, green, blue
            cube_actor = spawn_fixed_cube(
                self,
                position=cube_position,
                half_size=self.cube_half_size/hidden_cfg["half_size_divisor"],
                color=cube_colors[i],  # Use red, green, blue in order
                name_prefix=f"target_cube_{color_names[i]}",
                yaw=hidden_cfg["yaw"],  # No rotation
                dynamic=True
            )

            spawned_dynamic_cubes.append(cube_actor)
            # Assign cube to self.target_cube_red, self.target_cube_green, self.target_cube_blue etc. attributes
            setattr(self, f"target_cube_{color_names[i]}", cube_actor)
            # Also store using numeric index for easy access
            setattr(self, f"target_cube_{i}", cube_actor)
            # Add newly generated cube to avoidance list
            avoid.append(cube_actor)



        tasks = [
            {
                "func": lambda: is_button_pressed(self, obj=self.button_left),
                "name": "press the button",
                "subgoal_segment":"press the button at <>",
                "choice_label": "press the button",
                "demonstration": False,
                "failure_func":None,
                "solve": lambda env, planner: solve_button(env, planner, obj=self.button_left),
                "segment":self.cap_link,
            },]
        tasks.append(
                    {
                        "func": (lambda: is_bin_pickup(self, obj=self.bin_0)),
                        "name": f"pick up the container that hides the {self.color_names[0]} cube",
                        "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[0]} cube",
                        "choice_label": "pick up the container",
                        "demonstration": False,
                        "failure_func": lambda: [
                                is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.bin_0]), ],
                        "solve": lambda env, planner: [solve_pickup_bin(env, planner, obj=self.bin_0)],
                         "segment":self.bin_0,
                    })
        if xhard:
            # V4 xhard: per pick_count, loop-append "put down the previous -> pick the k-th" (the original branch hardcodes bin_0/bin_1, so only 2 picks);
            # the single-element list form of the first pickup above is kept as is
            self._append_xhard_pick_tasks(tasks, decision_cfg["pick_count"][self.difficulty])
        elif decision_cfg["pick_count"][self.difficulty]>1:
            tasks.append({
                    "func": (lambda: is_bin_putdown(self, obj=self.bin_0)),
                    "name": "put down the container",
                    "subgoal_segment":"put down the container",
                    "choice_label": "put down the container",
                    "demonstration": False,
                    "failure_func": lambda:is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.bin_0]),
                    "solve": lambda env, planner: solve_putdown_whenhold(env, planner),
                })
            tasks.append(
                {
                    "func": (lambda: is_bin_pickup(self, obj=self.bin_1)),
                        "name": f"pick up the container that hides the {self.color_names[1]} cube",
                        "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[1]} cube",
                    "choice_label": "pick up the container",
                    "demonstration": False,
                    "failure_func": lambda: is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.bin_1]),
                    "solve": lambda env, planner: solve_pickup_bin(env, planner, obj=self.bin_1),
                    "segment":self.bin_1,
                })
        self.task_list = tasks
        # Set recovery related attributes
        # Record pickup related task indices and items for recovery
        self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
        if self.robomme_failure_recovery:
            # Only inject an intentional failed grasp when recovery mode is enabled
            # Choosing the recovery action is a real draw: the original draw happens as usual; re-injection mode uses the frozen index
            self.fail_grasp_task_index = self._spec.value(
                "actions.recovery.selected_action_index",
                inject_fail_grasp(
                self.task_list,
                generator=self.generator,
                mode=self.robomme_failure_recovery_mode,
            ),
            )
        else:
            self.fail_grasp_task_index = None

        if xhard:
            # V4 xhard distractor bins: must come after all existing value points (red line N5). The last existing draw from the scene-local generator
            # is color_order; recovery draws use the constructor's self.generator, the two streams do not interact. Button OBB is avoided via avoid as well.
            # V5 (L13): unified distractor sampler; still uses the main scene generator, still after all existing value points; inner-ring values bit-identical to V4 for the same seed
            self.distractor_bins, self.distractor_cubes, self.distractor_layout = spawn_distractor_layout(
                self,
                cfg=decision_cfg[self.difficulty]["distractor"],
                decision_prefix=f"{self.difficulty}.distractor",
                avoid=avoid,
                generator=generator,
                recorder=self._spec,
                hidden_half_size=self.cube_half_size/hidden_cfg["half_size_divisor"],
            )
            # V4 xhard (user 2026-09-22 "wrong grasp = failure"): every pick/place task that already has failure_func gets appended
            # "fail if any distractor bin is lifted (z>0.15, same criterion as in-region bins)"; original three tiers do not enter this branch
            add_distractor_misgrasp_failure(self, self.task_list)

    def _append_xhard_pick_tasks(self, tasks, pick_total):
        """xhard only: append picks 2..pick_total to the task list as "put down the previous bin -> pick the next".

        Each entry mirrors the original hard branch's second pick item by item, only replacing hardcoded bin_0/bin_1 and color_names[0]/[1] with index k;
        lambdas bind the current bin via default arguments to avoid late binding of the loop variable.
        """
        if pick_total > min(len(self.spawned_bins), len(self.color_names)):
            raise SceneGenerationError(
                f"pick_count={pick_total} exceeds the number of pickable hiding bins {min(len(self.spawned_bins), len(self.color_names))}"
            )
        # The task goal text (utils/task_goal.py) reads this actual count under xhard
        self.xhard_pick_count = pick_total
        self._spec.record("objects.n_picks", pick_total)
        self._spec.record("objects.pick_order", list(range(pick_total)))
        for k in range(1, pick_total):
            prev_bin = getattr(self, f"bin_{k-1}")
            cur_bin = getattr(self, f"bin_{k}")
            color = self.color_names[k]
            tasks.append({
                    "func": (lambda b=prev_bin: is_bin_putdown(self, obj=b)),
                    "name": "put down the container",
                    "subgoal_segment":"put down the container",
                    "choice_label": "put down the container",
                    "demonstration": False,
                    "failure_func": lambda b=prev_bin: is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != b]),
                    "solve": lambda env, planner: solve_putdown_whenhold(env, planner),
                })
            tasks.append(
                {
                    "func": (lambda b=cur_bin: is_bin_pickup(self, obj=b)),
                    "name": f"pick up the container that hides the {color} cube",
                    "subgoal_segment":f"pick up the container at <> that hides the {color} cube",
                    "choice_label": "pick up the container",
                    "demonstration": False,
                    "failure_func": lambda b=cur_bin: is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != b]),
                    "solve": lambda env, planner, b=cur_bin: solve_pickup_bin(env, planner, obj=b),
                    "segment":cur_bin,
                })

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
        all_tasks_completed, current_task_name, task_failed ,self.current_task_specialflag= sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

        #print(f"Current Task: {current_task_name}")
        
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

        total_bins = len(self.spawned_bins)
        if idx_a >= total_bins or idx_b >= total_bins:
            return []

        # Prefer precomputed lists when available
        if hasattr(self, "otherbins") and idx_a < len(self.otherbins):
            other_candidates = [
                bin_actor
                for bin_actor in self.otherbins[idx_a]
                if bin_actor is not self.spawned_bins[idx_b]
            ]
            return other_candidates

        return [
            bin_actor
            for i, bin_actor in enumerate(self.spawned_bins)
            if i not in (idx_a, idx_b)
        ]


#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


     
        timestep = self.elapsed_steps
        
                #Lift and drop bins (bin_0 to bin_4 if they exist)
        if is_newvalue_difficulty(self.difficulty):
            # V5 xhard (L14, main session 2026-09-24: inner-ring bins also use independent parking): window and half-window lowering steps identical to the original mechanism,
            # only "far away" changes from the shared (10,10,10) to a per-object off-screen parking point, so 20+ stacked bins do not slow physics.
            # Inner-ring bins scan the same bin_<i> (i < step_bin_scan) as the original loop; the i-th parks at xhard_park_point("bin", i);
            # hidden cubes do not move during this env's reveal, so no parking needed.
            reveal_window = self._sampling["positions"]["reveal_window"]
            reveal_actors_parked(
                self,
                [getattr(self, f"bin_{i}") for i in range(self._sampling["parameters"]["step_bin_scan"])
                 if hasattr(self, f"bin_{i}")],
                group="bin",
                start_step=reveal_window["start_step"],
                end_step=reveal_window["end_step"],
                cur_step=timestep,
            )
            # Distractor bins (user 2026-09-22 "take part in the reveal"): same window, one parking point each
            reveal_distractor_bins_parked(
                self,
                start_step=reveal_window["start_step"],
                end_step=reveal_window["end_step"],
                cur_step=timestep,
            )
        else:
            for i in range(self._sampling["parameters"]["step_bin_scan"]):
                bin_attr = f"bin_{i}"
                if hasattr(self, bin_attr):
                    lift_and_drop_objects_back_to_original(
                        self,
                        obj=getattr(self, bin_attr),
                        start_step=self._sampling["positions"]["reveal_window"]["start_step"],
                        end_step=self._sampling["positions"]["reveal_window"]["end_step"],
                        cur_step=timestep,
                    ) 

        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
