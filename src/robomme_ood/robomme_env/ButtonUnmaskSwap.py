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
# V5 L3 (following VideoPlaceOrder's K2 fix): the `from .utils import *` line above lets the same-named submodule
# `utils.SceneGenerationError` shadow the name `SceneGenerationError` (confirmed by import introspection), so in the original three tiers
# raise / except become TypeError (original three tiers kept as is per H2). xhard uses the alias below to get the real exception class.
from .utils.SceneGenerationError import SceneGenerationError as _RealSceneGenerationError
from .utils.subgoal_evaluate_func import static_check
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils import reset_panda
from .utils.difficulty import NEWVALUE_DIFFICULTIES, is_newvalue_difficulty, newvalue_tier, normalize_robomme_difficulty
from .utils.episode_spec import SpecRecorder
from .utils.sampling_config import assert_native_decision, split_sampling_config
from .utils.bin_collision import BinCollisionError, SpecBindingError, check_bin_state, check_swap_sweep, object_state_from_actor
from .utils.unmask_distractors import add_distractor_misgrasp_failure
from .utils.unmask_distractor_sampler import (
    park_cubes_onto_bins,
    reveal_actors_parked,
    reveal_distractor_bins_parked,
)
from .utils.unmask_swap_xhard import (
    SWAP_WINDOW_START,
    SWAP_WINDOW_STEPS,
    NEWVALUE_SWAP_SPEED_MULTIPLIER,
    LEGACY_V4_DISTRACTOR,
    distractor_generator,
    joint_sweep_from_actual,
    run_outer_swaps,
    scaled_window_steps,
    spawn_swap_distractors_v5,
    v5_distractor_cfg,
    v5_distractor_swap_cfg,
    # V6 (plan 2.2): inner ring S5 and outer ring O4
    plan_inner_swaps_v6,
    v6_distractor_cfg,
    v6_distractor_swap_cfg,
    v6_inner_swap_plan_cfg,
    validate_hidden_bin_selection,
)
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


# ── Original values of the decision/native blocks (newtaskRelease-v3 step 3, mapping in plan section 2.8) ────────
NATIVE_SAMPLING = {
    "parameters": {
        "bin_count": "FROM_CLASS_CONFIGS",
        "color_pool": [
            {"rgba": [1, 0, 0, 1], "name": "red"},
            {"rgba": [0, 1, 0, 1], "name": "green"},
            {"rgba": [0, 0, 1, 1], "name": "blue"},
        ],
        "color_order": {"sampler": "torch.randperm(3)"},
        "hidden_rule": "first three bins hide the three colors, the fourth is empty",
        "pick_rule": "after left button -> right button (robot frame, left = +y), pick selected_bins[0], and [1] too when count=2",
        "partner_rule": "at swap start take nearest neighbor by actual XY; no extra draw",
        # Swap window: first segment start and per-segment steps of the original three tiers (named constants; the six original literals now read these);
        # actually consumed since V4; xhard per-segment steps = round(duration_steps / 1.5) = 33 (2.11)
        "swap_window": {"start_step": SWAP_WINDOW_START, "duration_steps": SWAP_WINDOW_STEPS},
        "swap_path": {"lane_offset": 0.07, "smooth": True, "keep_upright": True},
        # V6 review fix F3 (user K4 "all left/right are in the robot frame"): button naming aligned to the robot frame (robot faces +x, left = +y);
        # construction order, positions and RNG consumption all unchanged, only name and button_order literals change: buttons[0] (y=-0.1) is the right button, buttons[1] (y=+0.1) is the left button,
        # the task chain still presses buttons[1] (left) then buttons[0] (right), the same physical buttons as before the rename.
        "button_order": ["left", "right"],
        "recovery": "the constructor's self.generator is used for recovery; the scene builds its own local stream with the same seed; the two streams are separate",
    },
    "positions": {
        "buttons": [
            {"name": "button_right", "center_xy": [-0.2, -0.1], "scale": 1.5,
             "randomize": True, "randomize_range": [0.05, 0.05]},
            {"name": "button_left", "center_xy": [-0.2, 0.1], "scale": 1.5,
             "randomize": True, "randomize_range": [0.05, 0.05]},
        ],
        "anchors": {
            "four_point": [[0, -0.1], [0, 0.1], [0.1, 0.1], [0.1, -0.1]],
            "triangle": [[-0.05, -0.15], [-0.05, 0.15], [0.05, 0]],
            "line": [[-0.05, -0.15], [-0.05, 0.15], [-0.05, 0]],
            "offset_scale": 0.1,
            "offset_note": "the four points draw one y offset per group of two; the three points each draw one x offset; "
                            "the unselected branch still consumes random numbers as usual (red line R8)",
            "choice_sampler": "torch.randint(0, 2)",
        },
        "bins": {"region_half_size": 0.07, "min_gap_factor": 1, "max_trials": 256},
        "hidden_cube": {"half_size_divisor": 1.2, "yaw": 0.0},
    },
}


def native_blocks(cls, *, release="newtask-v6"):
    """Original ``(decision, native)`` blocks of this env; shared by external export and internal parsing."""
    native = copy.deepcopy(NATIVE_SAMPLING)
    if release in ("newtask-v4", "newtask-v5"):
        # V4/V5 snapshots froze the pre-rename button naming (before F3: buttons[0] named button_left, order ["right","left"]);
        # when exporting old releases, backfill the frozen values so V5 snapshots reproduce byte for byte. Physical objects and task chain order are the same in both.
        native["parameters"]["button_order"] = ["right", "left"]
        native["parameters"]["pick_rule"] = "after right button -> left button, pick selected_bins[0], and [1] too when count=2"
        native["positions"]["buttons"][0]["name"] = "button_left"
        native["positions"]["buttons"][1]["name"] = "button_right"
        legacy_configs = {
            "easy": cls.configs["easy"], "medium": cls.configs["medium"], "hard": cls.configs["hard"],
            "xhard": {"bin": 4, "swap_min": 6, "swap_max": 8, "pick_min": 3, "pick_max": 3},
        }
        native["parameters"]["bin_count"] = {
            difficulty: cfg["bin"] for difficulty, cfg in legacy_configs.items()
        }
        decision = _legacy_decision(legacy_configs, release)
        return decision, native
    # newtask-v7: same parsing path as v6, using current class constants (i.e. V7 fixed values; v6 values live only in the packaged v6 spec header, 0928 plan R3)
    if release not in ("newtask-v6", "newtask-v7"):
        raise ValueError(f"ButtonUnmaskSwap does not support sampling_config release {release!r}")
    native["parameters"]["bin_count"] = {difficulty: cfg["bin"] for difficulty, cfg in cls.configs.items()}
    native["parameters"]["configs"] = copy.deepcopy(cls.configs)
    return _native_decision(cls), native


def _legacy_decision(configs, release):
    decision = {
        "swap_count_range": {difficulty: [cfg["swap_min"], cfg["swap_max"]]
                             for difficulty, cfg in configs.items()},
        "pick_count_range": {difficulty: [cfg["pick_min"], cfg["pick_max"]]
                             for difficulty, cfg in configs.items()},
        "swap_speed_multiplier": 1,
        "distractor": None,
        "xhard": {
            "swap_speed_multiplier": NEWVALUE_SWAP_SPEED_MULTIPLIER,
            "distractor": (copy.deepcopy(LEGACY_V4_DISTRACTOR) if release == "newtask-v4"
                           else v5_distractor_cfg("ButtonUnmaskSwap")),
        },
    }
    if release == "newtask-v5":
        decision["xhard"]["distractor_swap"] = v5_distractor_swap_cfg("ButtonUnmaskSwap")
    return decision


def _native_decision(cls):
    """Slice the decision block per plan section 2.8 (equals the original in the original-value stage)."""
    return {
        "swap_count_range": {
            difficulty: [cfg["swap_min"], cfg["swap_max"]] for difficulty, cfg in cls.configs.items()
        },
        "pick_count_range": {
            difficulty: [cfg["pick_min"], cfg["pick_max"]] for difficulty, cfg in cls.configs.items()
        },
        # Swap speed multiplier: original value 1 (50 steps per segment), consumed by the original three tiers (=1 => original 50 steps).
        "swap_speed_multiplier": 1,
        "distractor": None,
        # New-value tiers use per-tier speed and outer-ring count; bin layout mechanism follows the original tier.
        # V5 (2.7, L13/L16 b): distractor bins use the unified sampler preset (V4 ring band, 10 bins, cube-containing [5,5]);
        # new distractor_swap: rule for outer ring swapping in sync with the inner ring (L17-L23, outer path >= 0.122 from button center).
        **{
            tier: {
                "swap_speed_multiplier": 1.0 if newvalue_tier(tier) == 1 else NEWVALUE_SWAP_SPEED_MULTIPLIER,
                "distractor": v6_distractor_cfg("ButtonUnmaskSwap", 2 * newvalue_tier(tier)),  # V7: outer ring 2/4/6/8
                "distractor_swap": v6_distractor_swap_cfg("ButtonUnmaskSwap"),
                "swap_plan_v6": v6_inner_swap_plan_cfg("ButtonUnmaskSwap"),
            }
            for tier in NEWVALUE_DIFFICULTIES
        },
    }


def _resolve_sampling_config(cls, override):
    """Split out this instance's private decision/native copies; draws no random numbers, must be called before the Generator."""
    decision_default, native_default = native_blocks(cls, release="newtask-v6")
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    native["parameters"].setdefault("configs", copy.deepcopy(cls.configs))
    native["decision"] = decision
    return native


@register_env("ButtonUnmaskSwap", override=True)
class ButtonUnmaskSwap(BaseEnv):


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
        "bin":3,
        "swap_min":1,
        "swap_max":2,
        "pick_min":1,
        "pick_max":2
    }
    config_medium= {
        "bin":4,
        "swap_min":1,
        "swap_max":2,
        "pick_min":1,
        "pick_max":1
    }
    config_hard = {
        "bin":4,
        "swap_min":2,
        "swap_max":3,
        "pick_min":2,
        "pick_max":2
    }


    # V7 fixed values (0928 plan 3.2.2): swap 3/5/7/9, pick 2/3/3/3, outer ring 2/4/6/8; inner bin layout mechanism follows original xhard.
    config_xhard1 = {"bin":4, "swap_min":3, "swap_max":3, "pick_min":2, "pick_max":2}
    config_xhard2 = {"bin":4, "swap_min":5, "swap_max":5, "pick_min":3, "pick_max":3}
    config_xhard3 = {"bin":4, "swap_min":7, "swap_max":7, "pick_min":3, "pick_max":3}
    config_xhard4 = {
        "bin":4,
        "swap_min":9,
        "swap_max":9,
        "pick_min":3,
        "pick_max":3
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
    # Named constants of the swap window (B4); runtime reads native.swap_window (defaults are these two constants)
    SWAP_WINDOW_START = SWAP_WINDOW_START
    SWAP_WINDOW_STEPS = SWAP_WINDOW_STEPS
    

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     sampling_config=None,
                     native_episode_spec=None,
                     **kwargs):
        # Must happen before any RNG call and before super().__init__()
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "ButtonUnmaskSwap", {"seed": seed},
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
            seed_mod = seed % 3
            if seed_mod == 0:
                self.difficulty = "easy"
            elif seed_mod == 1:
                self.difficulty = "medium"
            else:  # seed_mod == 2
                self.difficulty = "hard"
        #self.difficulty = "hard"
        # Use seed to randomly determine number of repetitions (1-5)
        generator = torch.Generator()
        generator.manual_seed(seed)
        decision = self._sampling["decision"]
        self._is_newvalue = is_newvalue_difficulty(self.difficulty)
        difficulty_cfg = self._sampling["parameters"]["configs"][self.difficulty]
        swap_range = (difficulty_cfg["swap_min"], difficulty_cfg["swap_max"])
        swap_decision_key = f"configs.{self.difficulty}.swap_min/swap_max"
        pick_range = (difficulty_cfg["pick_min"], difficulty_cfg["pick_max"])
        pick_decision_key = f"configs.{self.difficulty}.pick_min/pick_max"
        self.swap_times = self._spec.value(
            "objects.n_swaps",
            torch.randint(swap_range[0], swap_range[1]+1, (1,), generator=generator).item(),
            decision_key=swap_decision_key,
        )
        logger.debug(f"Task will swap {self.swap_times} times")


        self.pick_times = self._spec.value(
            "objects.n_picks",
            torch.randint(pick_range[0], pick_range[1]+1, (1,), generator=generator).item(),
            decision_key=pick_decision_key,
        )
        logger.debug(f"Task will pick {self.pick_times} times")

        # Swap window (2.11): native.swap_window is actually consumed; original three tiers take decision's top-level multiplier 1 => original {64, 50},
        # new-value tiers consume per-tier multipliers: tier 1 50 steps, the other three tiers 33 steps. No random draws.
        window = self._sampling["parameters"]["swap_window"]
        multiplier = decision[self.difficulty]["swap_speed_multiplier"] if self._is_newvalue else decision["swap_speed_multiplier"]
        self.swap_window_start = int(window["start_step"])
        self.swap_window_steps = scaled_window_steps(int(window["duration_steps"]), multiplier)
        # xhard runs runtime collision checks (H1: initial state + continuous sweep of each swap segment, including distractor bins); original three tiers do not check, behavior unchanged
        self._newvalue_collision_checks = self._is_newvalue
        self._runtime_checks = []
        # Distractor bins stored separately, not in spawned_bins; always empty for the original three tiers
        self.distractor_bins = []
        self.distractor_cubes = []
        if self._is_newvalue:
            # Outer-ring swap pairs, outer-ring cube follow pairs, reset-rehearsed inner pairs; filled by _spawn_newvalue_distractors
            self.distractor_swap_pairs = []
            self.distractor_cube_bin_pairs = []
            self.predicted_inner_swap_pairs = []
            self._spec.record(
                "actions.swap_window",
                {"start_step": self.swap_window_start, "duration_steps": self.swap_window_steps,
                 "speed_multiplier": multiplier},
            )

        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    def _refresh_swap_schedule(self):
        # General formula (same as VideoUnmaskSwap): the k-th swap occupies [S+Lk, S+L(k+1)], back to back;
        # S = swap_window_start (always 64), L = swap_window_steps (50 for the original three tiers, 33 for xhard).
        # For 1/2/3 swaps identical to the original three branches item by item; the original branches assign nothing for >=4 swaps (AttributeError in step); the formula fixes that too.
        if self.swap_times < 1:
            return
        start, length = self.swap_window_start, self.swap_window_steps
        self.swap_schedule = [
            (getattr(self, f"swap_pair{k+1}_idx1"), getattr(self, f"swap_pair{k+1}_idx2"), start + length * k, start + length * (k + 1))
            for k in range(self.swap_times)
        ]

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

        avoid=[]

        avoid=[]
        buttons_cfg = self._sampling["positions"]["buttons"]
        anchors_cfg = self._sampling["positions"]["anchors"]
        bins_cfg = self._sampling["positions"]["bins"]
        button_obb_1 = build_button(
            self,
            center_xy=tuple(buttons_cfg[0]["center_xy"]),
            scale=buttons_cfg[0]["scale"],
            generator=generator,
            name=buttons_cfg[0]["name"],
            randomize=buttons_cfg[0]["randomize"],
            randomize_range=tuple(buttons_cfg[0]["randomize_range"])
        )
        # Store first button before building second one
        # F3: buttons[0] (y=-0.1) is the right button in the robot frame; cap link stored per object, no longer looked up by config name
        self.button_right = self.button
        self.button_joint_1 = self.button_joint
        self.button_right_cap_link = self.cap_link

        avoid = [button_obb_1]

        button_obb_2 = build_button(
            self,
            center_xy=tuple(buttons_cfg[1]["center_xy"]),
            scale=buttons_cfg[1]["scale"],
            generator=generator,
            name=buttons_cfg[1]["name"],
            randomize=buttons_cfg[1]["randomize"],
            randomize_range=tuple(buttons_cfg[1]["randomize_range"])
        )
        # Store second button (buttons[1], y=+0.1, left button in the robot frame)
        self.button_left = self.button
        self.button_joint_2 = self.button_joint
        self.button_left_cap_link = self.cap_link

         # Generate 3 bins
        self.spawned_bins = []
        # Generate y offsets for region4 using torch generator
        offset_scale = anchors_cfg["offset_scale"]
        four_point = anchors_cfg["four_point"]
        y_offset_1 = (torch.rand(1, generator=generator).item()) * offset_scale  # for first two points
        y_offset_2 = (torch.rand(1, generator=generator).item()) * offset_scale  # for last two points

        region4=[[four_point[0][0], four_point[0][1] + y_offset_1],
                 [four_point[1][0], four_point[1][1] + y_offset_1],
                 [four_point[2][0], four_point[2][1] + y_offset_2],
                 [four_point[3][0], four_point[3][1] + y_offset_2]]


        # Generate independent random x offsets for each point using torch generator
        triangle = anchors_cfg["triangle"]
        line = anchors_cfg["line"]
        x_offset_tri_1 = (torch.rand(1, generator=generator).item()) * offset_scale
        x_offset_tri_2 = (torch.rand(1, generator=generator).item()) * offset_scale
        x_offset_tri_3 = (torch.rand(1, generator=generator).item()) * offset_scale

        x_offset_line_1 = (torch.rand(1, generator=generator).item()) * offset_scale
        x_offset_line_2 = (torch.rand(1, generator=generator).item()) * offset_scale
        x_offset_line_3 = (torch.rand(1, generator=generator).item()) * offset_scale

        region3_tri=[[triangle[0][0] + x_offset_tri_1, triangle[0][1]],
                     [triangle[1][0] + x_offset_tri_2, triangle[1][1]],
                     [triangle[2][0] + x_offset_tri_3, triangle[2][1]]]
        region3_line=[[line[0][0] + x_offset_line_1, line[0][1]],
                      [line[1][0] + x_offset_line_2, line[1][1]],
                      [line[2][0] + x_offset_line_3, line[2][1]]]

        # Use generator to randomly select region3_tri or region3_line
        region3_choice = self._spec.value(
            "layout.type_choice", torch.randint(0, 2, (1,), generator=generator).item()
        )
        region3 = region3_tri if region3_choice == 0 else region3_line

        if self._sampling["parameters"]["bin_count"][self.difficulty]==4:
            region=region4
        else:
             region=region3
        #angle, region = rotate_points_random(region,(0,180),generator)

        # # Safety check: ensure x coordinates are not less than 0
        # for i in range(len(region)):
        #     if region[i][0] < -0:
        #         region[i][0] = 0

        for i in range(self._sampling["parameters"]["bin_count"][self.difficulty]):
            try:
                bin_actor = spawn_random_bin(
                    self,
                    avoid=avoid,  # Use current avoidance list, containing all spawned objects
                    region_center=region[i],
                    region_half_size=bins_cfg["region_half_size"],
                    min_gap=self.cube_half_size*bins_cfg["min_gap_factor"],  # bins need larger gap, increased to 6x to avoid collision
                    name_prefix=f"bin_{i}",
                    max_trials=bins_cfg["max_trials"],
                    generator=generator,
                    recorder=self._spec,
                    spec_path=f"layout.bins.{i}"
                )
            except RuntimeError as e:
                if self._is_newvalue:
                    # V5 L15: xhard must not truncate silently (truncation would change swap pairs and ring-band obstacles); raise a real exception to trigger a candidate-level redraw
                    raise _RealSceneGenerationError(f"xhard inner-ring bin bin_{i} does not fit: {e}") from e
                break

            self.spawned_bins.append(bin_actor)
            # Assign bin to self.bin_0, self.bin_1 etc. attributes
            setattr(self, f"bin_{i}", bin_actor)
            # Add newly generated bin to avoidance list
            avoid.append(bin_actor)


        # Generate 3 dynamic cubes under each bin (using fixed position, colors red, green, blue)
        spawned_dynamic_cubes = []
        self.cube_bin_pairs = []
        self.bin_to_cube = {}
        self.bin_to_color = {}
        self.spawned_dynamic_cubes = spawned_dynamic_cubes
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

        # Randomly select 3 bins from all bins to spawn cube
        num_bins_to_select = min(3, len(self.spawned_bins))
        # M5(b): hiding positions are still chosen only from the first three bins; the fourth bin bin_3 is always empty.
        selected_bin_indices = self._spec.value(
            "objects.selected",
            torch.randperm(3, generator=generator)[:num_bins_to_select].tolist(),
        )
        if self._is_newvalue:
            selected_bin_indices = validate_hidden_bin_selection(selected_bin_indices, permutation_size=3)
        selected_bins = [self.spawned_bins[idx] for idx in selected_bin_indices]
        self.selected_bin_indices = selected_bin_indices
        self.selected_bins = selected_bins  # Save selected bins, corresponding to color_names order

        for i, (bin_idx, bin_actor) in enumerate(zip(selected_bin_indices, selected_bins)):
            # Get bin position
            bin_pos = bin_actor.pose.p
            if isinstance(bin_pos, torch.Tensor):
                bin_pos = bin_pos[0].detach().cpu().numpy()

            cube_position = [bin_pos[0], bin_pos[1]]
            # Generate cube using fixed position, colors red, green, blue
            cube_actor = spawn_fixed_cube(
                self,
                position=cube_position,
                half_size=self.cube_half_size/self._sampling["positions"]["hidden_cube"]["half_size_divisor"],
                color=cube_colors[i],  # Use red, green, blue in order
                name_prefix=f"target_cube_{color_names[i]}",
                yaw=self._sampling["positions"]["hidden_cube"]["yaw"],  # No rotation
                dynamic=True
            )

            spawned_dynamic_cubes.append(cube_actor)
            # Assign cube to self.target_cube_red, self.target_cube_green, self.target_cube_blue etc. attributes
            setattr(self, f"target_cube_{color_names[i]}", cube_actor)
            # Also store using numeric index for easy access
            setattr(self, f"target_cube_{i}", cube_actor)
            setattr(self, f"target_cube_for_bin_{bin_idx}", cube_actor)
            self.cube_bin_pairs.append((cube_actor, bin_actor))
            self.bin_to_cube[bin_idx] = cube_actor
            self.bin_to_color[bin_idx] = color_names[i]

            # Add newly generated cube to avoidance list
            avoid.append(cube_actor)

        self.cube_bins = selected_bins
        self.cube_bin_indices = selected_bin_indices
        self.target_bin = None
        self.target_bin_index = None
        self.target_cube = None
        self.target_cube_color = None
        self.other_cube_bins = []
        self.other_cube_bin_indices = []
        self.other_cubes = []

        if self.cube_bin_pairs:
            target_choice = self._spec.value("objects.target_choice", int(
                torch.randint(
                    len(self.cube_bin_pairs),
                    (1,),
                    generator=generator,
                ).item()
            ))
            target_cube_actor, target_bin_actor = self.cube_bin_pairs[target_choice]
            self.target_cube = target_cube_actor
            self.target_bin = target_bin_actor
            self.target_bin_index = selected_bin_indices[target_choice]
            self.target_cube_color = color_names[target_choice]
            self.target_cube_name = (
                getattr(target_cube_actor, "name", None)
                or f"target_cube_{self.target_cube_color}"
            )
            self.target_label = self.target_cube_color or self.target_cube_name or "target"

            for idx_i, (cube_actor, bin_actor) in enumerate(self.cube_bin_pairs):
                if idx_i == target_choice:
                    continue
                self.other_cube_bins.append(bin_actor)
                self.other_cube_bin_indices.append(selected_bin_indices[idx_i])
                self.other_cubes.append(cube_actor)
        else:
            self.target_cube = None
            self.target_bin = None
            self.target_bin_index = None
            self.target_cube_color = None
            self.target_cube_name = None
            self.target_label = "target"

       # Randomly select 2 unique bins as target_bin_1 and target_bin_2
        # target_indices is index to selected_bin_indices (0, 1, 2)
        # The spec stores integer lists; downstream still uses tensors (.item()/.tolist()/torch.cat), so wrap back into a tensor
        target_indices = torch.tensor(self._spec.value(
            "objects.swap_initiator_indices",
            torch.randperm(len(selected_bin_indices), generator=generator)[:2].tolist(),
        ))
        # Use selected_bins to get correct bin (corresponding to color_names order)
        self.target_bin_1=self.selected_bins[target_indices[0]]
        self.target_bin_2=self.selected_bins[target_indices[1]]
        # Record cube colors corresponding to these two bins, index directly using color_names
        self.target_bin_1_cube_color = color_names[target_indices[0].item()]
        self.target_bin_2_cube_color = color_names[target_indices[1].item()]
        # swap_indices must include target_indices, then select 1 from remaining indices
        remaining_indices = [i for i in range(len(self.spawned_bins)) if i not in target_indices.tolist()]
        if remaining_indices:
            third_idx = self._spec.value(
                "objects.swap_initiator_third",
                remaining_indices[torch.randint(0, len(remaining_indices), (1,), generator=generator).item()],
            )
            swap_indices = torch.cat([target_indices, torch.tensor([third_idx])])
        else:
            swap_indices = target_indices
        self.swap_pair1_idx1=self.spawned_bins[swap_indices[0]]
        self.swap_pair2_idx1=self.spawned_bins[swap_indices[1]]
        self.swap_pair3_idx1=self.spawned_bins[swap_indices[2]]
        self.swap_pair1_idx2=None
        self.swap_pair2_idx2=None
        self.swap_pair3_idx2=None
        # V4 xhard (6-8 swaps): the k-th initiator cycles through the first 3 (a,b,c,a,b,...); with <= 3 swaps this loop does not run
        for k in range(3, self.swap_times):
            setattr(self, f"swap_pair{k+1}_idx1", self.spawned_bins[swap_indices[k % 3]])
            setattr(self, f"swap_pair{k+1}_idx2", None)
        if self._is_newvalue:
            # V6 (plan 2.2 inner ring S5): the V5 initiator value point is drawn as before, then the main stream appends one planning seed; S5 pre-plans the whole sequence and overrides the initiators;
            # partners are read from the pre-plan by step's xhard branch
            self._plan_inner_swaps_v6(generator)


        self._refresh_swap_schedule()

        self.button_list= [self.button_right, self.button_left]  # F3: order kept as [buttons[0] object, buttons[1] object]
        self.generator=generator

        if self._is_newvalue:
            # V4 xhard distractor bins: use a dedicated random stream; the main stream (including self.generator that
            # inject_fail_grasp in _initialize_episode keeps consuming) draws not a single extra number (N5)
            self._spawn_newvalue_distractors([button_obb_1, button_obb_2])

    def _plan_inner_swaps_v6(self, generator):
        """V6 xhard inner-ring S5 reset pre-planning (see ``unmask_swap_xhard.plan_inner_swaps_v6``); raises a real ``SceneGenerationError`` if G is disconnected."""
        plan = plan_inner_swaps_v6(self, generator)
        self._newvalue_inner_plan = plan
        self._newvalue_swap_partners = [int(b) for _a, b in plan["pairs"]]
        for k, (a, _b) in enumerate(plan["pairs"]):
            setattr(self, f"swap_pair{k+1}_idx1", self.spawned_bins[int(a)])
            setattr(self, f"swap_pair{k+1}_idx2", None)

    def _newvalue_planned_partner(self, sweep_index, initiator):
        """Partner actor planned by S5 for swap ``sweep_index`` (called by the new-value branch of ``step``)."""
        partners = getattr(self, "_newvalue_swap_partners", None)
        if partners is None or sweep_index >= len(partners):
            raise SpecBindingError(f"{self.difficulty}: swap {sweep_index} has no reset-planned partner")
        partner = self.spawned_bins[partners[sweep_index]]
        if partner is initiator:
            raise SpecBindingError(f"{self.difficulty}: swap {sweep_index} planned partner is the same bin as the initiator")
        return partner

    def _spawn_newvalue_distractors(self, button_obbs):
        """New-value tier outer-ring distractor bins (unified sampler, button OBB as exact obstacle) + reset planning of outer ring swapping in sync with the inner ring.

        Same structure as VideoUnmaskSwap (L20 pre-check -> placement + H1 -> outer-ring planning, at most 16 full redraws), plus outer path
        >= 0.122 from both button centers (L19). All sampling uses an independent stream; the main stream (including self.generator that
        inject_fail_grasp in _initialize_episode keeps consuming) draws not a single extra number; distractor bins still stay out of spawned_bins (this env has no _verify_swap_binding; mixing them in would silently change swap pairs).
        """
        result = spawn_swap_distractors_v5(
            self,
            generator=distractor_generator(self.seed),
            # This env's inner-ring nearest neighbor is hardcoded [:2] (XY); the rehearsal uses the same axes
            partner_axes=[0, 1],
            button_obbs=list(button_obbs),
            hidden_half_size=self.cube_half_size / self._sampling["positions"]["hidden_cube"]["half_size_divisor"],
        )
        self.distractor_bins = result.bins
        self.distractor_cubes = result.cubes
        self.distractor_cube_bin_pairs = result.cube_bin_pairs
        self.distractor_cube_colors = result.layout.cube_colors
        self.distractor_layout = result.layout
        self.distractor_swap_pairs = result.pairs
        self.predicted_inner_swap_pairs = result.predicted_inner_pairs
        self._distractor_plan_timing = result.timing

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            if getattr(self, "_newvalue_collision_checks", False):
                # V4 xhard initial-state re-check (H1): bins + distractor bins pairwise read real collision boxes
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
        tasks = [
            {
                "func": lambda: is_any_button_pressed_removelist(self, button_list=self.button_list),
                "name": "press the first button",
                "subgoal_segment":"press the first button at <>",
                "choice_label": "press the first button",
                "demonstration": False,
                "failure_func":None,
                "solve": lambda env, planner: solve_button(env, planner, obj=self.button_left),
                "segment":self.button_left_cap_link
            },
                  {
                "func": lambda: is_any_button_pressed_removelist(self, button_list=self.button_list),
                "name": "press the second button",
                "subgoal_segment":"press the second button at <>",
                "choice_label": "press the second button",
                "demonstration": False,
                "failure_func":None,
                "solve": lambda env, planner: solve_button(env, planner, obj=self.button_right),
                "segment":self.button_right_cap_link
            },

            {
                "func": (lambda: is_bin_pickup(self, obj=self.selected_bins[0])),
                "name": f"pick up the container that hides the {self.color_names[0]} cube",
                "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[0]} cube",
                "choice_label": "pick up the container",
                "demonstration": False,
                "failure_func": lambda: is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.selected_bins[0]]),
                "solve": lambda env, planner: [solve_pickup_bin(env, planner, obj=self.selected_bins[0])],
                "segment":self.selected_bins[0]
            }
        ]
        if self.pick_times==2:
            tasks.append({
                    "func": (lambda: is_bin_putdown(self, obj=self.selected_bins[0])),
                    "name": "put down the container",
                    "subgoal_segment":"put down the container",
                    "choice_label": "put down the container",
                    "demonstration": False,
                    "failure_func": lambda:is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.selected_bins[0]]),
                    "solve": lambda env, planner: solve_putdown_whenhold(env, planner),
                })
            tasks.append(
                {
                    "func": (lambda: is_bin_pickup(self, obj=self.selected_bins[1])),
                        "name": f"pick up the container that hides the {self.color_names[1]} cube",
                        "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[1]} cube",
                    "choice_label": "pick up the container",
                    "demonstration": False,
                    "failure_func": lambda: is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.selected_bins[1]]),
                    "solve": lambda env, planner: solve_pickup_bin(env, planner, obj=self.selected_bins[1]),
                    "segment":self.selected_bins[1],
                })
        if self._is_newvalue:
            # V4 xhard: all tasks in this env have demonstration=False; after both buttons are pressed it is usually only around step 200,
            # while 6-8 swaps end at 64+33n (262-328); the original solver would grasp while bins are still swapping (measured 2/2 failures).
            # V6 review fix N1 (user "n1 subgoal set to wait"): waiting is no longer hidden inside the second button's solver (then the subgoal during waiting would still read
            # "press the second button"), but inserted as an independent subgoal "wait for the containers to finish swapping"
            # after the second button: completion = reached the end step of the last segment of the swap schedule; solver = wait in place until that absolute step.
            # Still not placed on the first pick, because inject_fail_grasp replaces the solve of the selected pick task entirely.
            tasks.insert(2, {
                "func": lambda: int(self.elapsed_steps) >= int(self.swap_schedule[-1][3]),
                "name": "wait for the containers to finish swapping",
                "subgoal_segment": "wait for the containers to finish swapping",
                "choice_label": "wait",
                "demonstration": False,
                "failure_func": None,
                "solve": lambda env, planner: self._solve_wait_swaps(env, planner),
            })
        if self._is_newvalue and self.pick_times > 2:
            # V4 xhard: the original branch uses strict `== 2`, so pick=3 would fall back to a single pick; here we loop over pick_times picking
            # selected_bins[0..pick_times-1], putting down the previous before each pick. lambdas bind this round's object via default arguments.
            if self.pick_times > len(self.selected_bins):
                raise ValueError(f"pick_times={self.pick_times} exceeds the number of hiding bins {len(self.selected_bins)}")
            for j in range(1, self.pick_times):
                prev_bin = self.selected_bins[j - 1]
                cur_bin = self.selected_bins[j]
                cur_color = self.color_names[j]
                tasks.append({
                    "func": (lambda prev_bin=prev_bin: is_bin_putdown(self, obj=prev_bin)),
                    "name": "put down the container",
                    "subgoal_segment": "put down the container",
                    "choice_label": "put down the container",
                    "demonstration": False,
                    "failure_func": lambda prev_bin=prev_bin: is_any_bin_pickup(self, [bin for bin in self.spawned_bins if bin != prev_bin]),
                    "solve": lambda env, planner: solve_putdown_whenhold(env, planner),
                })
                tasks.append({
                    "func": (lambda cur_bin=cur_bin: is_bin_pickup(self, obj=cur_bin)),
                    "name": f"pick up the container that hides the {cur_color} cube",
                    "subgoal_segment": f"pick up the container at <> that hides the {cur_color} cube",
                    "choice_label": "pick up the container",
                    "demonstration": False,
                    "failure_func": lambda cur_bin=cur_bin: is_any_bin_pickup(self, [bin for bin in self.spawned_bins if bin != cur_bin]),
                    "solve": lambda env, planner, cur_bin=cur_bin: solve_pickup_bin(env, planner, obj=cur_bin),
                    "segment": cur_bin,
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
        if self._is_newvalue:
            # V4 xhard (user 2026-09-22 "wrong grasp = failure"): every pick/place task that already has failure_func gets appended
            # "fail if any distractor bin is lifted (z>0.15, same criterion as in-region bins)"
            add_distractor_misgrasp_failure(self, self.task_list)
            # V6 review fix F4 (K5): tag inner-ring bins -- when the subgoal has not switched but the target segmentation center moves more than 8 pixels,
            # process_segmentation recomputes grounded coordinates (no longer reusing the old position after a swap). Untagged actors (original three tiers, other envs) take the original branch.
            for bin_actor in self.spawned_bins:
                bin_actor._robomme_refresh_on_move_px = 8
            
    def _get_obs_extra(self, info: Dict):
        return dict()



    def evaluate(self,solve_complete_eval=False):
        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([False])
        target_color = getattr(self, "target_cube_color", None)
        if target_color is None and getattr(self, "color_names", None):
            target_color = self.color_names[0]
        if target_color is None:
            target_color = getattr(self, "target_label", None)
        if target_color is None:
            target_color = "target"
        self.target_label = target_color


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

        # If task failed, mark as failed immediately
        if task_failed:
            self.failureflag = torch.tensor([True])
            logger.debug(f"Task failed: {current_task_name}")
        else:
            self.failureflag = torch.tensor([False])

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

    def _solve_wait_swaps(self, env, planner):
        """V6 N1: solver of the wait subgoal -- wait in place until the last swap segment ends (absolute step swap_schedule[-1][3]); return immediately if already past."""
        solve_hold_obj_absTimestep(env, planner, absTimestep=self.swap_schedule[-1][3])
        return None

    def _solve_press_then_wait_swaps(self, env, planner, button):
        """V4 xhard-only solver: after pressing the button, wait in place until the last swap segment ends (absolute step swap_schedule[-1][3]).

        If the button solver fails (returns -1), return it as is without swallowing the failure; return immediately if the swap end time has passed.
        """
        result = solve_button(env, planner, obj=button)
        if isinstance(result, (int, np.integer)) and int(result) == -1:
            return result
        solve_hold_obj_absTimestep(env, planner, absTimestep=self.swap_schedule[-1][3])
        return result

    def _object_states_for_collision(self):
        """V4 xhard (H1): read all bins + distractor bins into the state used by the collision criterion (real collision boxes)."""
        return [
            object_state_from_actor(actor, f"bin_{index}")
            for index, actor in enumerate(self.spawned_bins)
            if actor is not None
        ] + [
            object_state_from_actor(actor, f"distractor_bin_{index}")
            for index, actor in enumerate(getattr(self, "distractor_bins", []))
            if actor is not None
        ]

    def _check_state_readonly(self, stage):
        """Read-only re-check at one instant; returns rejection evidence without raising, the caller decides how to handle it."""
        return check_bin_state(self._object_states_for_collision(), stage=stage)

    def _check_swap_sweep_from_actual(self, sweep_index, initiator, partner):
        """V4 xhard (H1): continuously check the whole swap path from actual poses; bystanders include the other bins and all distractor bins;
        on a hit raise ``BinCollisionError`` to abort the sample. Never called by the original three tiers.

        V5: xhard switches to a joint re-check of the inner pair (resolved at runtime, L22 a) and this window's outer pair (with certified prefilter, L23); the V4 single-pair
        branch below runs only on instances without ``_is_newvalue`` (this env's original three tiers never call this function)."""
        if getattr(self, "_is_newvalue", False):
            gap, rejection, info = joint_sweep_from_actual(self, sweep_index, initiator, partner)
            if info["inner_partner_mismatch"]:
                logger.warning(
                    f"ButtonUnmaskSwap xhard segment {sweep_index} inner pair {info['inner_pair']} differs from reset rehearsal "
                    f"{info['predicted_inner_pair']} (runtime still swaps by actual nearest neighbor; the joint re-check uses the actual pair)"
                )
                self._spec.record(f"actions.inner_swap_mismatch.{sweep_index}",
                                  {"runtime": info["inner_pair"], "predicted": info["predicted_inner_pair"]})
            self._runtime_checks.append(
                {
                    "kind": "swap_sweep",
                    "sweep_index": sweep_index,
                    "control_step": int(self.elapsed_steps),
                    "min_g_m": None if rejection is not None else gap,
                    "rejection": None if rejection is None else rejection.as_dict(),
                    **info,
                }
            )
            if rejection is not None:
                raise BinCollisionError(rejection)
            return
        states = {
            index: object_state_from_actor(actor, f"bin_{index}")
            for index, actor in enumerate(self.spawned_bins)
            if actor is not None
        }
        a = self.spawned_bins.index(initiator)
        b = self.spawned_bins.index(partner)
        bystanders = [state for index, state in sorted(states.items()) if index not in (a, b)]
        bystanders += [
            object_state_from_actor(actor, f"distractor_bin_{index}")
            for index, actor in enumerate(getattr(self, "distractor_bins", []))
            if actor is not None
        ]
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

    def _select_swap_pair_from_positions(self, positions, generator):
        """Select one swap pair given current planned positions."""
        num_bins = len(positions)
        if num_bins < 2:
            return None

        candidate_map = self._compute_dynamic_swap_candidates(positions)
        valid_indices = [idx for idx, cands in candidate_map.items() if cands]
        if not valid_indices:
            return None

        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(self.seed))
            self._swap_rng = generator

        first_idx = valid_indices[
            int(torch.randint(0, len(valid_indices), (1,), generator=generator).item())
        ]
        candidates = candidate_map[first_idx]
        second_idx = candidates[
            int(torch.randint(0, len(candidates), (1,), generator=generator).item())
        ]

        distance = float(
            np.linalg.norm(positions[first_idx][:2] - positions[second_idx][:2])
        )

        return {"idx1": first_idx, "idx2": second_idx, "distance": distance}


#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):



        timestep = self.elapsed_steps
        
        if self._is_newvalue:
            # V5 xhard (L14, main session decided inner ring uses it too): inner-ring bins and outer-ring distractor bins share the window [0, 64) and the reveal timeline,
            # but each object parks at its own off-screen point instead of all stacking at (10,10,10); distractor bins still stay out of spawned_bins
            reveal_actors_parked(self, getattr(self, "spawned_bins", []), group="bin",
                                 start_step=0, end_step=self.swap_window_start, cur_step=timestep)
            reveal_distractor_bins_parked(self, start_step=0, end_step=self.swap_window_start, cur_step=timestep)
        else:
            # Keep all spawned bins in their original placement during the pre-swap window
            for bin_actor in getattr(self, "spawned_bins", []):
                lift_and_drop_objects_back_to_original(
                    self,
                    obj=bin_actor,
                    start_step=0,
                    end_step=self.swap_window_start,  # End of the pre-swap lock segment = start of the first swap segment (64)
                    cur_step=timestep,
                )
        for i in range(len(self.swap_schedule)):
            start = self.swap_schedule[i][2]
            end = self.swap_schedule[i][3]
            if timestep in range (start,end):
                # Select corresponding swap pair based on index
                pair_idx1 = getattr(self, f'swap_pair{i+1}_idx1')
                pair_idx2 = getattr(self, f'swap_pair{i+1}_idx2')

                if pair_idx2 is None and pair_idx1 is not None:
                    if getattr(self, "_is_newvalue", False):
                        # V6 xhard (plan 2.2 inner ring 3): partner read from the S5 reset pre-plan; the two-pair joint re-check below still runs
                        closest_actor = self._newvalue_planned_partner(i, pair_idx1)
                    else:
                        reference_pos = self._get_actor_position(pair_idx1)
                        closest_actor = None
                        closest_dist = float("inf")
                        for candidate in self.spawned_bins:
                            if candidate is None or candidate is pair_idx1:
                                continue
                            candidate_pos = self._get_actor_position(candidate)
                            dist = np.linalg.norm(reference_pos[:2] - candidate_pos[:2])
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_actor = candidate
                    if closest_actor is not None:
                        if getattr(self, "_newvalue_collision_checks", False):
                            # V4 xhard (H1): after fixing the partner and before moving, run a continuous sweep check including distractor bins
                            self._check_swap_sweep_from_actual(i, pair_idx1, closest_actor)
                        setattr(self, f'swap_pair{i+1}_idx2', closest_actor)
                        self._refresh_swap_schedule()
        
        for idx_a, idx_b, start_step, end_step in self.swap_schedule:
            
            if idx_a is None or idx_b is None:
                continue

            swap_flat_two_lane(
                self,
                cube_a=idx_a,
                cube_b=idx_b,
                start_step=start_step,
                end_step=end_step,
                cur_step=timestep,
                lane_offset=0.07,
                smooth=True,
                keep_upright=True,
                other_cube=[b for b in self.spawned_bins if b not in (idx_a, idx_b)],  # Keep all other bins in place to prevent collision during swap
            )


        if self._is_newvalue:
            # V5 xhard (2.7, criterion 4): outer ring swaps in the same window as the inner ring; written outside the inner partner loop, adds no control steps
            run_outer_swaps(self, timestep)
            # Hidden cubes of inner and outer rings each park at an independent point during [64, last_end); at step last_end they land at their bins' final XY (L14)
            park_cubes_onto_bins(self, getattr(self, "cube_bin_pairs", []), group="hidden_cube",
                                 start_step=self.swap_window_start, end_step=self.swap_schedule[-1][3], cur_step=timestep)
            park_cubes_onto_bins(self, getattr(self, "distractor_cube_bin_pairs", []), group="distractor_cube",
                                 start_step=self.swap_window_start, end_step=self.swap_schedule[-1][3], cur_step=timestep)
        else:
            for cube_actor, bin_actor in getattr(self, "cube_bin_pairs", []):
                if cube_actor is None or bin_actor is None:
                    continue

                lift_and_drop_objectA_onto_objectB(
                    self,
                    obj_a=cube_actor,
                    obj_b=bin_actor,
                    start_step=self.swap_window_start,
                    end_step=self.swap_schedule[-1][3],
                    cur_step=timestep,
                )

        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
