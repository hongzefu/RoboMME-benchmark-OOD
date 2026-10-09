import copy
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import math
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

# NOTE: keep wildcard import for legacy helpers that the environment relies on.
from .utils import *
# V5 L3 (following VideoPlaceOrder's K2 fix): the `from .utils import *` line above lets the same-named submodule
# `utils.SceneGenerationError` shadow the name `SceneGenerationError` (confirmed by import introspection), so in the original three tiers
# raise / except become TypeError (original three tiers kept as is per H2). xhard uses the alias below to get the real exception class.
from .utils.SceneGenerationError import SceneGenerationError as _RealSceneGenerationError
from .utils.subgoal_evaluate_func import *
from .utils.object_generation import *
from .utils import reset_panda
from .utils.difficulty import NEWVALUE_DIFFICULTIES, is_newvalue_difficulty, normalize_robomme_difficulty
from .utils.episode_spec import SpecRecorder
from .utils.episode_spec import EpisodeSpecError as _EpisodeSpecError
from .utils.sampling_config import assert_native_decision, fill_missing_newvalue, split_sampling_config
from ..logging_utils import logger


def _scene_gen_error(difficulty):
    """V5 L3: select the scene-generation exception class by tier.

    xhard returns the real ``SceneGenerationError`` (retryable task failure); original three tiers return this module's
    shadowed name ``SceneGenerationError`` as is (a submodule, so raise / except still give TypeError; behavior verbatim unchanged).
    Usage: ``raise _scene_gen_error(self.difficulty)("message")``, ``except _scene_gen_error(self.difficulty):``;
    code executed only on the xhard path uses ``_RealSceneGenerationError`` directly.
    """
    # V6 criterion 11: the new-value family (xhard1/2/3/xhard) all use the real exception class; original three tiers unchanged
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


# ── Original values of the decision/native blocks (newtaskRelease-v3 step 3, mapping in plan section 2.15) ────────
NATIVE_SAMPLING = {
    "parameters": {
        "path_selection": {
            "start_end_sampler": "torch.randperm(num_targets)[:2]",
            "search": "random DFS allowing diagonals (find_path_0_to_8, diagonals=True)",
            "max_attempts": 1000,
            "exhausted_rule": "after exhaustion use the last path found; no extra draw",
        },
        "path_binding": "demonstration and execution reuse the same path; the first target keeps NO RECORD",
        "motion_template": "for each target solve_swingonto does two screw motions and close_gripper",
        "recovery": "keep the entry-provided fail recover mode and the original generator",
    },
    "positions": {
        "grid_center": [-0.1, 0],
        "grid_spacing": 0.1,
        "node_position_expression": "center + (index - (n-1)/2) * spacing",
    },
}


# ── V5 xhard-specific decision (plan 2.10, L35/L36) ──────────────────────────────
# Path search budget: original three tiers still read native's ``path_selection.max_attempts`` (1000); xhard reads this,
# frozen into the spec header with decision. (V6 review N13 note: xhard4 node count range is [21,25], not V5's fixed 25;
# "25-node hit rate 1.000 at 20000 attempts" is historical data from V5 planning probe P3. DFS stops at the first path within range,
# so the node count distribution hugs the lower bound -- xhard4 10 candidates measured {21:6, 22:2, 23:2}; user ruled to change only the comment, not the distribution.)
XHARD_DECISION = {
    "path_search_max_attempts": 20000,
}

# V6 (plan 2.11): new-value tier table. xhard1/2/3 have exactly the same key structure as xhard; the plan gives no difference for new tiers
# other than search budget -- the budget always follows xhard's 20000, and exhaustion likewise raises a real SceneGenerationError.
NEWVALUE_DECISION = {
    "xhard1": {"path_search_max_attempts": 20000},
    "xhard2": {"path_search_max_attempts": 20000},
    "xhard3": {"path_search_max_attempts": 20000},
    "xhard4": XHARD_DECISION,
}


def native_blocks(cls):
    """Original ``(decision, native)`` blocks of this env; shared by external export and internal parsing."""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


def _native_decision(cls):
    """Slice the decision block per plan section 2.15 (equals the original in the original-value stage)."""
    return {
        # Target demonstration video duration and how to adjust it: not enabled this round (None = decided by the original path and solver motion).
        "demo_duration_seconds_range": None,
        "demonstration_duration_policy": "native",
        # Grid size and path node count range are native rules per the field table; only the original values are recorded here for verification, not as new parameters.
        "grid": {difficulty: cfg["grid"] for difficulty, cfg in cls.configs.items()},
        "path_length_range": {difficulty: list(cfg["length"]) for difficulty, cfg in cls.configs.items()},
        # V5 xhard-specific (plan 2.10): key named xhard; the guard only lets this subtree take new values; the part visible to the original three tiers is unchanged.
        "xhard4": copy.deepcopy(NEWVALUE_DECISION["xhard4"]),
        # V6: append three same-structure subtrees xhard1/2/3 (xhard stays first with original values)
        # v8: xhard4 written as a literal (no longer NEWVALUE_DIFFICULTIES[-1] to denote the hardest tier)
        **{d: copy.deepcopy(NEWVALUE_DECISION[d]) for d in NEWVALUE_DIFFICULTIES if d != "xhard4"},
    }


def _resolve_sampling_config(cls, override):
    """Split out this instance's private decision/native copies; draws no random numbers, must be called before the Generator."""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    # V6: V5 snapshots (with a top-level xhard subtree) missing xhard1/2/3 are filled from source; earlier snapshots are not filled,
    # keeping V5's criterion 13 "V4 and earlier snapshots error out directly on new-value tiers, never silently take new values from source".
    if "xhard4" in decision:
        fill_missing_newvalue(decision, decision_default)
    native["decision"] = decision
    return native


@register_env("PatternLock", override=True)
class PatternLock(BaseEnv):

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
        "grid":5,
        "length":[4,8]
    }

    config_easy = {
        "grid":3,
         "length":[2,4]
    }

    config_medium = {
        "grid":4,
        "length":[3,5]
    }

    # V4 xhard (derived from hard, B8): layout and search method unchanged, only node count raised to [20,24].
    # V5 (plan 2.10, L35/L36) once fixed the node count at 25 (the maximum for a non-revisiting 5x5 path).
    # V6 (0925 plan section 3): xhard4 node count changed to closed interval [21,25] (xhard1/2/3 are [9,12]/[13,16]/[17,20]),
    # search budget 20000 (decision.<tier>.path_search_max_attempts) and raising a real SceneGenerationError on exhaustion are kept.
    # WARNING: DFS breaks at the first path within the interval, so actual node counts hug the lower bound (V6 review N13: xhard4 10 candidates {21:6, 22:2, 23:2});
    # the user ruled to update only the comment, not the sampling distribution.
    # V7 fixed values: node counts 12/15/18/21 (user 2026-09-29 "just change to 12/15/18/21"; fixed length 24 was almost never found in v6 measurements)
    config_xhard4 = {
        "grid": 5,
        "length": [21, 21]
    }

    # V6 (plan 2.11): three tiers inserted between hard and xhard; layout, search method, budget and exhaustion error all follow xhard, only node count changes
    # v8 (1001 plan 1 table 1 / 2.1): xhard1-3 change from fixed 12/15/18 to intervals [9,12]/[13,15]/[16,18];
    # sampling code unchanged (rejection sampling takes the first path within the interval), so node counts are not guaranteed uniform within the interval; xhard4 still 21 (not delivered).
    config_xhard1 = {
        "grid": 5,
        "length": [9, 12]
    }
    config_xhard2 = {
        "grid": 5,
        "length": [13, 15]
    }
    config_xhard3 = {
        "grid": 5,
        "length": [16, 18]
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


    def __init__(self, *args, robot_uids="panda_stick", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     sampling_config=None,
                     native_episode_spec=None,
                     **kwargs):
        # Must happen before any RNG call and before super().__init__()
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "PatternLock", {"seed": seed},
                                  difficulty=kwargs.get("difficulty"))
        # Initialization index starts at -1; _initialize_episode increments it on each entry;
        # value points in _load_scene use index-free paths, so this is only a fallback.
        self._native_init_index = -1
        self.achieved_list=[]
        self.match=False
        self.after_demo=False

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
        #self.difficulty = "hard"
               # Use seed to determine number of repetitions (1-5) arbitrarily
        generator = torch.Generator()
        generator.manual_seed(seed)


        self.highlight_starts = {}  # Use dictionary to store highlight start time for each button
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

        # Generate 3x3 grid of buttons
        layout_cfg = self._sampling["positions"]
        decision_cfg = self._sampling["decision"]
        grid_center = list(layout_cfg["grid_center"])  # Grid center position
        grid_spacing = layout_cfg["grid_spacing"]  # Spacing between buttons

        self.buttons_grid = []
        self.button_joints_grid = []
        avoid = []
        button_index = 0

        
        num_rows, num_cols = 5, 8
        num_rows, num_cols = decision_cfg["grid"][self.difficulty],decision_cfg["grid"][self.difficulty]
        row_center = (num_rows - 1) / 2
        col_center = (num_cols - 1) / 2



        for row in range(num_rows):  # 3 rows (x direction)
            for col in range(num_cols):  # 5 columns (y direction)
                x_pos = grid_center[0] + (row - row_center) * grid_spacing
                y_pos = grid_center[1] + (col - col_center) * grid_spacing



                target_name = f"target_{button_index}"

                # Create rotation quaternion for vertical target
                angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))
                rotate = matrix_to_quaternion(
                    euler_angles_to_matrix(angles, convention="XYZ")
                )

                # Build purple and white target
                target = build_gray_white_target(
                    scene=self.scene,
                    radius=0.02,
                    thickness=0.01,
                    name=target_name,
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=sapien.Pose(p=[x_pos, y_pos, 0.01], q=rotate),
                )

                self.buttons_grid.append(target)
                # Note: purple_white_target doesn't have joints, so we append None
                self.button_joints_grid.append(None)
                logger.debug(f"Generated target {button_index} at position ({x_pos:.3f}, {y_pos:.3f})")
                button_index += 1

        self.targets_grid = self.buttons_grid

                # Generate task list to move to each button sequentially
        tasks = []

        # start_end_set = [
        #     [0, 1, 8, 9],
        #     [6, 7, 14, 15],
        #     [24, 25, 32, 33],
        #     [30, 31, 38, 39]
        # ]

        # # Randomly select 2 different sets from start_end_set
        # set_indices = torch.randperm(len(start_end_set), generator=generator)[:2].tolist()
        # start_set = start_end_set[set_indices[0]]
        # end_set = start_end_set[set_indices[1]]

        # # Randomly select one node from each set
        # start_idx = torch.randint(0, len(start_set), (1,), generator=generator).item()
        # end_idx = torch.randint(0, len(end_set), (1,), generator=generator).item()

        # num_targets = len(self.targets_grid)
        # node_choices = torch.randperm(num_targets, generator=generator)[:2]
        # start_node, end_node = node_choices.tolist()


        # path_nodes, _, _, _ = find_path_0_to_8(
        #     start=start_node,
        #     target=end_node,
        #     R=num_rows,
        #     C=num_cols,
        #     diagonals=True,
        #     generator=generator,
        # )
        # self.selected_buttons = [self.buttons_grid[i] for i in path_nodes]

        num_targets = len(self.targets_grid)
        max_attempts = self._sampling["parameters"]["path_selection"]["max_attempts"]  # Safety limit
        if is_newvalue_difficulty(self.difficulty):
            # V5 (plan 2.10): xhard's search budget comes from decision (frozen into the spec header); original three tiers still use the 1000 on the line above
            max_attempts = self._xhard_decision("path_search_max_attempts")

        self._spec.identity.setdefault("difficulty", getattr(self, "difficulty", None))
        for attempt in range(max_attempts):
            # Every attempt draws as usual; the accepted one is fixed by the spec (failed attempts still consume random numbers)
            node_choices = torch.randperm(num_targets, generator=generator)[:2]
            start_node, end_node = node_choices.tolist()
            
            path_nodes, _, _, _ = find_path_0_to_8(
                start=start_node,
                target=end_node,
                R=num_rows,
                C=num_cols,
                diagonals=True,
                generator=generator,
            )

            length_range = decision_cfg["path_length_range"][self.difficulty]
            if length_range[0] <= len(path_nodes) <= length_range[1]:
                break
        else:
            if is_newvalue_difficulty(self.difficulty):
                # V5 (plan 2.10 / L36 / L3): when xhard search is exhausted it no longer silently reuses the last wrong-length path (K1 root cause),
                # but raises a real SceneGenerationError (retryable task failure); original three tiers still take the silent fallback on the next line.
                raise _RealSceneGenerationError(
                    f"PatternLock {self.difficulty}: within {max_attempts} searches no path with node count in "
                    f"{decision_cfg['path_length_range'][self.difficulty]} was found"
                )
            # If we couldn't find a path < 5 after max_attempts, use the last one
            logger.debug(f"Warning: Could not find path after {max_attempts} attempts")

        # Every attempt in the search loop draws random numbers as usual; only the finally adopted path is frozen here
        path_nodes = self._spec.value("actions.path_nodes", list(path_nodes),
                                      decision_key=f"path_length_range.{self.difficulty}")
        self._spec.record("actions.path_attempts", attempt + 1)
        if is_newvalue_difficulty(self.difficulty):
            # V5 (in the spirit of N17): when replaying a frozen spec value() returns the frozen value without re-checking, so node count and adjacency are re-checked here
            self._check_xhard_path(path_nodes, num_rows, num_cols,
                                   decision_cfg["path_length_range"][self.difficulty])
        self.selected_buttons = [self.buttons_grid[i] for i in path_nodes]
        current_target=self.selected_buttons[0]
        tasks.append({
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t),
            "name":  "NO RECORD",
            "subgoal_segment":f"NO RECORD",
            "demonstration": True,
            "failure_func":  lambda expected=current_target: self._wrong_button_touch(expected_button=expected),
            "solve": lambda env, planner, t=current_target: solve_swingonto(env, planner, target=t,record_swing_qpos=True),
        })  
        for i, current_target in enumerate(self.selected_buttons[1:]):
            last_target = self.selected_buttons[i]
            tasks.append({
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t),
            "name": f"move {direction(current_target, last_target)}",
            "subgoal_segment":f"move {direction(current_target, last_target)}",
            "choice_label": f"move {direction(current_target, last_target)}",
            "demonstration": True,
            "failure_func":  lambda expected=current_target, last=last_target: self._wrong_button_touch(expected_button=expected, last_button=last),
            "solve": lambda env, planner, t=current_target: solve_swingonto(env, planner, target=t),
            #"segment":current_target,
        })  
        
        tasks.append({
                    "func": lambda:reset_check(self,gripper="stick"),
                    "name": "NO RECORD",
                    "subgoal_segment":f"NO RECORD",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [solve_strong_reset(env,planner,gripper="stick")],
                    },)
        
      
        self.selected_buttons = [self.buttons_grid[i] for i in path_nodes]
        current_target=self.selected_buttons[0]
        tasks.append({
            "func":   lambda:reset_check(self,gripper="stick",target_qpos=self.swing_qpos),
            "name":  "NO RECORD",
            "subgoal_segment":f"NO RECORD",
            "demonstration": True,
            "failure_func":  None,
            "solve": lambda env, planner, t=current_target: [solve_strong_reset(env, planner,gripper="stick",action=self.swing_qpos)],
        })  
        for i, current_target in enumerate(self.selected_buttons[1:]):
            last_target = self.selected_buttons[i]
            tasks.append({
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t),
            "name": f"move {direction(current_target, last_target)}",
            "subgoal_segment":f"move {direction(current_target, last_target)}",
            "choice_label": f"move {direction(current_target, last_target)}",
            "demonstration": False,
            "failure_func": lambda expected=current_target, last=last_target: self._wrong_button_touch(expected_button=expected, last_button=last),
            "solve": lambda env, planner, t=current_target: solve_swingonto(env, planner, target=t),
            #"segment":current_target,
        })  

        # Store task list for RecordWrapper use
        self.task_list = tasks




    def _xhard_decision(self, key):
        """V5: read xhard-specific decision keys (called only on the xhard path).

        A missing key means the sampling_config comes from a V4 or earlier snapshot (no ``decision.xhard``);
        V4 is obsolete (criterion 13), so raise directly instead of falling back to the original three tiers' values.
        """
        # V6: fetch the subtree by this episode's tier (the four new-value tiers share structure)
        xhard_cfg = self._sampling["decision"].get(self.difficulty)
        if not isinstance(xhard_cfg, dict) or key not in xhard_cfg:
            raise ValueError(
                f"PatternLock {self.difficulty}: sampling_config.decision is missing {self.difficulty}.{key}"
                " (V4 and earlier snapshots are not usable with V5 code)"
            )
        return xhard_cfg[key]

    @staticmethod
    def _check_xhard_path(path_nodes, num_rows, num_cols, length_range):
        """V5 (in the spirit of N17): re-check the xhard path -- node count within range, no revisits, consecutive points 8-adjacent.

        In export mode the search loop already guarantees these, so this is a self-check; when replaying a frozen spec ``SpecRecorder.value``
        returns the frozen value without re-checking, so this blocks specs that were corrupted or come from old rules. Raises ``EpisodeSpecError`` on violation.
        """
        nodes = [int(v) for v in path_nodes]
        low, high = int(length_range[0]), int(length_range[1])
        if not low <= len(nodes) <= high:
            raise _EpisodeSpecError(f"PatternLock xhard: path node count {len(nodes)} not in [{low}, {high}]")
        if len(set(nodes)) != len(nodes):
            raise _EpisodeSpecError("PatternLock xhard: path revisits a node")
        if any(not 0 <= v < num_rows * num_cols for v in nodes):
            raise _EpisodeSpecError("PatternLock xhard: path node outside the grid")
        for a, b in zip(nodes, nodes[1:]):
            dr = abs(a // num_cols - b // num_cols)
            dc = abs(a % num_cols - b % num_cols)
            if max(dr, dc) != 1:
                raise _EpisodeSpecError(f"PatternLock xhard: path {a} -> {b} is not 8-adjacent")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos",gripper="stick")
            self.agent.reset(qpos)



    def _get_obs_extra(self, info: Dict):
        return dict()

    def evaluate(self,solve_complete_eval=False):
        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([False])

        for idx, button in enumerate(self.buttons_grid):
           
            if is_obj_swing_onto(self, obj=self.agent.tcp, target=button):# Only execute when gripper is closed
                # Update start time to refresh highlight effect when repeatedly triggered
                self.highlight_starts[idx] =int(self.elapsed_steps[0].item())
                # Only record when not recording
                if self.after_demo==True:
                    if not self.achieved_list or self.achieved_list[-1] is not button:
                        self.achieved_list.append(button) # highlight=reach record, not necessarily target

        def _to_label(item):
            name = getattr(item, "name", None)
            return name if name is not None else str(item)
                # Backtrack from end of completed button sequence to check if it matches current target sequence exactly
        achieved_labels = [_to_label(item) for item in self.achieved_list]
        selected_labels = [_to_label(item) for item in getattr(self, "selected_buttons", [])]
        remaining = [label for label in selected_labels if label not in achieved_labels]
        if selected_labels:
            recent_achieved = achieved_labels[-len(selected_labels):]
            if len(recent_achieved) == len(selected_labels) and recent_achieved == selected_labels:
                logger.debug("match success")
                self.match=True
        # print(f"achieved_list: {achieved_labels}")
        # print(f"selected_buttons: {selected_labels}")
        # print(f"remaining_targets: {len(remaining)}")





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

        if all_tasks_completed and self.match==False:# Manually set to fail if string match fails
            logger.debug("match failure")
            task_failed=True

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
        # tcp_to_obj_dist = torch.linalg.norm(
        #     self.agent.tcp_pose.p - self.agent.tcp_pose.p, axis=1
        # )
        # reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        # reward = reaching_reward*0
        reward=torch.tensor([0])
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    def _wrong_button_touch(self, expected_button, last_button=None):
        # If button touched by is_obj_swing_onto is neither current expected target nor previous button (debounce), check as error
        for button in self.buttons_grid:
            if button is expected_button:
                continue
            if last_button is not None and button is last_button:
                continue
            if is_obj_swing_onto(self, obj=self.agent.tcp, target=button):
                return True
        return False


#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        obs, reward, terminated, truncated, info = super().step(action)
        # def _to_label(item):
        #     name = getattr(item, "name", None)
        #     return name if name is not None else str(item)

        # # Backtrack from end of completed button sequence to check if it matches current target sequence exactly
        # achieved_labels = [_to_label(item) for item in self.achieved_list]
        # selected_labels = [_to_label(item) for item in getattr(self, "selected_buttons", [])]
        # remaining = [label for label in selected_labels if label not in achieved_labels]
        # if selected_labels:
        #     recent_achieved = achieved_labels[-len(selected_labels):]
        #     if len(recent_achieved) == len(selected_labels) and recent_achieved == selected_labels:
        #         print("match success")
        #         self.match=True
        # print(f"achieved_list: {achieved_labels}")
        # print(f"selected_buttons: {selected_labels}")
        # print(f"remaining_targets: {len(remaining)}")

        

        # Check if each button is swum onto, and record highlight start time
        cur_step = int(self.elapsed_steps[0].item())
        highlight_position(
            self,
            self.agent.tcp.pose.p,
            start_step=cur_step,
            end_step=cur_step + 40,
            cur_step=cur_step,
            disk_radius=0.005,
        )
        # for idx, button in enumerate(self.buttons_grid):
        #     if is_obj_swing_onto(self, obj=self.agent.tcp, target=button):
        #         # Update start time to refresh highlight effect when repeatedly triggered
        #         self.highlight_starts[idx] = cur_step
        #         # Only record when not recording
        #         if self.after_demo==True:
        #             if not self.achieved_list or self.achieved_list[-1] is not button:
        #                 self.achieved_list.append(button) # highlight=reach record, not necessarily target

        # Apply highlight effect to each triggered button
        for idx, button in enumerate(self.buttons_grid):
            start_step = self.highlight_starts.get(idx)
            if start_step is not None:
                highlight_obj(
                    self,
                    button,
                    start_step=start_step,
                    end_step=start_step + 40,
                    cur_step=cur_step,
                    disk_radius=0.02*1.002,
                    disk_half_length=0.01*2*1.002,
                    highlight_color=[1.0, 0.0, 0.0, 1.0],
                    use_target_style=True,
                )


        return obs, reward, terminated, truncated, info
