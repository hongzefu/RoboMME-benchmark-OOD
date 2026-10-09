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
from .utils.object_generation import *
from .utils import reset_panda
from .utils.difficulty import is_newvalue_difficulty, normalize_robomme_difficulty
from .utils.episode_spec import SpecRecorder
from .utils.sampling_config import assert_native_decision, fill_missing_newvalue, split_sampling_config
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


# ── Original values of the decision/native blocks (newtaskRelease-v3 step 3, mapping in plan section 2.4) ────────
# decision: candidate tiers of cube motion speed, and on which pass over the target to stop.
# native: target and button positions, cube color, overall route rotation, round-trip segment count and timing formula, and the
#        "drawn then overridden" interval draw (the plan requires keeping the original random consumption; must not be deleted).
NATIVE_SAMPLING = {
    "parameters": {
        "interval_sample": {
            "sampler": "torch.randint",
            "low": 27,
            "high_exclusive": 33,
            "shape": [1],
            "overridden_to": 30,
            "note": "the original code overwrites this draw with the constant 30 immediately; the draw is kept to avoid shifting the random stream (red line R8)",
        },
        "route_rotation_deg": {
            "sampler": "torch.FloatTensor(1).uniform_",
            "low": -30,
            "high": 30,
        },
        "motion_segments": 5,
        # V6 review N15 (user "n15 a"): motion_segments=5 is only a descriptive value, the code does not read it; the actual segment count, set in _initialize_episode,
        # is always 5 for the original three tiers and max(5, stop_time) for xhard4 (delivered specs are 6/14/15), recorded into the spec with actions.motion_segments
        "motion_segments_note": "descriptive value, not read by code; actual segments: 5 for the original three tiers, max(5, stop_time) for xhard4",
        "steps_press_expression": "move_interval * stop_time - move_interval / 2",
        "stop_window_expression": "[move_interval * (stop_time - 1), move_interval * stop_time]",
        "press_lead_steps": "self.interval",
        "route_endpoints": {"start": [0, -0.3], "end": [0, 0.3]},
        "recovery": "StopCube never had failed-grasp injection; it only accepts the entry-provided recovery mode",
    },
    "positions": {
        "button": {"center_xy": [-0.2, 0], "scale": 1.5, "randomize": True},
        "target": {
            "xy_sampler": "torch.FloatTensor(1).uniform_",
            "low": -0.1,
            "high": 0.1,
            "z": 0.01,
            "euler_deg": [0.0, 90.0, 0.0],
            "radius_factor": 1.8,
            "thickness": 0.01,
        },
        "cube_color": {"sampler": "torch.rand", "shape": [3], "alpha": 1.0},
        "cube_initial_position": [-0.3, -0.3],
    },
}


def native_blocks(cls):
    """Original ``(decision, native)`` blocks of this env; shared by external export and internal parsing."""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


# ── Difficulty tiers (V4 plan 2.6, user decision A6) ──────────────────────────────────────
# This env originally had no difficulty tiers: easy/medium/hard share **identical values**, all equal to the existing global constants,
# so whichever tier is passed, behavior is verbatim identical to pre-change; only xhard takes V4 new values.
# move_interval_choices: candidate one-way step counts of the cube (smaller is faster); stop_time_range: on which pass over the target to stop (half-open interval).
_CONFIG_CURRENT = {
    # Original values for the three tiers; randint draws the index with equal probability
    "move_interval_choices": [60, 80, 120],
    # The original randint(2, 6) is the closed interval [2, 5]
    "stop_time_range": {"low": 2, "high_exclusive": 6},
}
# v8 (1001 plan 1 table 1 / 2.1): new-value tiers xhard1-xhard5, each with one fixed stop index k = 6/7/8/9/10
# (half-open: low=k, high_exclusive=k+1); cube speed always the fastest tier [60] (same as v7 xhard4).
# v7's xhard4 was random in [6, 15]; v8 changes it to the fixed value 9; xhard1-3 and xhard5 are new in v8.
_XHARD_STOP_TIME = {"xhard1": 6, "xhard2": 7, "xhard3": 8, "xhard4": 9, "xhard5": 10}


def _config_xhard(stop_time):
    """New-value tier config: fastest tier [60] + fixed stop index ``stop_time``."""
    return {
        "move_interval_choices": [60],
        "stop_time_range": {"low": stop_time, "high_exclusive": stop_time + 1},
    }


def _native_decision(cls):
    """Slice the decision block per plan section 2.4.

    The two top-level keys are original values shared by the original three tiers (identical values, taken from ``configs["hard"]``), verbatim identical to the V3 snapshot;
    V4 new values live only under the ``xhard`` subkey, admitted by key name by ``assert_native_decision``.
    """
    hard = cls.configs["hard"]
    return {
        "move_interval_choices": list(hard["move_interval_choices"]),
        "stop_time_range": dict(hard["stop_time_range"]),
        # v8: five new-value tier subkeys (xhard1-5), admitted by key name by assert_native_decision
        **{tier: copy.deepcopy(cls.configs[tier]) for tier in _XHARD_STOP_TIME},
    }


def _resolve_sampling_config(cls, override):
    """Split out this instance's private decision/native copies; draws no random numbers, must be called before the Generator."""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    # Old snapshots (exported in v2/v3 without xhard entries) still pass the guard; here we fill in source-declared new-value tier defaults,
    # affecting only new-value tier episodes; original three tiers do not read these keys. fill_missing_newvalue fills xhard1/2/3 only when xhard4 already exists at the same level;
    # the old-snapshot fallback for xhard4 keeps this env's original V4/V5 form (since v8 written as the literal "xhard4", no longer NEWVALUE_DIFFICULTIES[-1]).
    fill_missing_newvalue(decision, decision_default)
    if "xhard4" not in decision:
        decision["xhard4"] = copy.deepcopy(decision_default["xhard4"])
    native["decision"] = decision
    return native


@register_env("StopCube", override=True)
class StopCube(BaseEnv):

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

    # A6: the three tiers share values (one deep copy each, to prevent cross-mutation); v8 new-value tiers xhard1-5 each take fixed values (xhard4 keeps its position, the rest are appended)
    configs = {
        "easy": copy.deepcopy(_CONFIG_CURRENT),
        "medium": copy.deepcopy(_CONFIG_CURRENT),
        "hard": copy.deepcopy(_CONFIG_CURRENT),
        "xhard4": _config_xhard(_XHARD_STOP_TIME["xhard4"]),
        "xhard1": _config_xhard(_XHARD_STOP_TIME["xhard1"]),
        "xhard2": _config_xhard(_XHARD_STOP_TIME["xhard2"]),
        "xhard3": _config_xhard(_XHARD_STOP_TIME["xhard3"]),
        "xhard5": _config_xhard(_XHARD_STOP_TIME["xhard5"]),
    }


    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     sampling_config=None,
                     native_episode_spec=None,
                     **kwargs):
        # Must happen before any RNG call and before super().__init__()
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "StopCube", {"seed": seed},
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
        self.stop=False

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
        # v8 (1001 plan 2.1): this env accepts the five new-value tiers xhard1-xhard5 and no longer calls require_xhard4_only

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



        button_cfg = self._sampling["positions"]["button"]
        button_obb = build_button(
            self,
            center_xy=tuple(button_cfg["center_xy"]),
            scale=button_cfg["scale"],
            generator=generator,
            randomize=button_cfg["randomize"],
            recorder=self._spec,
            spec_path="layout.button_xy",
        )
        #avoid = [button_obb]

        target_cfg = self._sampling["positions"]["target"]
        angles = torch.deg2rad(torch.tensor(target_cfg["euler_deg"], dtype=torch.float32))
        rotate = matrix_to_quaternion(
                    euler_angles_to_matrix(angles, convention="XYZ")
                )
        
        target_x = torch.FloatTensor(1).uniform_(target_cfg["low"], target_cfg["high"], generator=generator).item()
        target_y = torch.FloatTensor(1).uniform_(target_cfg["low"], target_cfg["high"], generator=generator).item()
        target_x, target_y = self._spec.value("layout.target_xy", [target_x, target_y])
        self.target = build_purple_white_target(
                scene=self.scene,
                radius=self.cube_half_size*target_cfg["radius_factor"],
                thickness=target_cfg["thickness"],
                name="target",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(p=[target_x, target_y, target_cfg["z"]], q=rotate),
            )
        color_cfg = self._sampling["positions"]["cube_color"]
        cube_color_rgb = self._spec.value(
            "objects.cube_rgb", torch.rand(*color_cfg["shape"], generator=generator).tolist()
        )
        cube_color = (cube_color_rgb[0], cube_color_rgb[1], cube_color_rgb[2], color_cfg["alpha"])
        self.cube= spawn_fixed_cube(
                self,
                position=[-0.3, -0.3,self.cube_half_size/2],
                half_size=self.cube_half_size,
                color=cube_color,
                name_prefix=f"target_cube",
                yaw=0.0,  # No rotation
            )


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):

            

            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            self.stop = False
            self.stop_timestep = None
            self._task_failed_persistent = False

            # Use generator to generate interval value, floating 5 around 20 (range 15-25)
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            interval_cfg = self._sampling["parameters"]["interval_sample"]
            # The result of this draw was always overwritten immediately; it is kept only to avoid shifting the random stream (red line R8)
            # The result of this draw was always overwritten; recorded in sampling_trace to prove it still happens (red line R8)
            self._spec.value(
                "actions.sampling_trace.interval_draw",
                torch.randint(interval_cfg["low"], interval_cfg["high_exclusive"], tuple(interval_cfg["shape"]), generator=generator).item(),
            )
            interval = interval_cfg["overridden_to"]
            self.interval = interval


            # The only place difficulty is actually consumed: xhard reads the decision.xhard subkey, original three tiers read top-level original values (identical across tiers).
            # Both branches have exactly the same number, order and interval form of random calls; only the interval endpoints differ (red line N5).
            xhard = is_newvalue_difficulty(self.difficulty)
            decision_cfg = self._sampling["decision"][self.difficulty] if xhard else self._sampling["decision"]
            key_prefix = f"{self.difficulty}." if xhard else ""

            move_interval_list = list(decision_cfg["move_interval_choices"])
            idx = self._spec.value(
                "actions.move_interval_idx",
                torch.randint(0, len(move_interval_list), (1,), generator=generator).item(),
                decision_key=f"{key_prefix}move_interval_choices",
            )
            self.move_interval = move_interval_list[idx]

            stop_cfg = decision_cfg["stop_time_range"]
            stop_time=self._spec.value(
                "actions.stop_time",
                torch.randint(stop_cfg["low"], stop_cfg["high_exclusive"], (1,), generator=generator).item(),
                decision_key=f"{key_prefix}stop_time_range",
            )

            self.steps_press=self.move_interval*(stop_time)-self.move_interval/2
            self.stop_time_range = (
                self.move_interval * (stop_time - 1),
                self.move_interval * (stop_time ),
            )
            self.stop_time=stop_time
            # Cube round-trip segment count: the original three tiers hardcode 5 trips in step (enough since stop_time <= 5); recorded here only, original path unchanged;
            # xhard's stop_time can reach 15, so segments must be expanded by the actual stop index, otherwise passes over the target from the 6th onward would not exist.
            # The n-th pass over the target occurs at the midpoint move_interval*(n-0.5) of segment n, so segments = max(5, stop_time) covers it exactly.
            self.motion_segments = max(5, int(stop_time)) if xhard else 5
            if xhard:
                # Derived quantities are recorded into the spec only for xhard (original three tiers' spec documents verbatim unchanged)
                self._spec.record("actions.move_interval", int(self.move_interval))
                self._spec.record("actions.motion_segments", int(self.motion_segments))
                self._spec.record("actions.steps_press", float(self.steps_press))
                self._spec.record("actions.stop_window", [float(v) for v in self.stop_time_range])
            # Get target xy coordinates (already randomized in _load_scene)
            target_pose = self.target.pose
            if isinstance(target_pose.p, torch.Tensor):
                target_x = target_pose.p[0, 0].item()
                target_y = target_pose.p[0, 1].item()
            else:
                target_x = target_pose.p[0]
                target_y = target_pose.p[1]
            target_center = np.array([target_x, target_y])

            # Generate random rotation angle (-30 to +30 degrees)
            rotation_cfg = self._sampling["parameters"]["route_rotation_deg"]
            rotation_angle = self._spec.value(
                "actions.rotation_deg",
                torch.FloatTensor(1).uniform_(rotation_cfg["low"], rotation_cfg["high"], generator=generator).item(),
            )
            rotation_rad = np.deg2rad(rotation_angle)

            # Define original start and end coordinates (around origin (0,0))
            original_start = np.array([0, -0.3])
            original_end = np.array([0, 0.3])

            # Rotation matrix
            cos_theta = np.cos(rotation_rad)
            sin_theta = np.sin(rotation_rad)
            rotation_matrix = np.array([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ])

            # Apply rotation (around origin), then add target xy coordinates
            self.start_pos_xy = rotation_matrix @ original_start + target_center
            self.end_pos_xy = rotation_matrix @ original_end + target_center

            # Set cube initial position to rotated start point
            self.cube.set_pose(sapien.Pose(p=[self.start_pos_xy[0], self.start_pos_xy[1], self.cube_half_size/2]))

            # Generate task list to move to each button sequentially
            tasks = []

            tasks.append(             {
                                "func": lambda: button_hover(self,button=self.button),
                                "name": "move to the top of the button to prepare",
                                "subgoal_segment": "move to the top of the button at <> to prepare",
                                "choice_label": "move to the top of the button to prepare",
                                "demonstration": False,
                                "failure_func": None,
                                "specialflag":"swap",
                                "solve": lambda env, planner: [solve_button_ready(env, planner, obj=self.button)],
                                "segment":self.cap_link 
                                },)

            final_abs_timestep = self.steps_press - interval
            static_checkpoints = list(range(100, int(final_abs_timestep), 100))
            if not static_checkpoints or static_checkpoints[-1] != final_abs_timestep:
                static_checkpoints.append(final_abs_timestep)

            for target_timestep in static_checkpoints:
                tasks.append({
                                    "func": lambda target_timestep=target_timestep: before_absTimestep(self, absTimestep=target_timestep),
                                    "name": "remain static",
                                    "subgoal_segment": "remain static",
                                    "choice_label": "remain static",
                                    "demonstration": False,
                                    "failure_func": None,
                                    "specialflag":"swap",
                                    "solve": lambda env, planner, target_timestep=target_timestep: solve_hold_obj_absTimestep(env, planner,absTimestep=target_timestep),
                                    },)
            tasks.append({
                        "func": lambda: is_obj_stopped_onto(self, obj=self.cube, target=self.target, stop=self.stop),
                        "name": "press the button to stop the cube on the target",
                        "subgoal_segment": "press the button to stop the cube on the target at <>",
                        "choice_label": "press button to stop the cube",
                        "demonstration": False,
                        "failure_func": lambda: None,
                        "solve": lambda env, planner: [solve_button(env, planner, obj=self.button,without_hold=True)
                                                       ],

                        "segment":self.target 
                        },
            )


            # Store task list for RecordWrapper use
            self.task_list = tasks

    def _get_obs_extra(self, info: Dict):
        return dict()




    def evaluate(self,solve_complete_eval=False):
        if not hasattr(self, "_task_failed_persistent"):
            self._task_failed_persistent = False
        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([True]) if self._task_failed_persistent else torch.tensor([False])




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
        task_failed = task_failed or self._task_failed_persistent# Ensure overshoot is covered

###################################################
        if all_tasks_completed:
            correct=correct_timestep(self,time_range=self.stop_time_range,stop_timestep=self.stop_timestep)# identify which pass pressed, stopped on, count error
            if correct!= True:
                task_failed=True

        current_stop = self.stop or is_button_pressed(self, obj=self.button)# Extra check for timing issue!
        press_before = (not is_obj_stopped_onto(self, obj=self.cube, target=self.target, stop=current_stop)) and is_button_pressed(self, obj=self.button)
        #print(f"press_before",press_before)
        # Manually set to fail if not stopped on target
        if press_before== True:
            #import pdb; pdb.set_trace()
            task_failed=True
##################################################
        # Fail immediately if exceeded without press
        current_step = int(getattr(self, "elapsed_steps", 0))
        if current_step > self.move_interval * self.stop_time:
            if not all_tasks_completed:
                #The issue is that the environment continues running after the task is successfully completed, 
                # eventually triggering a timeout check that incorrectly marks the episode as a failure.
                task_failed = True


#################################################

        # If task failed, mark as failed immediately
        if task_failed:
            self._task_failed_persistent = True
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
        

        if is_button_pressed(self, obj=self.button):# Chronological issue, MUST be placed before super!!!
            self.stop=True

            
        obs, reward, terminated, truncated, info = super().step(action)


        # Use the rotated xy coordinates calculated in _initialize_episode
        start_pos = [self.start_pos_xy[0], self.start_pos_xy[1], self.cube_half_size / 2]
        end_pos = [self.end_pos_xy[0], self.end_pos_xy[1], self.cube_half_size / 2]

        # Alternate between the two waypoints so the cube makes five passes
        # (original three tiers keep range(5) verbatim; xhard expands by the actual segment count computed in _initialize_episode,
        #   the alternating start/end rule of segment % 2 is unchanged)
        if is_newvalue_difficulty(getattr(self, "difficulty", None)):
            segments = range(self.motion_segments)
        else:
            segments = range(5)
        for segment in segments:
            move_straight_line(
                self,
                cube=self.cube,
                start_step=self.move_interval * segment,
                end_step=self.move_interval * (segment + 1),
                cur_step=int(self.elapsed_steps),
                start_pos=start_pos if segment % 2 == 0 else end_pos,
                end_pos=end_pos if segment % 2 == 0 else start_pos,
                stop=self.stop,
            )
        return obs, reward, terminated, truncated, info
