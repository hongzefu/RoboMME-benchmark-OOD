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
# V5 L3（仿 VideoPlaceOrder 的 K2 修法）：上一行的 `from .utils import *` 会把同名子模块
# `utils.SceneGenerationError` 盖到名字 `SceneGenerationError` 上（import 自省核实），原三档的
# raise / except 因此是 TypeError（按 H2 原三档保持现状）。xhard 用下面这个别名拿到真正的异常类。
from .utils.SceneGenerationError import SceneGenerationError as _RealSceneGenerationError
from .utils.subgoal_evaluate_func import static_check
from .utils.object_generation import *
from .utils import reset_panda
from .utils.difficulty import is_newvalue_difficulty, normalize_robomme_difficulty
from .utils.episode_spec import SpecRecorder
from .utils.sampling_config import assert_native_decision, fill_missing_newvalue, split_sampling_config
from ..logging_utils import logger


def _scene_gen_error(difficulty):
    """V5 L3：按档选场景生成异常类。

    xhard 返回真正的 ``SceneGenerationError``（可重试的任务性失败）；原三档原样返回本模块里
    被遮蔽的名字 ``SceneGenerationError``（子模块，raise / except 时仍是 TypeError，行为逐字不变）。
    用法：``raise _scene_gen_error(self.difficulty)("说明")``、``except _scene_gen_error(self.difficulty):``；
    只在 xhard 路径上执行的代码直接用 ``_RealSceneGenerationError``。
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


# ── decision／native 两块的原值（newtaskRelease-v3 步 3，映射见方案第二节 2.4）────────
# decision：方块运动速度的候选档位、第几次经过目标时停止。
# native：目标与按钮位置、方块颜色、路线整体旋转、往返段数与时间公式，以及那次
#        「抽了又被覆盖」的 interval 采样（方案要求保留原随机消费，不得删）。
NATIVE_SAMPLING = {
    "parameters": {
        "interval_sample": {
            "sampler": "torch.randint",
            "low": 27,
            "high_exclusive": 33,
            "shape": [1],
            "overridden_to": 30,
            "note": "原代码抽完立刻被常量 30 覆盖；保留这次抽样以免随机流平移（红线 R8）",
        },
        "route_rotation_deg": {
            "sampler": "torch.FloatTensor(1).uniform_",
            "low": -30,
            "high": 30,
        },
        "motion_segments": 5,
        # V6 审查 N15（用户「n15 a」）：motion_segments=5 只是描述值，代码不读它；实际段数在 _initialize_episode 里
        # 原三档恒 5、xhard4 取 max(5, stop_time)（交付 spec 为 6/14/15），随 actions.motion_segments 记入规格
        "motion_segments_note": "描述值，代码不读；实际段数：原三档 5，xhard4 max(5, stop_time)",
        "steps_press_expression": "move_interval * stop_time - move_interval / 2",
        "stop_window_expression": "[move_interval * (stop_time - 1), move_interval * stop_time]",
        "press_lead_steps": "self.interval",
        "route_endpoints": {"start": [0, -0.3], "end": [0, 0.3]},
        "recovery": "StopCube 原本就没有失败抓取注入，只接收入口给定的恢复模式",
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
    """本环境的 ``(decision, native)`` 原值块；外部导出与内部解析共用同一份。"""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


# ── 难度分档（V4 计划 2.6，用户决策 A6）──────────────────────────────────────
# 本环境原本没有难度分档：easy/medium/hard 三档**同值**，都等于原有的全局常量，
# 所以不管传哪档，行为都与改动前逐字一致；只有 xhard 取 V4 新值。
# move_interval_choices：方块单程步数候选（越小越快）；stop_time_range：第几次经过目标时停（半开区间）。
_CONFIG_CURRENT = {
    # 原值三档；randint 等概率抽下标
    "move_interval_choices": [60, 80, 120],
    # 原 randint(2, 6) 即闭区间 [2, 5]
    "stop_time_range": {"low": 2, "high_exclusive": 6},
}
# v8（1001 方案 §1 表 1 / §2.1）：新值档 xhard1～xhard5，每档停止序号一个定数 k = 6／7／8／9／10
# （半开写 low=k, high_exclusive=k+1）；方块速度一律最快档 [60]（同 v7 xhard4）。
# v7 的 xhard4 是 [6, 15] 随机，v8 改为定值 9；xhard1～3、xhard5 为 v8 新增。
_XHARD_STOP_TIME = {"xhard1": 6, "xhard2": 7, "xhard3": 8, "xhard4": 9, "xhard5": 10}


def _config_xhard(stop_time):
    """新值档配置：最快档 [60] + 停止序号定值 ``stop_time``。"""
    return {
        "move_interval_choices": [60],
        "stop_time_range": {"low": stop_time, "high_exclusive": stop_time + 1},
    }


def _native_decision(cls):
    """按方案第二节 2.4 切出 decision 块。

    顶层两键是原三档共用的原值（三档同值，取 ``configs["hard"]``），与 V3 快照逐字相同；
    V4 新值只放在 ``xhard`` 子键下，``assert_native_decision`` 按键名放行。
    """
    hard = cls.configs["hard"]
    return {
        "move_interval_choices": list(hard["move_interval_choices"]),
        "stop_time_range": dict(hard["stop_time_range"]),
        # v8：五个新值档子键（xhard1～5），assert_native_decision 按键名放行
        **{tier: copy.deepcopy(cls.configs[tier]) for tier in _XHARD_STOP_TIME},
    }


def _resolve_sampling_config(cls, override):
    """拆出本实例专属的 decision／native 副本；不抽随机数，必须在 Generator 之前调用。"""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    # 旧快照（v2/v3 导出时还没有 xhard 条目）守卫照旧放行；这里补上源码申报的新值档默认值，
    # 只影响新值档局，原三档不读这些键。fill_missing_newvalue 只在同层已有 xhard4 时补 xhard1/2/3；
    # xhard4 的旧快照兜底保持本环境 V4/V5 原有写法（v8 起写字面 "xhard4"，不再用 NEWVALUE_DIFFICULTIES[-1]）。
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

    # A6：三档同值（深拷贝各一份，防止互相串改）；v8 新值档 xhard1～5 各取定值（xhard4 保持原位置，其余追加在后）
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
        # 必须落在任何随机数调用与 super().__init__() 之前
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "StopCube", {"seed": seed},
                                  difficulty=kwargs.get("difficulty"))
        # 初始化序号从 -1 起，_initialize_episode 每次进来先加一；
        # _load_scene 里的取值点用不带序号的路径，所以这里只作兜底。
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
        # v8（1001 方案 §2.1）：本环境接受 xhard1～xhard5 五个新值档，不再调用 require_xhard4_only

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
            # 这次抽样的结果原本就立刻被覆盖，保留它只为不平移随机流（红线 R8）
            # 这次抽样的结果原本就被覆盖；记进 sampling_trace 以证明它照常发生（红线 R8）
            self._spec.value(
                "actions.sampling_trace.interval_draw",
                torch.randint(interval_cfg["low"], interval_cfg["high_exclusive"], tuple(interval_cfg["shape"]), generator=generator).item(),
            )
            interval = interval_cfg["overridden_to"]
            self.interval = interval


            # 难度真正被消费的唯一位置：xhard 读 decision.xhard 子键，原三档读顶层原值（三档同值）。
            # 两个分支的随机调用次数、顺序、区间形式完全相同，只是区间端点不同（红线 N5）。
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
            # 方块往返段数：原三档在 step 里写死 5 趟（stop_time ≤ 5 恰好够用），这里只作记录不改原路径；
            # xhard 的 stop_time 可达 15，必须按实际停止序号展开，否则第 6 次起的「经过目标」根本不存在。
            # 第 n 次经过目标发生在第 n 段的中点 move_interval*(n-0.5)，所以段数 = max(5, stop_time) 恰好覆盖。
            self.motion_segments = max(5, int(stop_time)) if xhard else 5
            if xhard:
                # 派生量只在 xhard 记进规格（原三档规格文档逐字不变）
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
        # （原三档逐字保持 range(5)；xhard 按 _initialize_episode 算出的实际段数展开，
        #   segment % 2 的起终点交替规则不变）
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
