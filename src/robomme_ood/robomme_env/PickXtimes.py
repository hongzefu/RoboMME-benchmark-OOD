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


# ── decision／native 两块的原值（newtaskRelease-v3 步 3，映射见方案第二节 2.2）────────
# decision：颜色数、重复抓放次数范围、目标方块与放置圆盘各自的位置采样区域、额外干扰物。
# native：颜色排列与目标选择、按钮位置、方块／圆盘的几何与拒绝条件、恢复与动作展开规则。
# 数值全部取自改动前写在调用点的字面量，原值阶段两块都等于原值（红线 R7）。
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
            "note": "前置颜色抽样被后面的目标方块选择覆盖，保留原抽样次数与顺序",
        },
        "recovery": "沿用入口给定的 fail recover 模式与 inject_fail_grasp 原抽法",
        "task_expansion": "反复抓同一 target_cube 放到 target，末尾按按钮；非目标取候选补集",
    },
    "positions": {
        "button": {
            "center_xy": [-0.2, 0],
            "scale": 1.5,
            "randomize_range_note": "原调用点未传 randomize_range，保持 build_button 形参默认值",
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


# ── xhard 专属 decision（V4 计划 2.4 / 2.21：C1 / G1 / A5 / B2；V5 计划 2.13：L43～L46）──────
# * target_cube_position_policy：目标候选方块（三个有色方块）的采样区域。V5 L43 (c) 取消边角偏置
#   （删 corner_bias 键，推翻 V4 J5），与干扰方块一样在区域内均匀抽；V5 L46 半宽 0.2 → 0.25
#   （P1 探针：10 cm 三块团 33.5% → 10.2%，演示 13/13）。
# * goal_position_policy：放置圆盘独立一套区域参数（C1：圆盘可以留在中间，值沿用原区域，V5 不变）。
# * distractor：四个干扰方块，黄／青／品红／第 4 色各一（BLOCK_DISTRACTOR_COLORS，V7），在方块区域内均匀放置；V5 L46 半宽同为 0.25。
# * min_center_dist_m：V5 L44，6 块（3 有色 + 3 干扰）两两中心距下限（米），留一个方块宽的缝。
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
    """V6（计划 2.8）：新值族某档的 decision 子树——键结构与 ``XHARD_DECISION`` 完全相同，
    只把干扰方块颜色截成 ``BLOCK_DISTRACTOR_COLORS`` 前 k 个；区域、中心距等其余字段沿用 xhard。"""
    tree = copy.deepcopy(XHARD_DECISION)
    tree["distractor"]["colors"] = [entry["name"] for entry in BLOCK_DISTRACTOR_COLORS[:n_distractors]]
    return tree


# V6 新值族档位表：干扰块数 xhard1=1、xhard2=2、xhard3=3、xhard4=3。
NEWVALUE_DECISION = {
    "xhard1": _newvalue_decision(1),
    "xhard2": _newvalue_decision(2),
    "xhard3": _newvalue_decision(3),
    "xhard4": _newvalue_decision(4),
}

# V5 L45：xhard 方块（有色 + 干扰）每块的拒绝采样预算（原三档沿用 spawn_random_cube 默认 256，不受影响）。
# 计划 2.13 估：加 8 cm 中心距后 256 次约 3.5% 的局放不下、1024 次约 1%；S3f 离线 3000 局实测 1024 次 0 失败。
XHARD_CUBE_MAX_TRIALS = 1024


def _disk_avoid_obb(target, clearance):
    """把放置圆盘换算成方块拒绝采样可用的预制 OBB ``(中心, 轴, 半边长)``。

    圆盘是 ``add_collision=False`` 的纯视觉 actor，``get_actor_obb`` 取不到网格，直接放进
    ``avoid`` 会被 ``spawn_random_cube`` 静默忽略（2026-09-22 实测）。xhard 先放圆盘后放方块（G1），
    必须显式给出它的外接正方形：半边长 = 圆盘半径 + 圆盘间距 − 方块自带间距，
    使轴向判据与原「圆盘避让方块」的圆–盒距离判据一致、对角方向更保守。
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
    """本环境的 ``(decision, native)`` 原值块；外部导出与内部解析共用同一份。"""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


def _native_decision(cls):
    """按方案第二节 2.2 切出 decision 块（原值阶段等于原值）。"""
    return {
        "color": {difficulty: cfg["color"] for difficulty, cfg in cls.configs.items()},
        "number_range": {
            difficulty: [cfg["number_min"], cfg["number_max"]]
            for difficulty, cfg in cls.configs.items()
        },
        # 目标方块与放置圆盘各自的位置采样区域；原值即两者同区域。
        "target_cube_position_policy": {"region_center": [-0.1, 0], "region_half_size": 0.2},
        "goal_position_policy": {"region_center": [-0.1, 0], "region_half_size": 0.2},
        # 第二节的「增加其他颜色 distractor」原三档不启用（原值保持 None）。
        "distractor": None,
        # V4 xhard 专属（计划 2.4）：键名为 xhard，守卫只放行这一子树取新值，原三档可见部分不变。
        # V6（计划 2.8）：xhard 之后追加 xhard1/2/3 三棵同结构子树（新值族，守卫同样放行）。
        "xhard4": copy.deepcopy(XHARD_DECISION),
        **{tier: copy.deepcopy(NEWVALUE_DECISION[tier]) for tier in ("xhard1", "xhard2", "xhard3")},
    }


def _resolve_sampling_config(cls, override):
    """拆出本实例专属的 decision／native 副本；不抽随机数，必须在 Generator 之前调用。"""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    # V6：旧快照（如 V5 快照）缺 xhard1/2/3 子树时从源码申报补齐
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

    # v8 定值（1001 方案 §1 表 1 / §2.1）：每档一个定数，抓放次数 6/7/8/9、干扰块 1/2/3/4（BLOCK_DISTRACTOR_COLORS）。
    # xhard4=9 次只为保住四键结构（v7 兼容的配置与测试），不交付、不评估（9 次超 1600 步上限，不得用于生成）；
    # 本环境不加 xhard5；圆盘区域（XHARD_DECISION）是非梯度参数，不动。
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
        # 必须落在任何随机数调用与 super().__init__() 之前：这里多抽或少抽一次会平移其后全部取值
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        # 步 4：只读导出（不传规格）或原值回注（传冻结规格）
        self._spec = SpecRecorder(native_episode_spec, "PickXtimes", {"seed": seed},
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

       

        # V4：xhard 走独立的生成路径（G1 先放圆盘、目标候选池解耦、D2 修复、按对象回填颜色），
        # 原三档仍走原代码（整段原样搬进 _spawn_scene_objects_native，一行未改，H2/N12）。
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
            # 恢复动作的选择是一次真实抽样：原位置照常抽，回注模式下用冻结的索引
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

        # V4 xhard：干扰方块是新增的随机取值，追加在本函数全部既有取值点（含恢复动作抽样）之后（N5）。
        if is_newvalue_difficulty(self.difficulty):
            self._spawn_distractors_xhard(generator, avoid)

    def _spawn_scene_objects_native(self, generator, avoid):
        """原三档的方块／圆盘／目标方块生成（原 ``_load_scene`` 中段，逐字搬出，行为不变）。"""
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
        """V4 xhard 的方块／圆盘／目标方块生成（计划 2.4）。

        与原路径的差别（全部只在 xhard 生效，H2）：
        * G1：**先放圆盘再放方块**，方块经 ``_disk_avoid_obb`` 显式避让圆盘；
        * 颜色取自 ``NATIVE_SAMPLING.parameters.color_pool``（xhard 路径的单一真值，不再读硬编码字面量）；
        * V5（计划 2.13）：三个有色方块在区域内均匀抽（L43 取消 V4 的 corner_bias）；6 块两两中心距
          ≥ ``min_center_dist_m``（L44，经 ``spawn_random_cube(min_center_dist=...)``）；已放方块以
          ``cube_obb2d_exact`` 精确 OBB 进 ``avoid``（修 2.0① 障碍框退化）；每块预算 ``XHARD_CUBE_MAX_TRIALS``（L45）；
        * D2：圆盘或方块放不下直接抛 ``SceneGenerationError``，不再静默截断或落到未绑定变量；
        * 目标方块从显式候选列表 ``self.target_candidates`` 抽（与 ``all_cubes`` 解耦，干扰物不会被抽中）；
        * ``target_color_name`` 按对象查表回填，不再走三色 if 链（新颜色不命中时会残留旧值）。
        随机调用的相对顺序：color_order → target_color_idx → 圆盘 → 方块 → target_cube_idx，
        其后才是恢复动作与干扰方块（N5）。
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
        # V5：已放方块（有色 + 干扰共用一张表）的精确 OBB；既作中心距规则的参考点，也作 avoid 里的障碍
        self._xhard_cube_obbs = []

        self.all_cubes = []
        self.distractor_cubes = []
        self._cube_color_of = []  # [(actor, 颜色名)]，按对象回填 target_color_name 用
        color_groups = []
        for entry in self._sampling["parameters"]["color_pool"]:
            name = entry["name"]
            cube_list, name_list = [], []
            # 保留 red_cubes / red_cube_names 这类属性名，下游按颜色取列表的代码照常可用
            setattr(self, f"{name}_cubes", cube_list)
            setattr(self, f"{name}_cube_names", name_list)
            color_groups.append({"color": tuple(entry["rgba"]), "name": name,
                                 "list": cube_list, "name_list": name_list})

        shuffle_indices = self._spec.value(
            "objects.color_order", torch.randperm(len(color_groups), generator=generator).tolist()
        )
        color_groups = [color_groups[i] for i in shuffle_indices]
        # 与原路径同一次抽样（其值随后被目标方块的颜色覆盖），保留以对齐取值点集合
        target_color_idx = self._spec.value(
            "objects.target_color_idx",
            torch.randint(0, len(color_groups), (1,), generator=generator).item(),
        )
        self.target_color_name = color_groups[target_color_idx]["name"]

        # G1：先放圆盘。此时 avoid 里只有按钮。
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
            # D2（xhard 专用修复）：原路径失败后会落到未绑定的 target 上抛 UnboundLocalError
            raise SceneGenerationError(f"PickXtimes xhard: 放置圆盘采样失败: {exc}") from exc
        self.target = target
        avoid.append(target)
        # 圆盘本身取不到 OBB（纯视觉 actor），方块避让靠这个预制外接正方形
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
                    raise SceneGenerationError(f"PickXtimes xhard: 方块 {cube_name} 放不下: {exc}") from exc
                self.all_cubes.append(cube)
                group["list"].append(cube)
                group["name_list"].append(cube_name)
                self._cube_color_of.append((cube, group["name"]))
                setattr(self, cube_name, cube)
                self._append_cube_obstacle_xhard(cube, avoid)
        # 2.2④：请求数 vs 实际数，不等即本局失败（上面的 raise 已保证，这里再显式记录与核对）
        self._spec.record("objects.cube_count", {"requested": requested, "actual": len(self.all_cubes)})
        if len(self.all_cubes) != requested or requested == 0:
            raise SceneGenerationError(
                f"PickXtimes xhard: 有色方块请求 {requested} 实际 {len(self.all_cubes)}"
            )

        # 目标候选池与 all_cubes 解耦：只含有色方块，之后追加的干扰方块永远不会被抽成目标
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
        """按对象查颜色名（xhard 路径用）；查不到说明登记漏了，直接报错而不是残留旧值。"""
        for actor, name in self._cube_color_of:
            if actor is cube:
                return name
        raise SceneGenerationError("PickXtimes xhard: 目标方块不在颜色登记表里")

    def _append_cube_obstacle_xhard(self, cube, avoid):
        """V5（计划 2.0① / 2.13）：把刚放下的方块以精确 OBB 登记为后续方块的障碍与中心距参考点。

        不再把 actor 本身放进 ``avoid``：actor 路径经 ``_trimesh_box_to_obb2d``，对正方体约 2/3 的姿态
        退化成线段，``min_gap`` 在其法向上失效。纯几何，不抽随机数。
        """
        obb = cube_obb2d_exact(cube, self.cube_half_size)
        self._xhard_cube_obbs.append(obb)
        avoid.append(obb)

    def _spawn_distractors_xhard(self, generator, avoid):
        """V4 xhard：放三个「其他颜色」干扰方块（A5/B2：黄／青／品红各一）。

        干扰方块进 ``all_cubes`` 与 ``non_target_cubes``（抓错即触发 failure_func 判失败），
        不进 ``target_candidates``。放不下直接抛 ``SceneGenerationError``（2.2④，不许静默截断）。
        V5：与有色方块共用中心距规则、精确 OBB 障碍与拒绝预算（计划 2.13）。
        """
        min_center_dist = float(self._sampling["decision"][self.difficulty]["min_center_dist_m"])
        dcfg = self._sampling["decision"][self.difficulty]["distractor"]
        palette = {entry["name"]: entry["rgba"] for entry in BLOCK_DISTRACTOR_COLORS}
        names = list(dcfg["colors"])
        unknown = [name for name in names if name not in palette]
        if unknown:
            raise SceneGenerationError(f"PickXtimes xhard: 干扰色不在 BLOCK_DISTRACTOR_COLORS 里: {unknown}")
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
                raise SceneGenerationError(f"PickXtimes xhard: 干扰方块 {cube_name} 放不下: {exc}") from exc
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
                f"PickXtimes xhard: 干扰方块请求 {len(names)} 实际 {len(self.distractor_cubes)}"
            )
        # failure_func 在调用时才读 self.non_target_cubes，这里重建即可让干扰方块参与判失败
        self.non_target_cubes = [cube for cube in self.all_cubes if cube is not self.target_cube]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 每次初始化各自记一份规格，不复用上一次的结果
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
