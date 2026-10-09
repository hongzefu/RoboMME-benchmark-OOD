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
# V5 L3（仿 VideoPlaceOrder 的 K2 修法）：上一行的 `from .utils import *` 会把同名子模块
# `utils.SceneGenerationError` 盖到名字 `SceneGenerationError` 上（import 自省核实），原三档的
# raise / except 因此是 TypeError（按 H2 原三档保持现状）。xhard 用下面这个别名拿到真正的异常类。
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


# ── decision／native 两块的原值（newtaskRelease-v3 步 3，映射见方案第二节 2.3）────────
# decision：摆动轮数范围、颜色数、额外其他颜色干扰物（本轮不启用）。
# native：颜色排列与目标选择、方块／两个圆盘／按钮的区域与几何、左右顺序与成功阈值、恢复规则。
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
        "recovery": "沿用入口给定的 fail recover 模式与原 generator",
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


# ── V4 xhard 专属 decision（计划 2.5，A5 / B2）─────────────────────────────────
# distractor：四个「其他颜色」干扰方块，黄／青／品红／第 4 色各一（BLOCK_DISTRACTOR_COLORS，V7），
# 区域沿用原方块区域（中心 [-0.1,0]、半边长 0.25，容量宽松）。
# 干扰色**不并入** native.color_pool：并入会改变 randperm(len(color_groups)) 的长度，平移原三档随机流。
# min_center_dist_m：V5 L44（计划 2.14），6 块（3 有色 + 3 干扰）两两中心距下限（米）；本环境没有 corner_bias。
XHARD_DECISION = {
    "distractor": {
        "colors": [entry["name"] for entry in BLOCK_DISTRACTOR_COLORS],
        "region_center": [-0.1, 0],
        "region_half_size": 0.25,
    },
    "min_center_dist_m": 0.08,
}


def _newvalue_decision(n_distractors):
    """V6（计划 2.8）：新值族某档的 decision 子树——键结构与 ``XHARD_DECISION`` 完全相同，
    只把干扰方块颜色截成 ``BLOCK_DISTRACTOR_COLORS`` 前 k 个；区域、中心距沿用 xhard。"""
    tree = copy.deepcopy(XHARD_DECISION)
    tree["distractor"]["colors"] = [entry["name"] for entry in BLOCK_DISTRACTOR_COLORS[:n_distractors]]
    return tree


# V6 新值族档位表：干扰块数 xhard1=1、xhard2=2、xhard3=3、xhard4=3。
NEWVALUE_DECISION = {
    "xhard1": _newvalue_decision(1),
    "xhard2": _newvalue_decision(2),
    "xhard3": _newvalue_decision(3),
    "xhard4": _newvalue_decision(4),
    # v8（1001 方案 §1 表 1）：xhard5 干扰块取 4——BLOCK_DISTRACTOR_COLORS 只有黄／青／品红／橙 4 色。
    "xhard5": _newvalue_decision(4),
}


def _disk_avoid_obb(target, clearance):
    """把圆盘换算成方块拒绝采样可用的预制 OBB ``(中心, 轴, 半边长)``。

    圆盘是 ``add_collision=False`` 的纯视觉 actor，``get_actor_obb`` 取不到网格，直接放进
    ``avoid`` 会被 ``spawn_random_cube`` 静默忽略（2026-09-22 实测）。干扰方块在圆盘之后放，
    必须显式给出外接正方形：半边长 = 圆盘半径 + 圆盘间距 − 方块自带间距。
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
    """按方案第二节 2.3 切出 decision 块（原值阶段等于原值）。"""
    return {
        # 一轮＝右、左各一次；原值取自类属性的 number_min/number_max。
        "number_range": {
            difficulty: [cfg["number_min"], cfg["number_max"]]
            for difficulty, cfg in cls.configs.items()
        },
        "color": {difficulty: cfg["color"] for difficulty, cfg in cls.configs.items()},
        "distractor": None,
        # V4 xhard 专属（计划 2.5）：键名为 xhard，守卫只放行这一子树取新值，原三档可见部分不变。
        "xhard4": copy.deepcopy(XHARD_DECISION),
        # V6（计划 2.8）：xhard 之后追加 xhard1/2/3 三棵同结构子树（新值族，守卫同样放行）。
        # v8（1001 方案 §2.1）：再追加 xhard5（只有本环境与 StopCube 有这一档，R8）。
        **{tier: copy.deepcopy(NEWVALUE_DECISION[tier]) for tier in ("xhard1", "xhard2", "xhard3", "xhard5")},
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

    # v8 定值（1001 方案 §1 表 1 / §2.1）：每档一个定数，摆动轮数 4/5/6/7/8、干扰块 1/2/3/4/4（BLOCK_DISTRACTOR_COLORS）。
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

    # v8 新增：xhard5 摆动 8 轮（干扰 4 块，见 NEWVALUE_DECISION）。
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
        # v8：xhard5 追加在末尾（测试断言前几键的顺序不变）
        'xhard5': config_xhard5,
    }


    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     sampling_config=None,
                     native_episode_spec=None,
                     **kwargs):
        # 必须落在任何随机数调用与 super().__init__() 之前
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "SwingXtimes", {"seed": seed},
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
            # V4（计划 2.5）：颜色池里出现红蓝绿以外的名字时按名动态建表，不再 KeyError；
            # 原快照只含红蓝绿，这段一次都不进，原三档行为不变。
            for entry in self._sampling["parameters"]["color_pool"]:
                if entry["name"] not in _color_lists:
                    extra_cubes, extra_names = [], []
                    setattr(self, f"{entry['name']}_cubes", extra_cubes)
                    setattr(self, f"{entry['name']}_cube_names", extra_names)
                    _color_lists[entry["name"]] = (extra_cubes, extra_names)
            # 按对象回填颜色名用（xhard 路径读；原三档只写不读）
            self._cube_color_of = []
            # 颜色池取自快照（顺序与原字面量一致：红、蓝、绿）
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
                # V5（计划 2.14）：xhard 的有色方块另走一支——两两中心距规则与精确 OBB 障碍；
                # 取值点、抽样次数上限与颜色／命名登记与下面原代码相同。
                self._spawn_colored_cubes_xhard(generator, avoid, color_groups)
            else:
                # 原三档：下面整段逐字保留原代码（仅缩进一级），行为不变（H2）。
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
                # V4 xhard：目标候选池与 all_cubes 解耦、颜色按对象回填（计划 2.5，与 PickXtimes 同构）
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



        except _scene_gen_error(self.difficulty):  # V5 L3：xhard 用真类，原三档仍是被遮蔽的原名字
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

        # 摆动成功阈值与抬升高度取自快照（原值 distance 0.03 / z 0.12 / height 0.1）
        _swing_cfg = self._sampling["parameters"]["swing_thresholds"]
        for i in range(self.num_repeats):
            # V6 审查修复 N10（用户「n10 修复 a」）：序数改用共享序数表 subgoal_language._ordinal_word——
            # 前十项与原本地列表逐字相同（原三档与 xhard1～3 文本不变），第 11 轮起给 eleventh…twentieth 而不是 11th
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

    def _color_name_of(self, cube):
        """按对象查颜色名（xhard 路径用）；查不到说明登记漏了，直接报错而不是残留旧值。"""
        for actor, name in self._cube_color_of:
            if actor is cube:
                return name
        raise _RealSceneGenerationError("SwingXtimes xhard: 目标方块不在颜色登记表里")

    def _select_target_xhard(self, generator):
        """V4 xhard 的目标方块选择：从显式候选列表抽，颜色按对象回填。

        抽样位置与原路径的 ``objects.target_cube_idx`` 相同（一次 randint），只是上界取候选池长度；
        候选池只含有色方块，之后追加的干扰方块永远不会被抽成目标。
        """
        self._spec.record("objects.cube_count", {
            "requested": min(self._sampling["decision"]["color"][self.difficulty],
                             len(self._sampling["parameters"]["color_pool"]))
            * self._sampling["parameters"]["cubes_per_color"],
            "actual": len(self.all_cubes),
        })
        self.target_candidates = list(self.all_cubes)
        if not self.target_candidates:
            raise _RealSceneGenerationError("SwingXtimes xhard: 没有可选的目标候选方块")
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
        """V5（计划 2.0① / 2.14）：把刚放下的方块以精确 OBB 登记为后续物体的障碍与中心距参考点。

        不再把 actor 本身放进 ``avoid``：actor 路径经 ``_trimesh_box_to_obb2d``，对正方体约 2/3 的姿态
        退化成线段，``min_gap`` 在其法向上失效。纯几何，不抽随机数。
        """
        obb = cube_obb2d_exact(cube, self.cube_half_size)
        self._xhard_cube_obbs.append(obb)
        avoid.append(obb)

    def _spawn_colored_cubes_xhard(self, generator, avoid, color_groups):
        """V5 xhard（计划 2.14）：放三个有色方块（目标候选）。

        与原三档共用循环的差别只有两条：候选中心与已放方块两两距离 ≥ ``min_center_dist_m``（L44，经
        ``spawn_random_cube(min_center_dist=...)``，自身不抽随机数）；放下的方块以 ``cube_obb2d_exact``
        精确 OBB 进 ``avoid``（其后的两个圆盘与干扰方块都据此避让）。区域、间距、yaw、取值点路径、
        每块拒绝预算（默认 256）与原循环相同；放不下抛真 ``SceneGenerationError``（L3）。
        """
        cubes_cfg = self._sampling["positions"]["cubes"]
        cubes_per_color = self._sampling["parameters"]["cubes_per_color"]
        min_center_dist = float(self._sampling["decision"][self.difficulty]["min_center_dist_m"])
        self._spec.record("layout.cube_min_center_dist", min_center_dist)
        # 已放方块（有色 + 干扰共用一张表）的精确 OBB；既作中心距规则的参考点，也作 avoid 里的障碍
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
                            f"SwingXtimes xhard: 方块 {cube_name} 放不下: {exc}"
                        ) from exc

                    self.all_cubes.append(cube)
                    group["list"].append(cube)
                    group["name_list"].append(cube_name)
                    self._cube_color_of.append((cube, group["name"]))
                    setattr(self, cube_name, cube)
                    self._append_cube_obstacle_xhard(cube, avoid)

                logger.debug(f"Generated {len(group['list'])} {group['name']} cubes")

    def _spawn_distractors_xhard(self, generator, avoid):
        """V4 xhard：放三个「其他颜色」干扰方块（A5/B2：黄／青／品红各一）。

        干扰方块进 ``all_cubes`` 与 ``non_target_cubes``（抓错即触发 failure_func 判失败），
        不进 ``target_candidates``；两个圆盘经 ``_disk_avoid_obb`` 显式避让。
        放不下直接抛 ``SceneGenerationError``（2.2④，不许静默截断）。
        V5：与有色方块共用中心距规则与精确 OBB 障碍（计划 2.14）。
        """
        min_center_dist = float(self._sampling["decision"][self.difficulty]["min_center_dist_m"])
        dcfg = self._sampling["decision"][self.difficulty]["distractor"]
        palette = {entry["name"]: entry["rgba"] for entry in BLOCK_DISTRACTOR_COLORS}
        names = list(dcfg["colors"])
        unknown = [name for name in names if name not in palette]
        if unknown:
            raise _RealSceneGenerationError(f"SwingXtimes xhard: 干扰色不在 BLOCK_DISTRACTOR_COLORS 里: {unknown}")
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
                raise _RealSceneGenerationError(f"SwingXtimes xhard: 干扰方块 {cube_name} 放不下: {exc}") from exc
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
                f"SwingXtimes xhard: 干扰方块请求 {len(names)} 实际 {len(self.distractor_cubes)}"
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
