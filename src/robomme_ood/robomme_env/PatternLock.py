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
# V5 L3（仿 VideoPlaceOrder 的 K2 修法）：上一行的 `from .utils import *` 会把同名子模块
# `utils.SceneGenerationError` 盖到名字 `SceneGenerationError` 上（import 自省核实），原三档的
# raise / except 因此是 TypeError（按 H2 原三档保持现状）。xhard 用下面这个别名拿到真正的异常类。
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
    """V5 L3：按档选场景生成异常类。

    xhard 返回真正的 ``SceneGenerationError``（可重试的任务性失败）；原三档原样返回本模块里
    被遮蔽的名字 ``SceneGenerationError``（子模块，raise / except 时仍是 TypeError，行为逐字不变）。
    用法：``raise _scene_gen_error(self.difficulty)("说明")``、``except _scene_gen_error(self.difficulty):``；
    只在 xhard 路径上执行的代码直接用 ``_RealSceneGenerationError``。
    """
    # V6 口径 11：新值族（xhard1/2/3/xhard）都走真异常类，原三档不变
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


# ── decision／native 两块的原值（newtaskRelease-v3 步 3，映射见方案第二节 2.15）────────
NATIVE_SAMPLING = {
    "parameters": {
        "path_selection": {
            "start_end_sampler": "torch.randperm(num_targets)[:2]",
            "search": "允许对角线的随机 DFS（find_path_0_to_8, diagonals=True）",
            "max_attempts": 1000,
            "exhausted_rule": "耗尽后使用最后一次搜索到的路径，不另抽",
        },
        "path_binding": "演示与执行复用同一条路径；首个目标保留 NO RECORD",
        "motion_template": "每个目标 solve_swingonto 两次 screw 运动并 close_gripper",
        "recovery": "沿用入口给定的 fail recover 模式与原 generator",
    },
    "positions": {
        "grid_center": [-0.1, 0],
        "grid_spacing": 0.1,
        "node_position_expression": "center + (index - (n-1)/2) * spacing",
    },
}


# ── V5 xhard 专属 decision（计划 2.10，L35/L36）──────────────────────────────
# 路径搜索预算：原三档仍读 native 的 ``path_selection.max_attempts``（1000）；xhard 读这里，
# 随 decision 一起冻进规格 header。（V6 审查 N13 注：xhard4 节点数区间为 [21,25]，不是 V5 的固定 25；
# 「20000 次下 25 节点命中率 1.000」是 V5 规划期探针 P3 的历史数据。DFS 命中第一条落在区间内的路径即停，
# 节点数分布贴下界——xhard4 10 候选实测 {21:6, 22:2, 23:2}，用户裁决只改注释、不改分布。）
XHARD_DECISION = {
    "path_search_max_attempts": 20000,
}

# V6（计划 2.11）：新值族档位表。xhard1/2/3 键结构与 xhard 完全相同；计划未给出新档的搜索预算
# 以外的差异——预算一律沿用 xhard 的 20000，耗尽同样抛真 SceneGenerationError。
NEWVALUE_DECISION = {
    "xhard1": {"path_search_max_attempts": 20000},
    "xhard2": {"path_search_max_attempts": 20000},
    "xhard3": {"path_search_max_attempts": 20000},
    "xhard4": XHARD_DECISION,
}


def native_blocks(cls):
    """本环境的 ``(decision, native)`` 原值块；外部导出与内部解析共用同一份。"""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


def _native_decision(cls):
    """按方案第二节 2.15 切出 decision 块（原值阶段等于原值）。"""
    return {
        # 演示视频目标时长及调节时长的方式：本轮不启用（None 即由原路径与求解运动决定）。
        "demo_duration_seconds_range": None,
        "demonstration_duration_policy": "native",
        # 网格边长与路径节点数范围按字段表属 native 规则，这里只记录原值供核验，不作为新参数。
        "grid": {difficulty: cfg["grid"] for difficulty, cfg in cls.configs.items()},
        "path_length_range": {difficulty: list(cfg["length"]) for difficulty, cfg in cls.configs.items()},
        # V5 xhard 专属（计划 2.10）：键名为 xhard，守卫只放行这一子树取新值，原三档可见部分不变。
        "xhard4": copy.deepcopy(NEWVALUE_DECISION["xhard4"]),
        # V6：追加 xhard1/2/3 三棵同结构子树（xhard 保持首位与原值不变）
        # v8：xhard4 写字面值（不再用 NEWVALUE_DIFFICULTIES[-1] 指代最难档）
        **{d: copy.deepcopy(NEWVALUE_DECISION[d]) for d in NEWVALUE_DIFFICULTIES if d != "xhard4"},
    }


def _resolve_sampling_config(cls, override):
    """拆出本实例专属的 decision／native 副本；不抽随机数，必须在 Generator 之前调用。"""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    # V6：V5 快照（已有顶层 xhard 子树）缺 xhard1/2/3 时从源码补齐；更早的快照不补，
    # 保持 V5「V4 及更早快照在新值档上直接报错、不静默取源码新值」的口径 13。
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

    # V4 xhard（派生自 hard，B8）：布局与搜法都不动，只把节点数提到 [20,24]。
    # V5（计划 2.10，L35/L36）曾把节点数固定为 25（5×5 不重访路径的上限）。
    # V6（0925 计划第三节）：xhard4 节点数改为闭区间 [21,25]（xhard1/2/3 为 [9,12]/[13,16]/[17,20]），
    # 搜索预算 20000（decision.<tier>.path_search_max_attempts）与耗尽抛真 SceneGenerationError 沿用。
    # ⚠ DFS 命中第一条在区间内的路径即 break，实际节点数贴下界（V6 审查 N13：xhard4 10 候选 {21:6, 22:2, 23:2}）；
    # 用户裁决只更新注释，不改采样分布。
    # V7 定值：节点数 12/15/18/21（用户 2026-09-29 定「直接改 12/15/18/21」，定长 24 在 v6 实测里几乎搜不出）
    config_xhard4 = {
        "grid": 5,
        "length": [21, 21]
    }

    # V6（计划 2.11）：hard 与 xhard 之间插入三档，布局、搜法、预算与耗尽抛错全部沿用 xhard，只改节点数
    # v8（1001 方案 §1 表 1 / §2.1）：xhard1～3 由定值 12/15/18 改区间 [9,12]/[13,15]/[16,18]；
    # 采样代码不改（拒绝采样取第一条落在区间内的路径），区间内各节点数比例不保证均匀；xhard4 仍 21（不交付）。
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
        # 必须落在任何随机数调用与 super().__init__() 之前
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "PatternLock", {"seed": seed},
                                  difficulty=kwargs.get("difficulty"))
        # 初始化序号从 -1 起，_initialize_episode 每次进来先加一；
        # _load_scene 里的取值点用不带序号的路径，所以这里只作兜底。
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
            # V5（计划 2.10）：xhard 的搜索预算来自 decision（冻进规格 header），原三档仍用上一行的 1000
            max_attempts = self._xhard_decision("path_search_max_attempts")

        self._spec.identity.setdefault("difficulty", getattr(self, "difficulty", None))
        for attempt in range(max_attempts):
            # 每次尝试都照常抽；被接受的那一次由规格定死（失败尝试仍消费随机数）
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
                # V5（计划 2.10 / L36 / L3）：xhard 搜索耗尽不再静默沿用最后一条错长路径（K1 根因），
                # 抛真正的 SceneGenerationError（可重试的任务性失败）；原三档仍走下一行的静默兜底。
                raise _RealSceneGenerationError(
                    f"PatternLock {self.difficulty}: {max_attempts} 次搜索内没有节点数落在 "
                    f"{decision_cfg['path_length_range'][self.difficulty]} 的路径"
                )
            # If we couldn't find a path < 5 after max_attempts, use the last one
            logger.debug(f"Warning: Could not find path after {max_attempts} attempts")

        # 搜索循环里每次尝试都照常抽随机数；这里只冻结最终被采用的那条路径
        path_nodes = self._spec.value("actions.path_nodes", list(path_nodes),
                                      decision_key=f"path_length_range.{self.difficulty}")
        self._spec.record("actions.path_attempts", attempt + 1)
        if is_newvalue_difficulty(self.difficulty):
            # V5（N17 精神）：回放冻结规格时 value() 直接返回冻结值、不复核，这里复核节点数与邻接
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
        """V5：读 xhard 专属 decision 键（只在 xhard 路径上调用）。

        缺键说明传入的 sampling_config 来自 V4 或更早的快照（没有 ``decision.xhard``）；
        V4 已作废（口径 13），直接报错而不是回退到原三档的值。
        """
        # V6：按本局档位取子树（新值族四档同结构）
        xhard_cfg = self._sampling["decision"].get(self.difficulty)
        if not isinstance(xhard_cfg, dict) or key not in xhard_cfg:
            raise ValueError(
                f"PatternLock {self.difficulty}: sampling_config.decision 缺少 {self.difficulty}.{key}"
                "（V4 及更早的快照在 V5 代码上不可用）"
            )
        return xhard_cfg[key]

    @staticmethod
    def _check_xhard_path(path_nodes, num_rows, num_cols, length_range):
        """V5（N17 精神）：复核 xhard 路径——节点数在范围内、不重访、相邻两点 8 邻接。

        导出模式下搜索循环已保证这些性质，这里等于自检；回放冻结规格时 ``SpecRecorder.value``
        直接返回冻结值、不复核，靠这里挡住被改坏或来自旧规则的规格。违反即抛 ``EpisodeSpecError``。
        """
        nodes = [int(v) for v in path_nodes]
        low, high = int(length_range[0]), int(length_range[1])
        if not low <= len(nodes) <= high:
            raise _EpisodeSpecError(f"PatternLock xhard: 路径节点数 {len(nodes)} 不在 [{low}, {high}]")
        if len(set(nodes)) != len(nodes):
            raise _EpisodeSpecError("PatternLock xhard: 路径重访了节点")
        if any(not 0 <= v < num_rows * num_cols for v in nodes):
            raise _EpisodeSpecError("PatternLock xhard: 路径节点越出网格")
        for a, b in zip(nodes, nodes[1:]):
            dr = abs(a // num_cols - b // num_cols)
            dc = abs(a % num_cols - b % num_cols)
            if max(dr, dc) != 1:
                raise _EpisodeSpecError(f"PatternLock xhard: 路径 {a} → {b} 不是 8 邻接")

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
