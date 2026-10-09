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
from .utils.sampling_config import (
    SamplingConfigError, assert_native_decision, fill_missing_newvalue, split_sampling_config,
)
from .utils.SceneGenerationError import SceneGenerationError
from .utils import reset_panda
from .utils.difficulty import normalize_robomme_difficulty, is_newvalue_difficulty
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


# ── decision／native 两块的原值（newtaskRelease-v3 步 3，映射见方案第二节 2.9）────────
# decision：方块摆放模式与采样区域、同时高亮并需抓取的目标数、场上方块总数、逐块颜色策略。
# native：每块的具体颜色与位姿抽法、高亮目标选择（randperm）、按钮位置、恢复、
#        高亮窗口与抓取顺序（所有目标共用同一个窗口、同时高亮）。
NATIVE_SAMPLING = {
    "parameters": {
        "color_pool": [
            {"rgba": [1, 0, 0, 1], "name": "red"},
            {"rgba": [0, 0, 1, 1], "name": "blue"},
            {"rgba": [0, 1, 0, 1], "name": "green"},
        ],
        "color_draw": {"sampler": "torch.randint", "low": 0, "high_exclusive": "len(color_pool)"},
        "target_selection": {"sampler": "torch.randperm(len(all_cubes))[:pickup]"},
        "spawn_failure": "生成失败 break，保存实际数量，不补抽",
        "recovery": "沿用入口给定的 fail recover 模式与原 generator",
    },
    "positions": {
        "button": {"center_xy": [-0.2, 0], "scale": 1.5},
        "cubes": {
            "region_center": [-0.1, 0],
            "region_half_size": 0.2,
            "min_gap_factor": 2,
            "random_yaw": True,
            "include_existing": False,
            "include_goal": False,
        },
        "highlight_window": {"start_step": 10, "end_step": 100, "simultaneous": True},
    },
}


def native_blocks(cls):
    """本环境的 ``(decision, native)`` 原值块；外部导出与内部解析共用同一份。"""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


# V4 xhard 专属的 decision 条目（NEWTASK_RELEASE_V4_PLAN 2.12）；放在名为 ``xhard`` 的子键下，
# 守卫（assert_native_decision）去掉 xhard 后原三档可见部分与原值逐字相同。
#   block_color_policy：「block 颜色任意」＝逐块独立抽色，色域按用户 2026-09-22 决定设饱和度/亮度下限
#     （"hsv_floor"：色相任意、S≥0.5、V≥0.4，见 utils/xhard.py::HSV_FLOOR_COLOR；alpha 固定 1）。
#   subgoal_color_suffix：任意 RGB 没有颜色名，subgoal 的 ``, which is {color}`` 后缀如何写。
#     用户 2026-09-22 定「整段去掉」（"omit"）；目前只实现这一种，其余取值直接拒绝。
from .utils.xhard import HSV_FLOOR_COLOR, cube_obb2d_exact, hsv_floor_rgb

XHARD_DECISION = {
    "block_color_policy": "hsv_floor",
    "block_color_hsv": copy.deepcopy(HSV_FLOOR_COLOR),
    "subgoal_color_suffix": "omit",
}

# V6：四档沿用 HSV 任意色、精确 OBB 与 subgoal 去色后缀；档位值只改变 pick 数与总块数。
NEWVALUE_DECISION = {
    "xhard4": XHARD_DECISION,
    "xhard1": copy.deepcopy(XHARD_DECISION),
    "xhard2": copy.deepcopy(XHARD_DECISION),
    "xhard3": copy.deepcopy(XHARD_DECISION),
}


def _native_decision(cls):
    """按方案第二节 2.9 切出 decision 块（原值阶段等于原值）。

    ``highlight_count`` / ``spawn_count`` 按 ``cls.configs`` 逐档展开：原三档是整数，
    xhard 是闭区间 ``[lo, hi]``（V4 新值，走守卫的 xhard 放行）。
    """
    return {
        "layout_mode": "native_region",
        "cube_region": {"region_center": [-0.1, 0], "region_half_size": 0.2},
        "highlight_count": {difficulty: copy.deepcopy(cfg["pickup"]) for difficulty, cfg in cls.configs.items()},
        "spawn_count": {difficulty: copy.deepcopy(cfg["spawn"]) for difficulty, cfg in cls.configs.items()},
        "block_color_policy": "native_per_cube_uniform",
        # V6：xhard 子树原值不变，再按档追加 xhard1/2/3 三棵同结构子树
        **{tier: copy.deepcopy(entry) for tier, entry in NEWVALUE_DECISION.items()},
    }


def _closed_range(value, key):
    """把 xhard 的闭区间 ``[lo, hi]`` 校验成两个整数；外部配置写坏时直接拒绝。"""
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
            or value[0] < 1 or value[0] > value[1]):
        raise SamplingConfigError(
            f"PickHighlight: decision.{key} 必须是闭区间 [lo, hi]（1≤lo≤hi 的整数），收到 {value!r}"
        )
    return int(value[0]), int(value[1])


def _resolve_sampling_config(cls, override):
    """拆出本实例专属的 decision／native 副本；不抽随机数，必须在 Generator 之前调用。"""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    # V6：旧快照（V5 没有 xhard1/2/3 子树）从源码申报补齐缺的新值档，已有的不动
    fill_missing_newvalue(decision, decision_default)
    native["decision"] = decision
    return native


@register_env("PickHighlight", override=True)
class PickHighlight(BaseEnv):

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
        'spawn': 6,
        "pickup": 3
    }

    config_easy = {
        'spawn': 3,
        "pickup": 1
    }

    config_medium = {
        'spawn': 4,
        "pickup": 2
    }

    # xhard4：定稿 pick 7、总块 10；区间退化为单值。
    config_xhard4 = {
        'spawn': [10, 10],
        "pickup": [7, 7]
    }

    # V6 新档按定稿取值，并保持 spawn 下界 ≥ pick 上界。
    config_xhard1 = {
        'spawn': [7, 7],
        "pickup": [4, 4]
    }

    config_xhard2 = {
        'spawn': [8, 8],
        "pickup": [5, 5]
    }

    config_xhard3 = {
        'spawn': [9, 9],
        "pickup": [6, 6]
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
        # 必须落在任何随机数调用与 super().__init__() 之前
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "PickHighlight", {"seed": seed},
                                  difficulty=kwargs.get("difficulty"))
        # 初始化序号从 -1 起，_initialize_episode 每次进来先加一；
        # _load_scene 里的取值点用不带序号的路径，所以这里只作兜底。
        self._native_init_index = -1
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
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
        self.generator.manual_seed(self.seed)

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
        self.generator.manual_seed(self.seed)
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()


        button_cfg = self._sampling["positions"]["button"]
        cubes_cfg = self._sampling["positions"]["cubes"]
        decision_cfg = self._sampling["decision"]
        cube_region = decision_cfg["cube_region"]
        self._spec.identity.setdefault("difficulty", getattr(self, "difficulty", None))
        button_obb = build_button(
            self,
            center_xy=tuple(button_cfg["center_xy"]),
            scale=button_cfg["scale"],
            generator=self.generator,
            recorder=self._spec,
            spec_path="layout.button_xy",
        )
        avoid = [button_obb]

        self.all_cubes = []  # Save all cube objects
        self.all_cube_names = []
        self.all_cube_colors = []

        # List of available colors
        # 颜色池取自快照（顺序与原字面量一致：红、蓝、绿）
        available_colors = [
            {"color": tuple(entry["rgba"]), "name": entry["name"]}
            for entry in self._sampling["parameters"]["color_pool"]
        ]

        # V4 xhard 分支（H2/N12：原三档一行不走这里）。
        # V6：新值族四档共用本分支，按本局档位取子树与区间
        xhard = is_newvalue_difficulty(self.difficulty)
        tier = self.difficulty
        if xhard:
            xhard_cfg = decision_cfg[tier]
            if xhard_cfg["block_color_policy"] != "hsv_floor":
                raise SamplingConfigError(
                    f"PickHighlight: decision.{tier}.block_color_policy 只支持 'hsv_floor'，"
                    f"收到 {xhard_cfg['block_color_policy']!r}"
                )
            if xhard_cfg["subgoal_color_suffix"] != "omit":
                raise SamplingConfigError(
                    f"PickHighlight: decision.{tier}.subgoal_color_suffix 目前只实现 'omit'，"
                    f"收到 {xhard_cfg['subgoal_color_suffix']!r}"
                )
            spawn_lo, spawn_hi = _closed_range(decision_cfg["spawn_count"][tier], f"spawn_count.{tier}")
            highlight_lo, highlight_hi = _closed_range(
                decision_cfg["highlight_count"][tier], f"highlight_count.{tier}"
            )
            # spawn≥highlight 硬断言（计划 2.12）：按生效区间的最坏情形判，外部收窄写坏时当场拒绝
            if spawn_lo < highlight_hi:
                raise SamplingConfigError(
                    f"PickHighlight: {tier} 须 spawn 下界 ≥ highlight 上界，收到 spawn=[{spawn_lo},{spawn_hi}] "
                    f"highlight=[{highlight_lo},{highlight_hi}]"
                )
            # 新增取值点：本局方块数（只在 xhard 抽；它决定下面循环的长度，只能排在方块循环之前）
            num_cubes_to_spawn = int(self._spec.value(
                "objects.n_cubes",
                int(torch.randint(spawn_lo, spawn_hi + 1, (1,), generator=self.generator).item()),
                decision_key=f"spawn_count.{tier}",
            ))
        else:
            # Get number of cubes to spawn based on difficulty
            num_cubes_to_spawn = decision_cfg["spawn_count"][self.difficulty]

        # Spawn specified number of cubes, each with random color
        for cube_idx in range(num_cubes_to_spawn):
            if xhard:
                # 颜色任意：逐块独立抽色（HSV 限定色域；替换原三色 randint，只在 xhard 生效）。
                # 没有颜色名 ⇒ label 置 None、actor 名用 "rgb"；subgoal 后缀按 subgoal_color_suffix 处理。
                rgba = self._spec.value(
                    f"objects.color_rgba.{cube_idx}",
                    hsv_floor_rgb(torch.rand(3, generator=self.generator).tolist(),
                                  xhard_cfg["block_color_hsv"]) + [1.0],
                    decision_key=f"{tier}.block_color_policy",
                )
                chosen_color = {"color": tuple(float(c) for c in rgba), "name": "rgb", "label": None}
            else:
                # Randomly select a color
                color_choice_idx = self._spec.value(
                    f"objects.color_choice.{cube_idx}",
                    torch.randint(
                        self._sampling["parameters"]["color_draw"]["low"],
                        len(available_colors), (1,), generator=self.generator,
                    ).item(),
                )
                chosen_color = available_colors[color_choice_idx]

            try:
                cube = spawn_random_cube(
                    self,
                    color=chosen_color["color"],
                    avoid=avoid,
                    include_existing=False,
                    include_goal=False,
                    region_center=list(cube_region["region_center"]),
                    region_half_size=cube_region["region_half_size"],
                    half_size=self.cube_half_size,
                    min_gap=self.cube_half_size*cubes_cfg["min_gap_factor"],
                    random_yaw=cubes_cfg["random_yaw"],
                    name_prefix=f"cube_{chosen_color['name']}_{cube_idx}",
                    generator=self.generator,
                    recorder=self._spec,
                    spec_path=f"layout.cubes.{cube_idx}",
                )

                cube_name = f"cube_{chosen_color['name']}_{cube_idx}"

                # Add cube immediately after successful creation
                self.all_cubes.append(cube)
                self.all_cube_names.append(cube_name)
                self.all_cube_colors.append(chosen_color.get("label", chosen_color["name"]))
                setattr(self, cube_name, cube)
                if xhard:
                    # V5（L2 b，计划 2.16）：已放方块以精确 2D 障碍进 avoid。actor 路径经
                    # _trimesh_box_to_obb2d 约 2/3 退化成线段，min_gap 在其法向失效；取建方块时的
                    # initial_pose（不依赖仿真已初始化），不抽随机数。原三档仍放 actor，逐字不变。
                    avoid.append(cube_obb2d_exact(cube.initial_pose, self.cube_half_size))
                else:
                    avoid.append(cube)

            except RuntimeError as e:
                if xhard:
                    # 2.2④：xhard 不许静默截断，放不满即判本局生成失败
                    raise SceneGenerationError(
                        f"PickHighlight xhard: 方块放不满，请求 {num_cubes_to_spawn} 实际 {len(self.all_cubes)}"
                        f"（第 {cube_idx} 块失败：{e}）"
                    ) from e
                logger.debug(f"Failed to spawn cube {cube_idx} ({chosen_color['name']}): {e}")
                break

        logger.debug(f"Generated {len(self.all_cubes)} cubes total")

        if xhard:
            # 请求数 vs 实际数（2.2④）；走到这里二者必相等
            self._spec.record("objects.n_cubes_spawned", len(self.all_cubes))
            # 原抽法 randperm(len(all_cubes)) 原位不动；高亮数是新增取值点，追加在 randperm 之后
            permutation = torch.randperm(len(self.all_cubes), generator=self.generator).tolist()
            highlight_count = int(self._spec.value(
                "objects.highlight_count",
                int(torch.randint(highlight_lo, highlight_hi + 1, (1,), generator=self.generator).item()),
                decision_key=f"highlight_count.{tier}",
            ))
            # randperm(len)[:k] 在 k>len 时会静默截断，这里显式挡住
            if highlight_count > len(permutation):
                raise SceneGenerationError(
                    f"PickHighlight xhard: 高亮数 {highlight_count} 超过实际方块数 {len(permutation)}"
                )
            target_cube_indices = self._spec.value(
                "objects.highlight_ids", permutation[:highlight_count],
                decision_key=f"highlight_count.{tier}",
            )
        else:
            # Randomly select one cube from all available cubes as the target
            target_cube_indices = self._spec.value(
                "objects.highlight_ids",
                torch.randperm(len(self.all_cubes), generator=self.generator)[:decision_cfg["highlight_count"][self.difficulty]].tolist(),
            )

        self.target_cubes = [self.all_cubes[idx] for idx in target_cube_indices]
        self.target_cube_names = [self.all_cube_names[idx] for idx in target_cube_indices]
        self.target_cube_colors = [self.all_cube_colors[idx] for idx in target_cube_indices]
        self.target_labels = [
            (color or name or "target") 
            for color, name in zip(self.target_cube_colors, self.target_cube_names)
        ]
        # Record pick count for each target cube; all must be > 1 for success
        self.target_cube_pickup_counts = {name: 0 for name in self.target_cube_names}

        # Define task list, each task contains a dictionary with function, name, demonstration flag, and optional failure_func
        tasks = []
        target_label = getattr(self, "target_cube_color", None) or getattr(
            self, "target_cube_name", None
        ) or getattr(self, "target_label", None) or "target"
        self.target_label = target_label

        if xhard:
            # D4 只在 xhard 修（H2）：原三档这里传的是构造时刻的求值结果（不是可调用），
            # 判据从未生效；xhard 包成 lambda，按下按钮之前抓起任何方块即失败。
            button_failure_func = lambda: is_any_obj_pickup(self, [cube for cube in self.all_cubes])
        else:
            button_failure_func = is_any_obj_pickup(self,[cube for cube in self.all_cubes])
        tasks.append({
            "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment":"press the button at <>",
                "choice_label": "press button",
                "demonstration": False,
                "failure_func":button_failure_func,
                "solve": lambda env, planner:solve_button(env, planner, obj=self.button),
                 "segment":self.cap_link,
            })
        # Pick each target cube once, lambda captures current cube explicitly to avoid closure issue
        num_targets = len(self.target_cubes)
        for cube_idx, cube in enumerate(self.target_cubes):
                # If only one target cube, do not show index
                if xhard:
                    # 任意 RGB 没有颜色名：subgoal_color_suffix="omit" ⇒ 去掉 ", which is {color}" 整段后缀
                    if num_targets == 1:
                        task_name = "pick up the highlighted cube"
                        task_subgoal = "pick up the highlighted cube at <>"
                    else:
                        task_name = subgoal_language.get_subgoal_with_index(cube_idx, "pick up the {idx} highlighted cube")
                        task_subgoal = subgoal_language.get_subgoal_with_index(cube_idx, "pick up the {idx} highlighted cube at <>")
                elif num_targets == 1:
                    task_name = f"pick up the highlighted cube, which is {self.target_labels[cube_idx]}"
                    task_subgoal = f"pick up the highlighted cube at <>, which is {self.target_labels[cube_idx]}"
                else:
                    task_name = subgoal_language.get_subgoal_with_index(cube_idx, "pick up the {idx} highlighted cube, which is {color}", color=self.target_labels[cube_idx])
                    task_subgoal = subgoal_language.get_subgoal_with_index(cube_idx, "pick up the {idx} highlighted cube at <>, which is {color}", color=self.target_labels[cube_idx])

                tasks.append({
                    "func": (lambda c=cube: is_any_obj_pickup_flag_currentpickup(self, objects=[c])),
                    "name": task_name,
                    "subgoal_segment": task_subgoal,
                    "choice_label": "pick up the highlighted cube",
                    "demonstration": False,
                    "failure_func": lambda idx=cube_idx:
                        [is_any_obj_pickup(self,[cube for cube in self.all_cubes if cube not in self.target_cubes] ),
                       ],
                    "solve": lambda env, planner, c=cube: solve_pickup(env, planner, obj=c),
                    "segment":cube,
                })
                if xhard or cube_idx!=num_targets-1:
                    # V6 审查修复 F1（用户 K2「f1同意修」）：新四档末块抓起后也放下，再以末尾按钮收尾；原三档末块抓起即止
                    tasks.append({
                        "func": (lambda :is_obj_dropped_currentpickup(self,self.target_cubes)),
                        "name": f"place the cube onto the table",
                        "subgoal_segment":"place the cube onto the table",
                        "choice_label": "place the cube onto the table",
                        "demonstration": False,
                        "failure_func": lambda idx=cube_idx:
                        [ is_any_obj_pickup(self,[cube for cube in self.all_cubes if cube not in self.target_cubes] ),
                           ],
                        "solve": lambda env, planner, c=cube: [solve_putdown_whenhold(env, planner, release_z=0.01),
                                                        # solve_pickup(env, planner, obj=c),
                                                        # solve_putdown_whenhold(env, planner, obj=c,release_z=0.01)# For testing
                                                        ],
                        "segment":None,
                    })
        if xhard:
            # V6 审查修复 F1（K2）：新四档链尾追加「press the button」，与题面「finally press the button to stop」一致；
            # 成功时点随之后移到末按钮（见 evaluate 的 xhard 分支）。原三档任务链不变。
            tasks.append({
                "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment": "press the button at <>",
                "choice_label": "press button",
                "demonstration": False,
                "failure_func": None,
                "solve": lambda env, planner: solve_button(env, planner, obj=self.button),
                "segment": self.cap_link,
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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 每次初始化各自记一份规格，不复用上一次的结果
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
        # Keep previous failure state (once failed, always failed)
        if not hasattr(self, 'failureflag') or self.failureflag is None:
            self.failureflag = torch.tensor([False])
        previous_failure = bool(self.failureflag.detach().cpu().item()) if isinstance(self.failureflag, torch.Tensor) else False
        # If previously failed, do not reset, keep failed state; otherwise reset
        if previous_failure:
            # Keep failed state, do not reset
            pass
        else:
            self.failureflag = torch.tensor([False])



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
            
        # Use encapsulated sequence task check function
        all_tasks_completed, current_task_name, task_failed ,self.current_task_specialflag= sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

        if task_failed:
            self.failureflag = torch.tensor([True])
            logger.debug(f"Task failed: {current_task_name}")
        
        # If previously failed, keep failed state
        if previous_failure:
            self.failureflag = torch.tensor([True])


        ############# Rising edge detection must be placed before fail detection
        target_cubes = getattr(self, "target_cubes", [])
        target_cube_names = getattr(self, "target_cube_names", [])

        if target_cubes and not hasattr(self, "target_cube_pickup_counts"):
            self.target_cube_pickup_counts = {name: 0 for name in target_cube_names}
            self.target_cube_pickup_active = {name: False for name in target_cube_names}


        if target_cubes and not hasattr(self, "target_cube_pickup_active"):
            self.target_cube_pickup_active = {name: False for name in target_cube_names}

        # Only count when cube changes from "not picked" to "picked", avoid duplicate counting in multiple frames for same pick
        for cube, name in zip(target_cubes, target_cube_names):
            pickup_tensor = is_obj_pickup(self, cube)
            if isinstance(pickup_tensor, torch.Tensor):
                picked_now = bool(pickup_tensor.detach().cpu().any())
            else:
                picked_now = bool(pickup_tensor)

            was_picked = self.target_cube_pickup_active.get(name, False)
            if picked_now and not was_picked:
                self.target_cube_pickup_counts[name] = (
                    self.target_cube_pickup_counts.get(name, 0) + 1
                )
            self.target_cube_pickup_active[name] = picked_now

        pickup_counts = getattr(self, "target_cube_pickup_counts", {})
        counts_satisfied = (
            len(pickup_counts) > 0
            and all(count >= 1 for count in pickup_counts.values())
        )
        ############# Rising edge detection must be placed before fail detection


        # Success if all picked at least once (counting discrete pick events)
        # V6 审查修复 F1（K2）：新四档还须整条任务链（含末尾按钮）走完才算成功；原三档仍在全部抓过一次时即成功
        if counts_satisfied and (all_tasks_completed or not is_newvalue_difficulty(self.difficulty)):
            self.successflag = torch.tensor([True])
       
       # Fail if planner finished but not successful
        if all_tasks_completed and not counts_satisfied:
            self.failureflag = torch.tensor([True])
            logger.debug(f"Pickup counts not satisfied: {pickup_counts}")

        if self.failureflag == torch.tensor([True]):
            pass
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


      
        timestep = self.elapsed_steps
        target_cubes = getattr(self, "target_cubes", [])

      

        if is_newvalue_difficulty(self.difficulty):
            # xhard 的 highlight_count 是区间，本局实际高亮数就是已抽定的目标数
            highlight_count = len(target_cubes)
        else:
            highlight_count = min(self._sampling["decision"]["highlight_count"][self.difficulty], len(target_cubes))
        for i in range(highlight_count):
            highlight_obj(
                self,
                target_cubes[i],
                start_step=self._sampling["positions"]["highlight_window"]["start_step"],
                end_step=self._sampling["positions"]["highlight_window"]["end_step"],
                cur_step=timestep,
            )
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
