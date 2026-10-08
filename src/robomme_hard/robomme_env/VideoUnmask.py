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
# V5 xhard（L13 / L14）：统一干扰采样器与独立停放点；只在 xhard 分支调用，原三档不进该模块
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


# ── decision／native 两块的原值（newtaskRelease-v3 步 3，映射见方案第二节 2.5）────────
NATIVE_SAMPLING = {
    "parameters": {
        "color_pool": [
            {"rgba": [1, 0, 0, 1], "name": "red"},
            {"rgba": [0, 1, 0, 1], "name": "green"},
            {"rgba": [0, 0, 1, 1], "name": "blue"},
        ],
        "color_order": {"sampler": "torch.randperm(3)"},
        "hidden_rule": "前 3 个容器各藏一块，其余为空",
        "pick_rule": "先抓 bin_0，pick>1 再抓 bin_1",
        "recovery": "沿用入口给定的 fail recover 模式与原 generator",
        "step_bin_scan": 15,
    },
    "positions": {
        "bins": {
            "min_gap_factor": 2,
            "max_trials": 256,
            "yaw_expression": "u * 90 度（通过拒绝检查后才抽）",
        },
        "hidden_cube": {"half_size_divisor": 1.2, "yaw": 0.0, "dynamic": True},
        "reveal_window": {"start_step": 0, "end_step": 64},
    },
}


# ── V4 xhard 专属的 decision 条目（NEWTASK_RELEASE_V4_PLAN 2.7 / 2.8；G2、B3、B13 均为用户已定数）──
# 原三档不读这些键；守卫对名为 xhard 的子键只校验结构、放行取值（sampling_config.assert_native_decision）。
XHARD_BIN_LAYOUT = {
    # G2（2026-09-22）：区域不动，间距系数 2 → 0.75，容器数 8；放不满直接判该局失败
    "min_gap_factor": 0.75,
}
# V5（NEWTASK_RELEASE_V5_PLAN 2.3；L6～L13）：贴身环带 [0.2425, 0.3289]，按内部密度定数，半数含 cube，
# 三色平衡轮转，1024 次；统一 7 键 schema（count / ring_max_abs_xy / cube_count_range / color_pool /
# color_rule / min_gap_factor / max_trials），取值见 unmask_distractor_sampler.V5_DISTRACTOR_PRESETS。
XHARD_DISTRACTOR = copy.deepcopy(V5_DISTRACTOR_PRESETS["VideoUnmask"])


# ── V6（NEWTASK_RELEASE_V6_PLAN 2.3）新值族档位表：xhard1/2/3 沿用 xhard 的全部机制（8 内环容器 + 贴身环带 +
# 三色轮转 + 独立停放），只按档改干扰总数 count 与含 cube 个数 cube_count_range；环带宽度、间距、
# 尝试次数等一律沿用 xhard（干扰数少时即在同一环带内更稀疏地随机放置，不重新推导带宽）。
# 「xhard」键即原 XHARD_* 常量本身，取值逐位不变。
def _newvalue_distractor(count, cube_count_range):
    """以 xhard 的干扰配置为底，只替换 count 与 cube_count_range。"""
    cfg = copy.deepcopy(XHARD_DISTRACTOR)
    cfg["count"] = count
    cfg["cube_count_range"] = list(cube_count_range)
    return cfg


# V7 定值（0928 方案 §3.2.2）：贴身环带干扰 0/4/8/12、含 cube 数恒为一半；内环 8 固定 ⇒ 桌面容器总数 8/12/16/20。
# v8（1001 方案 §1 表 1 / §2.1）：xhard1 干扰 0 → 4、含 cube 0 → 2（pick_count 仍 2），与 xhard2 同干扰、少抓一次；
# xhard2～4 不动 ⇒ 干扰 4/4/8/12、桌面容器总数 12/12/16/20。
# XHARD_DISTRACTOR（V5 预设 15 个）本身不动，xhard4 改用 12 个（顶档下调）。
NEWVALUE_DISTRACTOR = {
    "xhard4": _newvalue_distractor(12, [6, 6]),
    "xhard1": _newvalue_distractor(4, [2, 2]),
    "xhard2": _newvalue_distractor(4, [2, 2]),
    "xhard3": _newvalue_distractor(8, [4, 4]),
}
# 内环容器摆放（间距系数 0.75）四档相同
NEWVALUE_BIN_LAYOUT = {tier: XHARD_BIN_LAYOUT for tier in NEWVALUE_DISTRACTOR}


def native_blocks(cls):
    """本环境的 ``(decision, native)`` 原值块；外部导出与内部解析共用同一份。"""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


def _native_decision(cls):
    """按方案第二节 2.5 切出 decision 块（原值阶段等于原值）。"""
    return {
        # 需要拾取的目标数量；V4 起 xhard 为 3（经 config_xhard4 进入本字典的 xhard 键）。
        "pick_count": {difficulty: cfg["pick"] for difficulty, cfg in cls.configs.items()},
        # 容器怎样摆放、摆放区域多大；原值＝按难度给的容器数 + 同一块区域。
        "bin_layout_policy": {
            "count": {difficulty: cfg["bin"] for difficulty, cfg in cls.configs.items()},
            "region_center": [0, 0],
            "region_half_size": 0.2,
            # xhard 在前（键序与 V5 相同），V6 的 xhard1/2/3 追加其后
            **{tier: copy.deepcopy(layout) for tier, layout in NEWVALUE_BIN_LAYOUT.items()},
        },
        # 原三档可见部分保持 None；V4 的干扰容器放在 xhard 子键下
        "distractor": None,
        **{tier: {"distractor": copy.deepcopy(dist)} for tier, dist in NEWVALUE_DISTRACTOR.items()},
    }


def _resolve_sampling_config(cls, override):
    """拆出本实例专属的 decision／native 副本；不抽随机数，必须在 Generator 之前调用。"""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    # V6：旧快照（V5 没有 xhard1/2/3 子树）从源码申报补齐
    fill_missing_newvalue(decision, decision_default)
    native["decision"] = decision
    return native


@register_env("VideoUnmask", override=True)
class VideoUnmask(BaseEnv):

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

    # V4 xhard（派生自 hard，计划 2.8）：pick 2 → 3；容器数 15 → 8（G2：配 min_gap_factor 0.75，
    # 见 XHARD_BIN_LAYOUT）。另有贴身环带干扰容器（V5：XHARD_DISTRACTOR，VU 15 个 / BU 14 个），不计入 bin。
    config_xhard4 = {
    'bin':8,
    "pick":3,
    }

    # V6（计划 2.3）新值族 xhard1/2/3：内环容器沿 xhard 为 8，pick 按档 2/3/3
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
        # 必须落在任何随机数调用与 super().__init__() 之前
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "VideoUnmask", {"seed": seed},
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
        generator = torch.Generator()
        generator.manual_seed(self.seed)
    
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        avoid=[]


         # Generate 3 bins
        self.spawned_bins = []
        decision_cfg = self._sampling["decision"]
        bin_layout = decision_cfg["bin_layout_policy"]
        bins_cfg = self._sampling["positions"]["bins"]
        hidden_cfg = self._sampling["positions"]["hidden_cube"]
        # V6：新值族（xhard1/2/3/xhard）同走 xhard 机制，数值按本局档位查表
        xhard = is_newvalue_difficulty(self.difficulty)
        # V4 xhard：间距系数取 decision 的 xhard 条目（G2 0.75）；原三档仍读 native 的原值，表达式不变
        gap_factor = (bin_layout[self.difficulty]["min_gap_factor"] if xhard
                      else bins_cfg["min_gap_factor"])
        if xhard:
            requested_bins = bin_layout["count"][self.difficulty]
            # 揭示动画只扫 bin_0..bin_{scan-1}：容器数超过它的部分不会被揭示（计划 2.8）
            if self._sampling["parameters"]["step_bin_scan"] < requested_bins:
                raise ValueError(
                    f"step_bin_scan={self._sampling['parameters']['step_bin_scan']} 小于容器数 {requested_bins}"
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
                logger.debug(f"Spawned bin_{i} at position {bin_actor.pose.p}")
            except RuntimeError as e:
                if xhard:
                    # 2.2④：xhard 下放不满即该局失败，不许静默截断
                    self._spec.record("layout.bin_count.placed", len(self.spawned_bins))
                    raise SceneGenerationError(
                        f"VideoUnmask xhard 容器放不满：请求 {requested_bins} 个，只放下 {len(self.spawned_bins)} 个"
                    ) from e
                break

            self.spawned_bins.append(bin_actor)
            # Assign bin to self.bin_0, self.bin_1 etc. attributes
            setattr(self, f"bin_{i}", bin_actor)
            # Add newly generated bin to avoidance list
            avoid.append(bin_actor)
        if xhard:
            self._spec.record("layout.bin_count.placed", len(self.spawned_bins))


        # Generate 3 dynamic cubes under each bin (use fixed position, colors red, green, blue)
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

        # Only generate cubes for first 3 bins
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
                dynamic=hidden_cfg["dynamic"]
            )

            spawned_dynamic_cubes.append(cube_actor)
            # Assign cube to attributes like self.target_cube_red, self.target_cube_green, etc.
            setattr(self, f"target_cube_{color_names[i]}", cube_actor)
            # Also store using numeric index for easy access
            setattr(self, f"target_cube_{i}", cube_actor)
            # Add newly generated cube to avoidance list
            avoid.append(cube_actor)

        tasks = [
             {
                            "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=64),
                            "name": "static",
                            "subgoal_segment": "static",
                            "choice_label": "static",
                            "demonstration": True,
                            "failure_func": None,
                            "solve": lambda env, planner: solve_hold_obj(env, planner, static_steps=64),
                        },
            

            
            {
                "func": (lambda: is_bin_pickup(self, obj=self.bin_0)),
                "name": f"pick up the container that hides the {self.color_names[0]} cube",
                "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[0]} cube",
                "choice_label": "pick up the container",
                "demonstration": False,
                "failure_func": lambda: is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.bin_0]),
                "solve": lambda env, planner: solve_pickup_bin(env, planner, obj=self.bin_0),
                "segment":self.bin_0,
            },]
        if xhard:
            # V4 xhard：按 pick_count 循环追加「放下上一个 → 抓第 k 个」（原分支写死 bin_0/bin_1，只能 2 抓）
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
                    "solve": lambda env, planner: solve_pickup_bin(env, planner,obj=self.bin_1),
                    "segment":self.bin_1,
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

        if xhard:
            # V4 xhard 干扰容器：必须在全部既有取值点之后（含上面 inject_fail_grasp 用同一 generator 的抽样），红线 N5
            # V5（L13）：统一干扰采样器；仍用主场景 generator、仍在全部既有取值点之后，内环取值与 V4 同 seed 逐位相同
            self.distractor_bins, self.distractor_cubes, self.distractor_layout = spawn_distractor_layout(
                self,
                cfg=decision_cfg[self.difficulty]["distractor"],
                decision_prefix=f"{self.difficulty}.distractor",
                avoid=avoid,
                generator=generator,
                recorder=self._spec,
                hidden_half_size=self.cube_half_size/hidden_cfg["half_size_divisor"],
            )
            # V4 xhard（用户 2026-09-22「误抓即失败」）：每个已有 failure_func 的抓取／放下任务追加
            # 「任一干扰容器被抬起（z>0.15，与区域内容器同一判据）即失败」；原三档不进此分支
            add_distractor_misgrasp_failure(self, self.task_list)

    def _append_xhard_pick_tasks(self, tasks, pick_total):
        """xhard 专用：把第 2..pick_total 次抓取按「放下上一个容器 → 抓下一个」追加进任务表。

        各条目与原 hard 分支的第 2 抓逐项同构，只把写死的 bin_0/bin_1、color_names[0]/[1] 换成按 k 取；
        lambda 用默认参数绑定当次的容器，避免循环变量晚绑定。
        """
        if pick_total > min(len(self.spawned_bins), len(self.color_names)):
            raise SceneGenerationError(
                f"pick_count={pick_total} 超过可抓的藏物容器数 {min(len(self.spawned_bins), len(self.color_names))}"
            )
        # 任务目标文本（utils/task_goal.py）在 xhard 下读这个实际次数
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
                    "solve": lambda env, planner, b=cur_bin: solve_pickup_bin(env, planner,obj=b),
                    "segment":cur_bin,
                })

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
        self.failureflag = torch.tensor([False])
        test=[bin for bin in self.spawned_bins if bin != self.bin_0]
       

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
            # V5 xhard（L14，主会话 2026-09-24 定：内环容器也停独立点）：窗口与半窗落回步与原机制逐步相同，
            # 只把「远处」从共用的 (10,10,10) 换成每个物体各自的画面外停放点，免得 20 多个容器叠放拖慢物理。
            # 内环容器与原循环扫同一组 bin_<i>（i < step_bin_scan），第 i 个停在 xhard_park_point("bin", i)；
            # 被藏 cube 在本环境的揭示里本来就不动，无需停放。
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
            # 干扰容器（用户 2026-09-22「参与揭示」）：同一窗口，每个一个停放点
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
