import copy
import json
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
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils import reset_panda
from .utils.difficulty import NEWVALUE_DIFFICULTIES, is_newvalue_difficulty, newvalue_tier, normalize_robomme_difficulty
from .utils.episode_spec import SpecRecorder
from .utils.sampling_config import assert_native_decision, split_sampling_config
from .utils.bin_collision import (
    BinCollisionError,
    SpecBindingError,
    check_bin_state,
    check_swap_sweep,
    nearest_partner_index,
    object_state_from_actor,
)
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
    solve_hold_obj_xhard,
    spawn_swap_distractors_v5,
    v5_distractor_cfg,
    v5_distractor_swap_cfg,
    # V6（计划 2.2）：内环 S5 与外环 O4
    plan_inner_swaps_v6,
    v6_distractor_cfg,
    v6_distractor_swap_cfg,
    v6_inner_swap_plan_cfg,
    validate_hidden_bin_selection,
)
from ..logging_utils import logger


def _scene_gen_error(difficulty):
    """按难度族选场景生成异常类。

    新值档返回真正的 ``SceneGenerationError``（可重试的任务性失败）；原三档原样返回本模块里
    被遮蔽的名字 ``SceneGenerationError``（子模块，raise / except 时仍是 TypeError，行为逐字不变）。
    用法：``raise _scene_gen_error(self.difficulty)("说明")``、``except _scene_gen_error(self.difficulty):``；
    只在新值路径上执行的代码直接用 ``_RealSceneGenerationError``。
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


# ── 原版采样输入的原值快照（newtask-v2 10.0）────────────────────────────────────
# 说明同 BinFill：本字典即不传 sampling_config 时的默认值，也是 --extract-config 的提取目标。
# 本类没有 self.generator：__init__ 与 _load_scene 各自建局部流、各自用同一个 seed 重播种，
# 不得为统一接口把它提升为实例属性。
NATIVE_SAMPLING = {
    "parameters": {
        "object_selection": {
            "hidden_bin_permutation_size": 3,
            "hidden_bin_count_max": 3,
            "pickup_selected_indices": [0, 1],
            "swap_seed_target_count": 2,
        },
        "swap_selection": {
            "initiator_mapping": "selected_local_indices_into_spawned_bins",
            "remaining_selection": "randint_from_spawned_indices_excluding_local_targets",
            "partner": {
                "selection": "nearest",
                "position_axes": [0, 1],
                "resolve_at": "swap_start",
                "exclude_self": True,
                "tie_break": "first_in_spawn_order",
            },
        },
    },
    "positions": {
        "containers": {
            "region3_tri": [[-0.05, -0.1], [-0.05, 0.1], [0.1, 0]],
            "region3_line": [[0, -0.15], [0, 0.15], [0, 0]],
            "region4": [[-0.05, -0.1], [-0.05, 0.1], [0.1, 0.1], [0.1, -0.1]],
            "region3_choice": {
                "sampler": "torch.randint",
                "low": 0,
                "high_exclusive": 2,
                "order": ["region3_tri", "region3_line"],
            },
            "layout_rotation_range_rad": [0, 180],
            "region_half_size": 0.07,
            "yaw_scale_deg": 90.0,
            "yaw_expression": "u * 90.0",
            "rotation_center": [0, 0],
            "min_gap": "self.cube_half_size",
            "min_gap_value": 0.02,
            "spawn_random_bin_default_min_gap": 0.05,
            "spawn_random_bin_has_include_flags": False,
        },
    },
}


def _resolve_episode_spec(spec, task):
    """准备本实例专属的固定规格副本（新值注入）；详见 BinFill 同名函数。

    传 ``None``（没传 ``--episode-specs``）时返回 ``None``，此后每个消费点都走原随机路径，
    链路与改动前逐字相同。
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("episode_spec 必须是字典")
    if spec.get("task") != task:
        raise ValueError(f"episode_spec 是 {spec.get('task')} 的规格，不能用于 {task}")
    return copy.deepcopy(spec)


def _bin_index_of(name):
    """把规格里的 ``bin_<i>`` 还原成生成序号 ``i``。"""
    return int(str(name).rsplit("_", 1)[1])


def native_blocks(cls, *, release="newtask-v6"):
    """本环境的 ``(decision, native)`` 原值块；外部导出与内部解析共用同一份，杜绝两套真值。"""
    native = copy.deepcopy(NATIVE_SAMPLING)
    if release in ("newtask-v4", "newtask-v5"):
        legacy_configs = {
            "easy": cls.configs["easy"], "medium": cls.configs["medium"], "hard": cls.configs["hard"],
            "xhard": {"bin": 4, "swap_min": 8, "swap_max": 12, "pick_min": 3, "pick_max": 3},
        }
        native["parameters"]["xhard"] = {"object_selection": {"pickup_selected_indices": [0, 1, 2]}}
        return _legacy_decision(cls, legacy_configs, release), native
    # newtask-v7：与 v6 同一解析路径，取当前类常量（即 V7 定值；v6 值只存于包内 v6 规格 header，0928 方案 R3）
    if release not in ("newtask-v6", "newtask-v7"):
        raise ValueError(f"VideoUnmaskSwap 不支持 sampling_config release {release!r}")
    native["parameters"]["configs"] = copy.deepcopy(cls.configs)
    return _native_decision(cls), native


def _legacy_decision(cls, configs, release):
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
                           else v5_distractor_cfg("VideoUnmaskSwap")),
        },
    }
    if release == "newtask-v5":
        decision["xhard"]["distractor_swap"] = v5_distractor_swap_cfg("VideoUnmaskSwap")
    return decision


def _native_decision(cls):
    """按方案第二节字段表切出 decision 块（原值阶段等于原值）。"""
    # 第二节 2.7：decision 为交换次数范围、交换后拾取数量范围、交换速度倍率与额外干扰物。
    # 原值阶段：次数与拾取数取自 configs，速度倍率为 1（每次交换 50 步），无额外干扰物。
    return {
        "swap_count_range": {
            difficulty: [cfg["swap_min"], cfg["swap_max"]] for difficulty, cfg in cls.configs.items()
        },
        "pick_count_range": {
            difficulty: [cfg["pick_min"], cfg["pick_max"]] for difficulty, cfg in cls.configs.items()
        },
        "swap_speed_multiplier": 1,
        "distractor": None,
        # V4 原难度配置由四个新值档继承；速度和干扰数量按新值档配置，机制沿用原档。
        # V5（2.6，L13/L16 b）：干扰容器改用统一采样器的预设（V4 环带、10 个、含 cube [5,5]）；
        # 新增 distractor_swap：外环随内环同步交换的规则（L17～L23）。
        **{
            tier: {
                "swap_speed_multiplier": 1.0 if newvalue_tier(tier) == 1 else NEWVALUE_SWAP_SPEED_MULTIPLIER,
                "distractor": v6_distractor_cfg("VideoUnmaskSwap", 2 * newvalue_tier(tier)),  # V7：外环 2/4/6/8
                "distractor_swap": v6_distractor_swap_cfg("VideoUnmaskSwap"),
                "swap_plan_v6": v6_inner_swap_plan_cfg("VideoUnmaskSwap"),
            }
            for tier in NEWVALUE_DIFFICULTIES
        },
    }


def _resolve_sampling_config(cls, override):
    """准备本实例专属的采样配置副本；不采样、不改随机流，详见 BinFill 同名函数。"""
    decision_default, native_default = native_blocks(cls, release="newtask-v6")
    decision, native = split_sampling_config(override, native_default, decision_default)
    # 第一轮只做原值导出／消费：decision 必须逐键等于原值（红线 R7）。
    assert_native_decision(decision, decision_default, cls.__name__)
    resolved = native
    resolved["parameters"].setdefault("configs", copy.deepcopy(cls.configs))
    resolved["decision"] = decision
    for key in ("object_selection", "swap_selection"):
        if json.dumps(resolved["parameters"].get(key), sort_keys=True) != json.dumps(NATIVE_SAMPLING["parameters"][key], sort_keys=True):
            raise ValueError(f"VideoUnmaskSwap.parameters.{key} 必须完整保留原版规则与类型")
    return resolved


@register_env("VideoUnmaskSwap", override=True)
class VideoUnmaskSwap(BaseEnv):

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
    # V7 定值（0928 方案 §3.2.2）：swap 5/7/9/11、pick 2/3/3/3、外环 2/4/6/8；容器布局机制沿用原 xhard。
    config_xhard1 = {"bin": 4, "swap_min": 5, "swap_max": 5, "pick_min": 2, "pick_max": 2}
    config_xhard2 = {"bin": 4, "swap_min": 7, "swap_max": 7, "pick_min": 3, "pick_max": 3}
    config_xhard3 = {"bin": 4, "swap_min": 9, "swap_max": 9, "pick_min": 3, "pick_max": 3}
    config_xhard4 = {
        "bin":4,
        "swap_min":11,
        "swap_max":11,
        "pick_min":3,
        "pick_max":3
    }
    # 交换窗口（B4）：首段起点与原三档每段步数；xhard 的每段步数按速度倍率取整（33）
    SWAP_WINDOW_START = SWAP_WINDOW_START
    SWAP_WINDOW_STEPS = SWAP_WINDOW_STEPS


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


    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     sampling_config=None,
                     episode_spec=None,
                     native_episode_spec=None,
                     **kwargs):
        # 必须落在任何随机数调用与 super().__init__() 之前
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._episode_spec = _resolve_episode_spec(episode_spec, "VideoUnmaskSwap")
        self._spec = SpecRecorder(native_episode_spec, "VideoUnmaskSwap", {"seed": seed},
                                  difficulty=kwargs.get("difficulty"))
        # 初始化序号从 -1 起，_initialize_episode 每次进来先加一；
        # _load_scene 里的取值点用不带序号的路径，所以这里只作兜底。
        self._native_init_index = -1
        self._injection_evidence = {}
        # 运行时碰撞与最近邻核验的记录；只有传了规格才写，关闭态恒为空
        self._runtime_checks = []
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

        # Use seed to randomly determine number of repetitions (1-5)
        generator = torch.Generator()
        generator.manual_seed(seed)
        difficulty_cfg = self._sampling["parameters"]["configs"][self.difficulty]
        if self._episode_spec is None:
            self.swap_times = self._spec.value(
                "objects.n_swaps",
                torch.randint(difficulty_cfg['swap_min'], difficulty_cfg['swap_max']+1, (1,), generator=generator).item(),
                decision_key=f"configs.{self.difficulty}.swap_min/swap_max",
            )
            self.pick_times = self._spec.value(
                "objects.n_picks",
                torch.randint(difficulty_cfg['pick_min'], difficulty_cfg['pick_max']+1, (1,), generator=generator).item(),
                decision_key=f"configs.{self.difficulty}.pick_min/pick_max",
            )
        else:
            self.swap_times = int(self._episode_spec["objects"]["n_swaps"])
            self.pick_times = int(self._episode_spec["objects"]["n_picks"])
        logger.debug(f"Task will swap {self.swap_times} times")

        logger.debug(f"Task will pick {self.pick_times} times")

        # 交换窗口（B4）：首段起点恒为 64；每段步数 = round(50 / 倍率)。原三档消费 decision 顶层的
        # swap_speed_multiplier（=1 ⇒ 原样 50），xhard 消费 decision.xhard 的 1.5 ⇒ 33。不抽随机数。
        decision = self._sampling["decision"]
        self._is_newvalue = is_newvalue_difficulty(self.difficulty)
        multiplier = decision[self.difficulty]["swap_speed_multiplier"] if self._is_newvalue else decision["swap_speed_multiplier"]
        self.swap_window_start = self.SWAP_WINDOW_START
        self.swap_window_steps = scaled_window_steps(self.SWAP_WINDOW_STEPS, multiplier)
        # xhard 的乙通道也做运行时碰撞检查（H1：初态＋每段交换的连续扫掠，含干扰容器）；
        # 原三档仍只在甲通道（传了 episode_spec）时检查，行为不变。
        self._newvalue_collision_checks = self._is_newvalue and self._episode_spec is None
        # 干扰容器单独存放，不进 spawned_bins；原三档恒为空
        self.distractor_bins = []
        self.distractor_cubes = []
        if self._is_newvalue:
            # 外环交换对、外环 cube 跟随对、reset 预演的内环对；由 _spawn_newvalue_distractors 填写
            self.distractor_swap_pairs = []
            self.distractor_cube_bin_pairs = []
            self.predicted_inner_swap_pairs = []
            if self._episode_spec is not None:
                raise ValueError("V4 xhard 不支持链路甲的 episode_spec（链路甲已退役）；请走 native_episode_spec")
            self._spec.record(
                "actions.swap_window",
                {"start_step": self.swap_window_start, "duration_steps": self.swap_window_steps,
                 "speed_multiplier": multiplier},
            )

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
        containers_cfg = self._sampling["positions"]["containers"]
        difficulty_cfg = self._sampling["parameters"]["configs"][self.difficulty]
        region4=[list(point) for point in containers_cfg["region4"]]
        region3_tri=[list(point) for point in containers_cfg["region3_tri"]]
        region3_line=[list(point) for point in containers_cfg["region3_line"]]

        spec = self._episode_spec
        if spec is None:
            # Use generator to randomly select region3_tri or region3_line
            choice_cfg = containers_cfg["region3_choice"]
            region3_choice = self._spec.value(
                "layout.type_choice",
                torch.randint(choice_cfg["low"], choice_cfg["high_exclusive"], (1,), generator=generator).item(),
            )
            region3 = region3_tri if region3_choice == 0 else region3_line

            if difficulty_cfg['bin']==4:
                region=region4
            else:
                 region=region3
            angle, region = rotate_points_random(region,tuple(containers_cfg["layout_rotation_range_rad"]),generator)
        else:
            # 规格里的 bins[i].xy 就是**最终**位置（锚点旋转 + 各自偏移在冻结前已经算好并过了
            # 碰撞检查），所以注入路径不再走 rotate_points_random，也就不抽随机数；
            # 这样实跑位置与规格逐值相同，不受 float32 锚点精度影响。
            angle = float(spec["layout"]["theta_rad"])
            region = None

        for i in range(difficulty_cfg['bin']):
            if spec is not None:
                entry = spec["layout"]["bins"][i]
                if entry["object_id"] != f"bin_{i}":
                    raise ValueError(f"规格第 {i} 个容器的 object_id 是 {entry['object_id']}，应为 bin_{i}")
                # 走 spawn_random_bin 末尾同一个 build_bin 调用（含 z=0.002 与 z_rotation_deg），
                # 只是跳过它的拒绝采样
                bin_actor = build_bin(
                    self,
                    callsign=f"bin_{i}",
                    position=[float(entry["xy"][0]), float(entry["xy"][1]), 0.002],
                    z_rotation_deg=float(entry["yaw_deg"]),
                )
            else:
                try:
                    bin_actor = spawn_random_bin(
                        self,
                        avoid=avoid,  # Use current avoidance list, containing all spawned objects
                        region_center=region[i],
                        region_half_size=containers_cfg["region_half_size"],
                        min_gap=self.cube_half_size*1,  # bins need larger gap, increased to 6x to avoid collision
                        name_prefix=f"bin_{i}",
                        max_trials=256,
                        generator=generator,
                        recorder=self._spec,
                        spec_path=f"layout.bins.{i}",
                        yaw_scale_deg=containers_cfg["yaw_scale_deg"]
                    )
                except RuntimeError as e:
                    if self._is_newvalue:
                        # V5（L15 同理扩到 VUS，主会话决定）：xhard 不许静默截断（截断后交换对与环带障碍都会变），
                        # 改抛真异常作候选级重抽；原三档仍 break
                        raise _RealSceneGenerationError(f"xhard 内环容器 bin_{i} 放不下：{e}") from e
                    break

            self.spawned_bins.append(bin_actor)
            # Assign bin to self.bin_0, self.bin_1 etc. attributes
            setattr(self, f"bin_{i}", bin_actor)
            # Add newly generated bin to avoidance list
            avoid.append(bin_actor)


        # Generate 3 dynamic cubes under each bin (use fixed position, colors red, green, blue)
        spawned_dynamic_cubes = []
        self.cube_bin_pairs = []
        self.bin_to_cube = {}
        self.bin_to_color = {}
        self.spawned_dynamic_cubes = spawned_dynamic_cubes
        cube_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # Red, Green, Blue
        color_names = ["red", "green", "blue"]

        # Use seed to randomly shuffle color order

        if spec is None:
            shuffle_indices = self._spec.value(
                "objects.color_order", torch.randperm(len(cube_colors), generator=generator).tolist()
            )
        else:
            # 规格的 color_order 是打乱后的名字序列，这里反解回原定义表 (red, green, blue) 的下标
            shuffle_indices = [color_names.index(name) for name in spec["objects"]["color_order"]]
        cube_colors = [cube_colors[i] for i in shuffle_indices]
        color_names = [color_names[i] for i in shuffle_indices]

        # Store color_names for RecordWrapper access
        self.color_names = color_names

        # Randomly select 3 bins from all bins to generate cube
        selection_cfg = self._sampling["parameters"]["object_selection"]
        num_bins_to_select = min(selection_cfg["hidden_bin_count_max"], len(self.spawned_bins))
        # V6 M5(b)：藏 cube 仍只从前三个容器里抽，bin_3 恒空；使用 native 的 hidden_bin_permutation_size=3。
        if spec is None:
            selected_bin_indices = self._spec.value(
                "objects.selected",
                torch.randperm(selection_cfg["hidden_bin_permutation_size"], generator=generator)[:num_bins_to_select].tolist(),
            )
        else:
            selected_bin_indices = [int(v) for v in spec["objects"]["selected"]]
        if self._is_newvalue:
            selected_bin_indices = validate_hidden_bin_selection(
                selected_bin_indices, permutation_size=selection_cfg["hidden_bin_permutation_size"])
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
                half_size=self.cube_half_size/1.2,
                color=cube_colors[i],  # Use red, green, blue in order
                name_prefix=f"target_cube_{color_names[i]}",
                yaw=0.0,  # No rotation
                dynamic=True
            )

            spawned_dynamic_cubes.append(cube_actor)
            # Assign cube to attributes like self.target_cube_red, self.target_cube_green, etc.
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
            target_choice = self._spec.value(
                "objects.target_choice",
                int(
                    torch.randint(
                        len(self.cube_bin_pairs),
                        (1,),
                        generator=generator,
                    ).item()
                ),
            )
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
        # target_indices are indices into selected_bin_indices (0, 1, 2)
        if spec is None:
            # 规格存的是整数列表；下游仍按张量用（.item()/.tolist()/torch.cat），所以包回张量
            target_indices = torch.tensor(self._spec.value(
                "objects.swap_initiator_indices",
                torch.randperm(len(selected_bin_indices), generator=generator)[:selection_cfg["swap_seed_target_count"]].tolist(),
            ))
        else:
            # ⚠ 原代码把「藏物排序里的局部位置」直接当生成序号用（U5 的索引混用），这里按原行为
            # 保留：规格的 swap_initiators 存的就是这些数，前两个当 target_indices，第三个当 third。
            spec_initiators = [_bin_index_of(name) for name in spec["objects"]["swap_initiators"]]
            target_indices = torch.tensor(spec_initiators[:selection_cfg["swap_seed_target_count"]])
        # Use selected_bins to get correct bin (corresponding to color_names order)
        self.target_bin_1=self.selected_bins[target_indices[0]]
        self.target_bin_2=self.selected_bins[target_indices[1]]
        # Record cube colors for these two bins, using color_names direct indexing
        self.target_bin_1_cube_color = color_names[target_indices[0].item()]
        self.target_bin_2_cube_color = color_names[target_indices[1].item()]
        # swap_indices must include target_indices, then select 1 from remaining indices
        remaining_indices = [i for i in range(len(self.spawned_bins)) if i not in target_indices.tolist()]
        # ⚠ 原随机分支保持为 third_idx 的**第一个**赋值语句：当初的旧测试按「第一个同名赋值」
        # 抽表达式对拍历史版本（旧测试已删除），保持这一顺序便于与历史源码逐行对照。
        if spec is None and remaining_indices:
            third_idx = self._spec.value(
                "objects.swap_initiator_third",
                remaining_indices[torch.randint(0, len(remaining_indices), (1,), generator=generator).item()],
            )
            swap_indices = torch.cat([target_indices, torch.tensor([third_idx])])
        elif spec is not None:
            third_idx = spec_initiators[selection_cfg["swap_seed_target_count"]]
            if third_idx not in remaining_indices:
                raise ValueError(f"规格的第三个交换发起者 bin_{third_idx} 不在其余生成序号 {remaining_indices} 里")
            swap_indices = torch.cat([target_indices, torch.tensor([third_idx])])
        else:
            swap_indices = target_indices
        self.swap_pair1_idx1=self.spawned_bins[swap_indices[0]]
        self.swap_pair2_idx1=self.spawned_bins[swap_indices[1]]
        self.swap_pair3_idx1=self.spawned_bins[swap_indices[2]]
        self.swap_pair1_idx2=None
        self.swap_pair2_idx2=None
        self.swap_pair3_idx2=None
        # xhard（swap 4～5 次）：第 k 次发起者循环沿用前 3 个（a,b,c,a,b）；swap ≤ 3 次时本循环不执行
        for k in range(3, self.swap_times):
            setattr(self, f"swap_pair{k+1}_idx1", self.spawned_bins[swap_indices[k % 3]])
            setattr(self, f"swap_pair{k+1}_idx2", None)
        if self._is_newvalue:
            # V6（计划 2.2 内环 S5）：旧发起者取值点照旧抽，随后主流追加一次规划种子，
            # 用 S5 预规划整段 (发起者, 搭档) 并覆盖 swap_pair{k}_idx1；搭档由 step 的新值分支读预规划
            self._plan_inner_swaps_v6(generator)
        self._refresh_swap_schedule()

        if spec is not None:
            # 只读证据：创建输入 vs 创建后 actor 实际位姿，供 INJECTION_BINDING 核对。
            # 只读位姿，不改状态、不抽随机数。
            self._injection_evidence = {
                "spec_sha256": spec.get("spec_sha256"),
                "episode": spec.get("episode"),
                "theta_rad": float(spec["layout"]["theta_rad"]),
                "layout_type": spec["layout"]["type"],
                "n_swaps": self.swap_times,
                "n_picks": self.pick_times,
                "selected": list(selected_bin_indices),
                "color_names": list(color_names),
                # 这三个是循环基：第 k 次的实际发起者是 swap_initiators[k mod 3]，不在证据里展开（与规格的 3 个一一对应）
                "swap_initiators": [self.spawned_bins.index(getattr(self, f"swap_pair{k}_idx1")) for k in (1, 2, 3)
                                    if getattr(self, f"swap_pair{k}_idx1", None) is not None],
                "bins": [
                    {
                        "object_id": entry["object_id"],
                        "requested_xy": [float(v) for v in entry["xy"]],
                        "requested_yaw_deg": float(entry["yaw_deg"]),
                        "actual_p": [float(v) for v in self._get_actor_position(actor)[:3]],
                    }
                    for entry, actor in zip(spec["layout"]["bins"], self.spawned_bins)
                ],
            }

        pickup_indices = selection_cfg["pickup_selected_indices"]
        if self._is_newvalue:
            # pick 数量由 native.parameters.configs[档] 取值；按抽到的数量依次抓前三个藏物容器。
            pickup_indices = list(range(self.pick_times))
        tasks = [
             {
                        "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=self.swap_schedule[-1][3]),
                        "name": "static",
                        "subgoal_segment": "static",
                        "choice_label": "static",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_hold_obj(env, planner, static_steps=self.swap_schedule[-1][3]),
                        },

            
            {
                "func": (lambda: is_bin_pickup(self, obj=self.selected_bins[pickup_indices[0]])),
                "name": f"pick up the container that hides the {self.color_names[pickup_indices[0]]} cube",
                "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[pickup_indices[0]]} cube",
                "choice_label": "pick up the container",
                "demonstration": False,
                "failure_func": lambda: is_any_bin_pickup(self, [bin for bin in self.spawned_bins if bin != self.selected_bins[pickup_indices[0]]]),
                "solve": lambda env, planner: solve_pickup_bin(env, planner, obj=self.selected_bins[pickup_indices[0]]),
                "segment":self.selected_bins[pickup_indices[0]],
            },
        ]
        if self.pick_times==2:
            tasks.append({
                    "func": (lambda: is_bin_putdown(self, obj=self.selected_bins[pickup_indices[0]])),
                    "name": "put down the container",
                    "subgoal_segment":"put down the container",
                    "choice_label": "put down the container",
                    "demonstration": False,
                    "failure_func": lambda:is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.selected_bins[pickup_indices[0]]]),
                    "solve": lambda env, planner: solve_putdown_whenhold(env, planner,),

                })
            tasks.append(
                {
                    "func": (lambda: is_bin_pickup(self, obj=self.selected_bins[pickup_indices[1]])),
                    "name": f"pick up the container that hides the {self.color_names[pickup_indices[1]]} cube",
                    "subgoal_segment":f"pick up the container at <> that hides the {self.color_names[pickup_indices[1]]} cube",
                    "choice_label": "pick up the container",
                    "demonstration": False,
                    "failure_func": lambda: is_any_bin_pickup(self,[bin for bin in self.spawned_bins if bin != self.selected_bins[pickup_indices[1]]]),
                    "solve": lambda env, planner: solve_pickup_bin(env, planner, obj=self.selected_bins[pickup_indices[1]]),
                    "segment":self.selected_bins[pickup_indices[1]],
                })
        elif self._is_newvalue and self.pick_times > 2:
            # V4 xhard：原分支是 `== 2` 严格相等，pick=3 会落到只抓一次；这里按 pick_times 循环，
            # 每一抓前先放下上一个容器。lambda 用默认参数绑定本轮序号，避免闭包晚绑定。
            if self.pick_times > len(pickup_indices):
                raise ValueError(f"pick_times={self.pick_times} 超过可抓序号 {pickup_indices}")
            for j in range(1, self.pick_times):
                prev_bin = self.selected_bins[pickup_indices[j - 1]]
                cur_bin = self.selected_bins[pickup_indices[j]]
                cur_color = self.color_names[pickup_indices[j]]
                tasks.append({
                    "func": (lambda prev_bin=prev_bin: is_bin_putdown(self, obj=prev_bin)),
                    "name": "put down the container",
                    "subgoal_segment": "put down the container",
                    "choice_label": "put down the container",
                    "demonstration": False,
                    "failure_func": lambda prev_bin=prev_bin: is_any_bin_pickup(self, [bin for bin in self.spawned_bins if bin != prev_bin]),
                    "solve": lambda env, planner: solve_putdown_whenhold(env, planner,),
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
        if self._is_newvalue:
            # V4 xhard：交换全部发生在首个「static」任务的等待里。xhard 乙通道打开了扫掠检查（H1），
            # 原 solve_hold_obj 的裸 except 会吞掉 step 抛出的 BinCollisionError 并死循环；
            # 这里整体替换为只吞 AttributeError 的专用等待（solve_hold_obj_xhard），原三档仍用原函数。
            tasks[0]["solve"] = lambda env, planner: solve_hold_obj_xhard(env, planner, static_steps=self.swap_schedule[-1][3])

        # Store task list for RecordWrapper use
        self.task_list = tasks

        # Record pickup related task indices and items for recovery
        self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
        if self.robomme_failure_recovery:
            # Only inject an intentional failed grasp when recovery mode is enabled
            self.fail_grasp_task_index = inject_fail_grasp(
                self.task_list,
                generator=generator,
                mode=self.robomme_failure_recovery_mode,
            )
        else:
            self.fail_grasp_task_index = None

        if self._is_newvalue:
            # V4 xhard 干扰容器：放在全部既有取值点之后，且走专用随机流（N5）
            self._spawn_newvalue_distractors()
            # V4 xhard（用户 2026-09-22「误抓即失败」）：每个已有 failure_func 的抓取／放下任务追加
            # 「任一干扰容器被抬起（z>0.15，与区域内容器同一判据）即失败」
            add_distractor_misgrasp_failure(self, self.task_list)

    def _plan_inner_swaps_v6(self, generator):
        """V6 xhard 内环 S5 reset 预规划（见 ``unmask_swap_xhard.plan_inner_swaps_v6``）；G 不连通抛真 ``SceneGenerationError``。"""
        plan = plan_inner_swaps_v6(self, generator)
        self._newvalue_inner_plan = plan
        self._newvalue_swap_partners = [int(b) for _a, b in plan["pairs"]]
        for k, (a, _b) in enumerate(plan["pairs"]):
            setattr(self, f"swap_pair{k+1}_idx1", self.spawned_bins[int(a)])
            setattr(self, f"swap_pair{k+1}_idx2", None)

    def _newvalue_planned_partner(self, sweep_index, initiator):
        """第 ``sweep_index`` 次交换由 S5 规划的搭档 actor（``step`` 的新值分支调用，仿 VideoRepick）。"""
        partners = getattr(self, "_newvalue_swap_partners", None)
        if partners is None or sweep_index >= len(partners):
            raise SpecBindingError(f"{self.difficulty}: 第 {sweep_index} 次交换没有 reset 规划的搭档")
        partner = self.spawned_bins[partners[sweep_index]]
        if partner is initiator:
            raise SpecBindingError(f"{self.difficulty}: 第 {sweep_index} 次交换规划的搭档与发起者是同一个容器")
        return partner

    def _spawn_newvalue_distractors(self):
        """新值档外环干扰容器（统一采样器）+ 外环随内环同步交换的 reset 规划。

        内环对内环扫掠预判（L20）→ 放置（H1：候选与预演的内环扫掠相交即拒）→ 外环规划（L17～L21），某窗全不可行
        整段重抽，最多 16 次；任一环节失败抛真 ``SceneGenerationError``（候选级重抽）。全部抽样只走独立流，
        主流一次不多抽；干扰容器仍不进 ``spawned_bins``、不用 ``bin_<i>`` 命名。
        """
        axes = self._sampling["parameters"]["swap_selection"]["partner"]["position_axes"]
        result = spawn_swap_distractors_v5(
            self,
            generator=distractor_generator(self.seed),
            partner_axes=axes,
            hidden_half_size=self.cube_half_size / 1.2,  # 与内环被藏 cube 同尺寸（_load_scene 里的 /1.2）
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
            if self._episode_spec is not None:
                # 初始化复核：读实际碰撞盒查全部容器两两之间。构造期内部 reset 与外层
                # record_env.reset() 各会走一次，两次都从同一份规格重建，不复用旧 actor。
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
            elif getattr(self, "_newvalue_collision_checks", False):
                # V4 xhard 乙通道的初态复核（H1）：容器＋干扰容器两两读真实碰撞盒
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


    def _verify_swap_binding(self, sweep_index, initiator, resolved_partner):
        """核验规格预写的搭档与执行时算出的实际最近邻一致；不符即失败，禁止换搭档。"""
        pairs = self._episode_spec["actions"]["swap_pairs"]
        if sweep_index >= len(pairs):
            raise SpecBindingError(
                f"第 {sweep_index} 段交换在规格里没有对应条目（规格只有 {len(pairs)} 段）"
            )
        expected = pairs[sweep_index]
        actual_initiator = self.spawned_bins.index(initiator)
        actual_partner = self.spawned_bins.index(resolved_partner)
        # 用同一份扫描语义独立复算一次，顺带拿到完整距离表当证据
        reference = self._get_actor_position(initiator)
        candidates = [
            (index, self._get_actor_position(actor))
            for index, actor in enumerate(self.spawned_bins)
            if actor is not None and actor is not initiator
        ]
        recomputed, table = nearest_partner_index(reference, candidates)
        detail = {
            "sweep_index": sweep_index,
            "control_step": int(self.elapsed_steps),
            "expected_initiator": expected["initiator"],
            "expected_partner": expected["partner"],
            "actual_initiator": f"bin_{actual_initiator}",
            "actual_partner": f"bin_{actual_partner}",
            "recomputed_partner": f"bin_{recomputed}",
            "distances": [[f"bin_{index}", dist] for index, dist in table],
        }
        self._runtime_checks.append({"kind": "swap_binding", **detail})
        if expected["initiator"] != f"bin_{actual_initiator}":
            raise SpecBindingError(
                f"第 {sweep_index} 段的发起者是 bin_{actual_initiator}，规格预写 {expected['initiator']}", detail
            )
        if expected["partner"] != f"bin_{actual_partner}":
            raise SpecBindingError(
                f"第 {sweep_index} 段的实际最近邻是 bin_{actual_partner}，规格预写 {expected['partner']}", detail
            )

    def _object_states_for_collision(self):
        """把场上全部容器读成碰撞判据用的状态；只读真实碰撞盒，不用中心距或外接圆。

        V4（H1）：干扰容器显式并入；原三档 ``distractor_bins`` 恒为空，结果与改动前相同。
        """
        return [
            object_state_from_actor(actor, f"bin_{index}")
            for index, actor in enumerate(self.spawned_bins)
            if actor is not None
        ] + [
            object_state_from_actor(actor, f"distractor_bin_{index}")
            for index, actor in enumerate(getattr(self, "distractor_bins", []))
            if actor is not None
        ]

    def _check_swap_sweep_from_actual(self, sweep_index, initiator, partner):
        """从实际位姿对整段交换路径做连续检查；命中即抛错中止该样本。"""
        if getattr(self, "_is_newvalue", False):
            # V5 xhard（2.6）：内环对（运行时解析，L22 a）与本窗外环对的两对联合复核（带认证预筛，L23）
            self._check_joint_sweep_xhard(sweep_index, initiator, partner)
            return
        states = {}
        for index, actor in enumerate(self.spawned_bins):
            if actor is None:
                continue
            states[index] = object_state_from_actor(actor, f"bin_{index}")
        a = self.spawned_bins.index(initiator)
        b = self.spawned_bins.index(partner)
        bystanders = [state for index, state in sorted(states.items()) if index not in (a, b)]
        # V4（H1）：干扰容器作为静止旁观者并入连续扫掠检查；原三档为空列表，行为不变
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

    def _check_joint_sweep_xhard(self, sweep_index, initiator, partner):
        """V5 xhard：两对联合连续复核（实际位姿、真实碰撞盒）；内环搭档与 reset 预演不一致时留日志（L22 a）。"""
        gap, rejection, info = joint_sweep_from_actual(self, sweep_index, initiator, partner)
        if info["inner_partner_mismatch"]:
            logger.warning(
                f"VideoUnmaskSwap xhard 第 {sweep_index} 段内环对 {info['inner_pair']} 与 reset 预演 "
                f"{info['predicted_inner_pair']} 不一致（运行时照常按实际最近邻交换，联合复核用实际对）"
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

    def _check_state_readonly(self, stage):
        """某一时刻的只读复核；返回拒绝证据，不抛错，由调用方决定怎么处置。"""
        gap, rejection = check_bin_state(self._object_states_for_collision(), stage=stage)
        return gap, rejection

    def _in_swap_window(self):
        """当前控制步是否落在某一段交换的时间窗内。

        ⚠ 子步检查只在窗口内做。窗口外容器／方块要么静止、要么被
        ``lift_and_drop_objects_back_to_original`` 逐帧按在原位，初态检查已经覆盖；
        而子步数远多于控制步，全程每个子步都跑一遍全量 SAT 会把单条样本拖慢十倍以上
        （实测冒烟时 VideoUnmaskSwap 从 ~47 秒涨到分钟级）。
        """
        step = int(self.elapsed_steps)
        return any(start <= step <= end for _a, _b, start, end in getattr(self, "swap_schedule", []))

    def _before_simulation_step(self):
        super()._before_simulation_step()
        if self._episode_spec is None or not self._in_swap_window():
            return
        gap, rejection = self._check_state_readonly("substep_before")
        if rejection is not None:
            self._runtime_checks.append(
                {"kind": "substep_before", "control_step": int(self.elapsed_steps), "rejection": rejection.as_dict()}
            )
            raise BinCollisionError(rejection)

    def _after_simulation_step(self):
        super()._after_simulation_step()
        if self._episode_spec is None or not self._in_swap_window():
            return
        gap, rejection = self._check_state_readonly("substep_after")
        if rejection is not None:
            self._runtime_checks.append(
                {"kind": "substep_after", "control_step": int(self.elapsed_steps), "rejection": rejection.as_dict()}
            )
            raise BinCollisionError(rejection)

    def _refresh_swap_schedule(self):
        # 通式：第 k 次 swap 占 [S+Lk, S+L(k+1)]，首尾相接；S = swap_window_start（恒 64），
        # L = swap_window_steps（原三档 50、xhard 33）。1/2/3 次时与原三分支逐项相同，0 次时不赋值
        if self.swap_times < 1:
            return
        start, length = self.swap_window_start, self.swap_window_steps
        self.swap_schedule = [
            (getattr(self, f"swap_pair{k+1}_idx1"), getattr(self, f"swap_pair{k+1}_idx2"), start + length * k, start + length * (k + 1))
            for k in range(self.swap_times)
        ]



#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


        timestep = self.elapsed_steps        
        if self._is_newvalue:
            # V5 xhard（L14，主会话定内环也用）：内环容器与外环干扰容器同一窗口 [0, 64)、同一时间线揭示，
            # 但每个物体停在各自的画面外停放点，不再全部叠在 (10,10,10)；干扰容器仍不进 spawned_bins
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
                    end_step=self.swap_window_start,  # 预交换锁定段终点 = 首段交换起点（64）
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
                        # V6 xhard（计划 2.2 内环 3）：搭档读 S5 reset 预规划，不再按实际 XY 取最近邻；
                        # 下面乙通道的两对联合复核照旧跑，作运行时守卫
                        closest_actor = self._newvalue_planned_partner(i, pair_idx1)
                    else:
                        reference_pos = self._get_actor_position(pair_idx1)
                        closest_actor = None
                        closest_dist = float("inf")
                        for candidate in self.spawned_bins:
                            if candidate is None or candidate is pair_idx1:
                                continue
                            candidate_pos = self._get_actor_position(candidate)
                            axes = self._sampling["parameters"]["swap_selection"]["partner"]["position_axes"]
                            dist = np.linalg.norm(reference_pos[axes] - candidate_pos[axes])
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_actor = candidate
                    if closest_actor is not None:
                        # 新值注入的两个运行时检查点（计划第五节步骤 0d）：先核验搭档身份，
                        # 再从**实际**起态做整段连续几何检查。关闭态（无规格）两项都不跑，
                        # 不新增几何读取、不新增随机调用。
                        if self._episode_spec is not None:
                            self._verify_swap_binding(i, pair_idx1, closest_actor)
                            self._check_swap_sweep_from_actual(i, pair_idx1, closest_actor)
                        elif getattr(self, "_newvalue_collision_checks", False):
                            # V4 xhard 乙通道（H1）：没有预写搭档可核，只做含干扰容器的连续扫掠检查
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
            # V5 xhard（2.6，口径 4）：外环与内环同窗口交换，写在 AST 锁定的内环搭档循环之外，不增加任何控制步
            run_outer_swaps(self, timestep)
            # 内环与外环被藏 cube 在 [64, last_end) 各停独立点，last_end 那一步落到各自容器最终 XY（L14）
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
