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


# ── decision／native 两块的原值（newtaskRelease-v3 步 3，映射见方案第二节 2.8）────────
NATIVE_SAMPLING = {
    "parameters": {
        "bin_count": "FROM_CLASS_CONFIGS",
        "color_pool": [
            {"rgba": [1, 0, 0, 1], "name": "red"},
            {"rgba": [0, 1, 0, 1], "name": "green"},
            {"rgba": [0, 0, 1, 1], "name": "blue"},
        ],
        "color_order": {"sampler": "torch.randperm(3)"},
        "hidden_rule": "前三容器藏三色，第四个为空",
        "pick_rule": "左按钮→右按钮后（机器人坐标系，左 = +y），抓 selected_bins[0]，count=2 时再抓 [1]",
        "partner_rule": "交换开始时按实际 XY 取最近邻，不另抽签",
        # 交换窗口：首段起点与原三档每段步数（具名常量，六处原字面量都改读这里）；
        # V4 起真正被消费，xhard 的每段步数 = round(duration_steps / 1.5) = 33（2.11）
        "swap_window": {"start_step": SWAP_WINDOW_START, "duration_steps": SWAP_WINDOW_STEPS},
        "swap_path": {"lane_offset": 0.07, "smooth": True, "keep_upright": True},
        # V6 审查修复 F3（用户 K4「所有的左右都是机器人坐标系」）：按钮命名对齐机器人坐标系（机器人朝 +x，左 = +y）；
        # 建构顺序、位置与随机数消费全部不变，只改 name 与 button_order 的字面：buttons[0]（y=-0.1）是右按钮，buttons[1]（y=+0.1）是左按钮，
        # 任务链仍先按 buttons[1]（左）再按 buttons[0]（右），与改名前按的物理按钮相同。
        "button_order": ["left", "right"],
        "recovery": "构造器的 self.generator 用于恢复；场景另建同 seed 局部流，两条流分开",
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
            "offset_note": "四点按两组各抽一次 y 偏移；三点各自抽一次 x 偏移；"
                            "未被选中的那一套分支仍然照常消费随机数（红线 R8）",
            "choice_sampler": "torch.randint(0, 2)",
        },
        "bins": {"region_half_size": 0.07, "min_gap_factor": 1, "max_trials": 256},
        "hidden_cube": {"half_size_divisor": 1.2, "yaw": 0.0},
    },
}


def native_blocks(cls, *, release="newtask-v6"):
    """本环境的 ``(decision, native)`` 原值块；外部导出与内部解析共用同一份。"""
    native = copy.deepcopy(NATIVE_SAMPLING)
    if release in ("newtask-v4", "newtask-v5"):
        # V4/V5 快照冻结的是改名前的按钮命名（F3 之前：buttons[0] 名 button_left、顺序 ["right","left"]）；
        # 导出旧 release 时按冻结值回填，保证 V5 快照可逐字节复现。物理对象与任务链顺序两版相同。
        native["parameters"]["button_order"] = ["right", "left"]
        native["parameters"]["pick_rule"] = "右按钮→左按钮后，抓 selected_bins[0]，count=2 时再抓 [1]"
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
    # newtask-v7：与 v6 同一解析路径，取当前类常量（即 V7 定值；v6 值只存于包内 v6 规格 header，0928 方案 R3）
    if release not in ("newtask-v6", "newtask-v7"):
        raise ValueError(f"ButtonUnmaskSwap 不支持 sampling_config release {release!r}")
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
    """按方案第二节 2.8 切出 decision 块（原值阶段等于原值）。"""
    return {
        "swap_count_range": {
            difficulty: [cfg["swap_min"], cfg["swap_max"]] for difficulty, cfg in cls.configs.items()
        },
        "pick_count_range": {
            difficulty: [cfg["pick_min"], cfg["pick_max"]] for difficulty, cfg in cls.configs.items()
        },
        # 交换速度倍率：原值 1（每段 50 步），原三档消费它（=1 ⇒ 原样 50 步）。
        "swap_speed_multiplier": 1,
        "distractor": None,
        # 新值档使用逐档速度与外环数量；容器布局机制沿用原档。
        # V5（2.7，L13/L16 b）：干扰容器改用统一采样器的预设（V4 环带、10 个、含 cube [5,5]）；
        # 新增 distractor_swap：外环随内环同步交换的规则（L17～L23，外环路径离按钮中心 ≥ 0.122）。
        **{
            tier: {
                "swap_speed_multiplier": 1.0 if newvalue_tier(tier) == 1 else NEWVALUE_SWAP_SPEED_MULTIPLIER,
                "distractor": v6_distractor_cfg("ButtonUnmaskSwap", 2 * newvalue_tier(tier)),  # V7：外环 2/4/6/8
                "distractor_swap": v6_distractor_swap_cfg("ButtonUnmaskSwap"),
                "swap_plan_v6": v6_inner_swap_plan_cfg("ButtonUnmaskSwap"),
            }
            for tier in NEWVALUE_DIFFICULTIES
        },
    }


def _resolve_sampling_config(cls, override):
    """拆出本实例专属的 decision／native 副本；不抽随机数，必须在 Generator 之前调用。"""
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


    # V7 定值（0928 方案 §3.2.2）：swap 3/5/7/9、pick 2/3/3/3、外环 2/4/6/8；内部容器布局机制沿用原 xhard。
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
    # 交换窗口的具名常量（B4）；运行时读 native.swap_window（默认值即这两个常量）
    SWAP_WINDOW_START = SWAP_WINDOW_START
    SWAP_WINDOW_STEPS = SWAP_WINDOW_STEPS
    

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     sampling_config=None,
                     native_episode_spec=None,
                     **kwargs):
        # 必须落在任何随机数调用与 super().__init__() 之前
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "ButtonUnmaskSwap", {"seed": seed},
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

        # 交换窗口（2.11）：native.swap_window 真正被消费；原三档倍率取 decision 顶层的 1 ⇒ 原样 {64, 50}，
        # 新值档按逐档倍率消费：第一档 50 步，其余三档 33 步。不抽随机数。
        window = self._sampling["parameters"]["swap_window"]
        multiplier = decision[self.difficulty]["swap_speed_multiplier"] if self._is_newvalue else decision["swap_speed_multiplier"]
        self.swap_window_start = int(window["start_step"])
        self.swap_window_steps = scaled_window_steps(int(window["duration_steps"]), multiplier)
        # xhard 做运行时碰撞检查（H1：初态＋每段交换的连续扫掠，含干扰容器）；原三档不检查，行为不变
        self._newvalue_collision_checks = self._is_newvalue
        self._runtime_checks = []
        # 干扰容器单独存放，不进 spawned_bins；原三档恒为空
        self.distractor_bins = []
        self.distractor_cubes = []
        if self._is_newvalue:
            # 外环交换对、外环 cube 跟随对、reset 预演的内环对；由 _spawn_newvalue_distractors 填写
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
        # 通式（与 VideoUnmaskSwap 同）：第 k 次 swap 占 [S+Lk, S+L(k+1)]，首尾相接；
        # S = swap_window_start（恒 64），L = swap_window_steps（原三档 50、xhard 33）。
        # 1/2/3 次时与原三分支逐项相同；原三分支对 ≥4 次不赋值（step 里 AttributeError），通式一并修好。
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
        # F3：buttons[0]（y=-0.1）在机器人坐标系里是右按钮；cap link 按对象保存，不再按配置里的名字查
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
        # Store second button（buttons[1]，y=+0.1，机器人坐标系左按钮）
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
                    # V5 L15：xhard 不许静默截断（截断后交换对与环带障碍都会变），改抛真异常作候选级重抽
                    raise _RealSceneGenerationError(f"xhard 内环容器 bin_{i} 放不下：{e}") from e
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
        # M5(b)：仍只从前三个容器里选藏物位置，第四个容器 bin_3 恒空。
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
        # 规格存整数列表，下游仍按张量用（.item()/.tolist()/torch.cat），所以包回张量
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
        # V4 xhard（swap 6～8 次）：第 k 次发起者循环沿用前 3 个（a,b,c,a,b,…）；swap ≤ 3 次时本循环不执行
        for k in range(3, self.swap_times):
            setattr(self, f"swap_pair{k+1}_idx1", self.spawned_bins[swap_indices[k % 3]])
            setattr(self, f"swap_pair{k+1}_idx2", None)
        if self._is_newvalue:
            # V6（计划 2.2 内环 S5）：V5 发起者取值点照旧抽，随后主流追加一次规划种子，S5 预规划整段并覆盖发起者；
            # 搭档由 step 的 xhard 分支读预规划
            self._plan_inner_swaps_v6(generator)


        self._refresh_swap_schedule()

        self.button_list= [self.button_right, self.button_left]  # F3：顺序保持 [buttons[0] 对象, buttons[1] 对象]
        self.generator=generator

        if self._is_newvalue:
            # V4 xhard 干扰容器：走专用随机流，主流（含 _initialize_episode 里 inject_fail_grasp
            # 继续消费的 self.generator）一次都不多抽（N5）
            self._spawn_newvalue_distractors([button_obb_1, button_obb_2])

    def _plan_inner_swaps_v6(self, generator):
        """V6 xhard 内环 S5 reset 预规划（见 ``unmask_swap_xhard.plan_inner_swaps_v6``）；G 不连通抛真 ``SceneGenerationError``。"""
        plan = plan_inner_swaps_v6(self, generator)
        self._newvalue_inner_plan = plan
        self._newvalue_swap_partners = [int(b) for _a, b in plan["pairs"]]
        for k, (a, _b) in enumerate(plan["pairs"]):
            setattr(self, f"swap_pair{k+1}_idx1", self.spawned_bins[int(a)])
            setattr(self, f"swap_pair{k+1}_idx2", None)

    def _newvalue_planned_partner(self, sweep_index, initiator):
        """第 ``sweep_index`` 次交换由 S5 规划的搭档 actor（``step`` 的新值分支调用）。"""
        partners = getattr(self, "_newvalue_swap_partners", None)
        if partners is None or sweep_index >= len(partners):
            raise SpecBindingError(f"{self.difficulty}: 第 {sweep_index} 次交换没有 reset 规划的搭档")
        partner = self.spawned_bins[partners[sweep_index]]
        if partner is initiator:
            raise SpecBindingError(f"{self.difficulty}: 第 {sweep_index} 次交换规划的搭档与发起者是同一个容器")
        return partner

    def _spawn_newvalue_distractors(self, button_obbs):
        """新值档外环干扰容器（统一采样器，按钮 OBB 精确进障碍）+ 外环随内环同步交换的 reset 规划。

        与 VideoUnmaskSwap 同构（L20 预判 → 放置 + H1 → 外环规划，最多 16 次整段重抽），另加外环路径离两个按钮中心
        ≥ 0.122（L19）。全部抽样只走独立流，主流（含 _initialize_episode 里 inject_fail_grasp 继续消费的
        self.generator）一次都不多抽；干扰容器仍不进 spawned_bins（本环境没有 _verify_swap_binding，混进去会静默改交换对）。
        """
        result = spawn_swap_distractors_v5(
            self,
            generator=distractor_generator(self.seed),
            # 本环境的内环最近邻是硬编码 [:2]（XY），预演用同一轴
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
                # V4 xhard 的初态复核（H1）：容器＋干扰容器两两读真实碰撞盒
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
            # V4 xhard：本环境全部任务 demonstration=False，按完两个按钮通常只到第 200 步上下，
            # 而 6~8 次交换要到 64+33n（262~328）才结束；原解法会在容器还在交换时就去抓（实测 2/2 失败）。
            # V6 审查修复 N1（用户「n1 subgoal定为wait」）：等待不再藏在第二按钮的解法里（那样等待期间子目标仍标
            # 「press the second button」），而是作为独立子目标「wait for the containers to finish swapping」插在
            # 第二按钮之后：完成判定 = 已到交换时间表的最后一段结束步；解法 = 原地等到该绝对步。
            # 仍不放在第一抓上，因为 inject_fail_grasp 会整个替换被选中抓取任务的 solve。
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
            # V4 xhard：原分支是 `== 2` 严格相等，pick=3 会落到只抓一次；这里按 pick_times 循环抓
            # selected_bins[0..pick_times-1]，每一抓前先放下上一个。lambda 用默认参数绑定本轮对象。
            if self.pick_times > len(self.selected_bins):
                raise ValueError(f"pick_times={self.pick_times} 超过藏物容器数 {len(self.selected_bins)}")
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
            # V4 xhard（用户 2026-09-22「误抓即失败」）：每个已有 failure_func 的抓取／放下任务追加
            # 「任一干扰容器被抬起（z>0.15，与区域内容器同一判据）即失败」
            add_distractor_misgrasp_failure(self, self.task_list)
            # V6 审查修复 F4（K5）：给内环容器打标——子目标未切换但目标分割中心位移超过 8 像素时，
            # process_segmentation 重算 grounded 坐标（交换后不再沿用旧位置）。未打标的 actor（原三档、其他环境）走原分支。
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
        """V6 N1：等待子目标的解法——原地等到最后一段交换结束（绝对步 swap_schedule[-1][3]）；已过则立即返回。"""
        solve_hold_obj_absTimestep(env, planner, absTimestep=self.swap_schedule[-1][3])
        return None

    def _solve_press_then_wait_swaps(self, env, planner, button):
        """V4 xhard 专用解法：按下按钮后原地等到最后一段交换结束（绝对步 swap_schedule[-1][3]）。

        按钮解法失败（返回 -1）时原样返回，不吞掉失败信号；已过交换结束时刻则立即返回。
        """
        result = solve_button(env, planner, obj=button)
        if isinstance(result, (int, np.integer)) and int(result) == -1:
            return result
        solve_hold_obj_absTimestep(env, planner, absTimestep=self.swap_schedule[-1][3])
        return result

    def _object_states_for_collision(self):
        """V4 xhard（H1）：全部容器＋干扰容器读成碰撞判据用的状态（真实碰撞盒）。"""
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
        """某一时刻的只读复核；返回拒绝证据，不抛错，由调用方决定怎么处置。"""
        return check_bin_state(self._object_states_for_collision(), stage=stage)

    def _check_swap_sweep_from_actual(self, sweep_index, initiator, partner):
        """V4 xhard（H1）：从实际位姿对整段交换路径做连续检查，旁观者含其余容器与全部干扰容器；
        命中即抛 ``BinCollisionError`` 中止该样本。原三档从不调用。

        V5：xhard 改为内环对（运行时解析，L22 a）与本窗外环对的两对联合复核（带认证预筛，L23）；下面 V4 的单对
        分支只在没有 ``_is_newvalue`` 的实例上走（本环境原三档从不调用本函数）。"""
        if getattr(self, "_is_newvalue", False):
            gap, rejection, info = joint_sweep_from_actual(self, sweep_index, initiator, partner)
            if info["inner_partner_mismatch"]:
                logger.warning(
                    f"ButtonUnmaskSwap xhard 第 {sweep_index} 段内环对 {info['inner_pair']} 与 reset 预演 "
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
                        # V6 xhard（计划 2.2 内环 3）：搭档读 S5 reset 预规划；下面的两对联合复核照旧跑
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
                            # V4 xhard（H1）：定下搭档后、开始移动前，做含干扰容器的连续扫掠检查
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
            # V5 xhard（2.7，口径 4）：外环与内环同窗口交换，写在内环搭档循环之外，不增加任何控制步
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
