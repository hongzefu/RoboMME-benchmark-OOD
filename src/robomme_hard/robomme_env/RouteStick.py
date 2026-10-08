


import copy
import json
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

from .utils import *
# V5 L3（仿 VideoPlaceOrder 的 K2 修法）：上一行的 `from .utils import *` 会把同名子模块
# `utils.SceneGenerationError` 盖到名字 `SceneGenerationError` 上（import 自省核实），原三档的
# raise / except 因此是 TypeError（按 H2 原三档保持现状）。xhard 用下面这个别名拿到真正的异常类。
from .utils.SceneGenerationError import SceneGenerationError as _RealSceneGenerationError
from .utils.subgoal_evaluate_func import *
from .utils.object_generation import *
from .utils import reset_panda
from .utils.route import *
from .utils.subgoal_planner_func import *
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


# If direction is reversed, modify evaluate and solve

# ── 原版采样输入的原值快照（newtask-v2 10.0）────────────────────────────────────
# 说明同 BinFill：本字典即不传 sampling_config 时的默认值，也是 --extract-config 的提取目标；
# 难度字典仍以类属性 config_easy / config_medium / config_hard 为准，不在这里重复。
NATIVE_SAMPLING = {
    "parameters": {
        "configs_fallback_difficulty": "easy",
        "walk": {
            "node_indices": [0, 2, 4, 6, 8],
            "start_selection": "randint",
            "neighbor_order": [-1, 1],
            "force_reverse_at_endpoint": True,
            "direction": {
                "sampler": "torch.rand",
                "shape": [1],
                "threshold": 0.5,
                "less_than": "clockwise",
                "otherwise": "counterclockwise",
            },
        },
    },
    "positions": {
        "grid_center": [-0.1, 0],
        "grid_spacing_x": 0.07,
        "grid_spacing_y": 0.07,
        "rotation_center": [0, 0],
        "yaw_deg": {"scale": 60, "subtract": 30},
        "yaw_expression": "math.radians(u * 60 - 30)",
        "grid_spacing_x_effect": "num_rows=1 使 row 偏移恒为 0，该参数对结果无影响（死参数）",
        "target_builder": "build_gray_white_target",
        "obstacle_builder": "_load_scene 内联 create_actor_builder",
        "cylinder_radius": 0.015,
        "cylinder_height": 0.1,
        "obstacle_color": {
            "sampler": "torch.rand",
            "shape": [3],
            "count": 4,
            "position_in_stream": "theta 之后、steps 之前，每根障碍柱一次",
        },
        "walk_start": {
            "sampler": "torch.randint",
            "low": 0,
            "high_exclusive": 5,
            "note": "generate_dynamic_walk 未传 start_idx 时抽取，是路线随机流的第一次抽取",
        },
        # ── 高亮渲染的原值（newtaskRelease-v3 步 2）──────────────────────────────
        # 白球尾迹会被渲进 front/wrist 的 rgb 与 depth，所以它是会影响对拍的观测输入，
        # 不是纯视觉装饰。官方 dataset-gen 的存活期为 40 步；2026-09-12 曾两次减半到
        # 10 步，A↔B 实测正是因此在 200 帧里有 184 帧的四路相机观测不同（动作与状态全同）。
        # 这里恢复官方原值 40，减半方案改为通过 sampling_config 显式覆盖。
        "tcp_trail": {
            "end_offset_steps": 40,
            "disk_radius": 0.005,
            "native_note": "官方 dataset-gen d53f21a 的 highlight_position 尾迹存活 40 步",
        },
        "button_highlight": {
            "end_offset_steps": 40,
            "disk_radius_scale": 1.002,
            "native_note": "官方与现行一致，未改动",
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


def native_blocks(cls):
    """本环境的 ``(decision, native)`` 原值块；外部导出与内部解析共用同一份，杜绝两套真值。"""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


def _native_decision(cls):
    """按方案第二节字段表切出 decision 块（原值阶段等于原值）。"""
    # 第二节 2.16：RouteStick 的 decision 只有「演示视频目标时长及调节时长的方式」，
    # 本轮不启用（值为 None 即保持原路径与求解运动决定时长）；段数 length 与 backtrack
    # 按字段表属 native，随 configs 一起留在 native 块里。
    return {
        "demo_duration_seconds_range": None,
        "demonstration_duration_policy": "native",
        # V5 xhard 专属（计划 2.11，L37/L38）：段数 L 的范围冻进 decision 与规格 header，回放时从 header 读，
        # 不再从类属性读。键名为 xhard，守卫只放行这一子树取新值，原三档可见部分不变。
        "xhard4": {"segment_count_range": list(cls.config_xhard4["length"])},
        # V6（计划 2.12）：追加 xhard1/2/3 三棵同结构子树，值取各档 config 的 length（xhard 保持首位不变）
        # v8：xhard4 写字面值（不再用 NEWVALUE_DIFFICULTIES[-1] 指代最难档）
        **{d: {"segment_count_range": list(cls.configs[d]["length"])}
           for d in NEWVALUE_DIFFICULTIES if d != "xhard4"},
    }


def _resolve_sampling_config(cls, override):
    """准备本实例专属的采样配置副本；不采样、不改随机流，详见 BinFill 同名函数。"""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    # 第一轮只做原值导出／消费：decision 必须逐键等于原值（红线 R7）。
    assert_native_decision(decision, decision_default, cls.__name__)
    # V6：V5 快照（已有顶层 xhard 子树）缺 xhard1/2/3 时从源码补齐；V4 及更早的快照不补，
    # 保持 V5「V4 header 在新值档上直接报错、不静默取源码新值」的口径 13。
    if "xhard4" in decision:
        fill_missing_newvalue(decision, decision_default)
    resolved = native
    resolved["parameters"].setdefault("configs", copy.deepcopy(cls.configs))
    resolved["decision"] = decision
    walk = copy.deepcopy(resolved["parameters"].get("walk"))
    if not isinstance(walk, dict) or not isinstance(walk.get("direction"), dict):
        raise ValueError("RouteStick.parameters.walk 缺少完整游走规则")
    threshold = walk["direction"].get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("RouteStick.walk.direction.threshold 必须为 [0,1] 内有限数值")
    walk["direction"]["threshold"] = NATIVE_SAMPLING["parameters"]["walk"]["direction"]["threshold"]
    if json.dumps(walk, sort_keys=True) != json.dumps(NATIVE_SAMPLING["parameters"]["walk"], sort_keys=True):
        raise ValueError("RouteStick.parameters.walk 除方向阈值外必须完整保留原版规则与类型")
    return resolved


@register_env("RouteStick", override=True)
class RouteStick(BaseEnv):

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
    'length':[2,3],
    'backtrack':False,
    }
    config_medium = {
    'length':[4,5],
    'backtrack':False,
    }
    config_hard = {
    'length':[4,7],
    'backtrack':True,
    }
    # V4 xhard（派生自 hard，B9；A7 作废 2026-09-11 的旧值 [8,10]）：布局不动，段数 12～15。
    # V5（计划 2.11，L37）：段数 L 改为 [15,21]。演示每段恰好 50 帧 ⇒ 750～1050 帧 ⇒ 25～35 s @30fps，
    # 均匀抽样均值 30 s；执行段 50·L（+1 初始帧），L=21 时 1050 步，在评估 1301 步预算内（截断点 L≥27）。
    # 抽样点与顺序不变，只改值域；xhard 实际消费的是 decision.xhard.segment_count_range（冻进 header），
    # 这里的 length 是它的默认来源。
    # V7 定值：段数 10/13/16/19（0928 方案 §3.2.2）
    config_xhard4 = {
    'length':[19,19],
    'backtrack':True,
    }

    # V6（计划 2.12）：hard 与 xhard 之间插入三档，布局与游走规则沿用 xhard，只改段数 L；backtrack 恒 True
    # v8（1001 方案 §1 表 1 / §2.1）：xhard1～3 由定值 10/13/16 改区间 [8,10]/[11,13]/[14,16]，
    # torch.randint(lo, hi+1) 在区间内均匀抽；xhard4 仍 19（不交付）。
    config_xhard1 = {
    'length':[8,10],
    'backtrack':True,
    }
    config_xhard2 = {
    'length':[11,13],
    'backtrack':True,
    }
    config_xhard3 = {
    'length':[14,16],
    'backtrack':True,
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
                     episode_spec=None,
                     native_episode_spec=None,
                     **kwargs):
        # 必须落在任何随机数调用与 super().__init__() 之前
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._episode_spec = _resolve_episode_spec(episode_spec, "RouteStick")
        self._spec = SpecRecorder(native_episode_spec, "RouteStick", {"seed": seed},
                                  difficulty=kwargs.get("difficulty"))
        # 初始化序号从 -1 起，_initialize_episode 每次进来先加一；
        # _load_scene 里的取值点用不带序号的路径，所以这里只作兜底。
        self._native_init_index = -1
        self._injection_evidence = {}
        self.achieved_list=[]
        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
        self.match=False
        self.after_demo=False
        self.current_task_demonstration = False
        self._gripper_xy_trace=[]
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
            self.difficulty = "easy"
               # Use seed to randomly determine number of repetitions (1-5)
        generator = torch.Generator()
        generator.manual_seed(seed)




        self.highlight_starts = {}  # Use dictionary to store highlight start time for each button
        self._first_non_record_step = None  # Start timestep for delayed highlight

        self.z_threshold=0.15
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
        camera_eye=[0.3,0,0.4]
        camera_target =[0,0,-0.2]
        pose = sapien_utils.look_at(
            eye=camera_eye, target=camera_target
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

        # Generate 3x3 grid of buttons（注释与实际不符：原布局是 1 x 9，按实际记录）
        layout_cfg = self._sampling["positions"]
        grid_center = list(layout_cfg["grid_center"])  # Grid center position
        grid_spacing_x = layout_cfg["grid_spacing_x"] # Spacing between buttons
        grid_spacing_y=layout_cfg["grid_spacing_y"]

        self.buttons_grid = []
        self.button_joints_grid = []
        avoid = []
        button_index = 0

        num_rows, num_cols = 1, 9
        row_center = (num_rows - 1) / 2
        col_center = (num_cols - 1) / 2


        spec = self._episode_spec
        yaw_cfg = layout_cfg["yaw_deg"]
        if spec is None:
            theta = math.radians(self._spec.value(
                "layout.rotation_deg",
                (torch.rand(1, generator=generator).item() * yaw_cfg["scale"]) - yaw_cfg["subtract"],
            ))
        else:
            # 整排绕世界原点的旋转角由规格定死；节点位置仍由下面同一段公式推出，
            # 不是创建后再整体挪物体
            theta = math.radians(float(spec["layout"]["rotation_deg"]))
        #theta=0
        for row in range(num_rows):
            for col in range(num_cols):  # Columns (y direction)
                x_pos = grid_center[0] + (row - row_center) * grid_spacing_x
                y_pos = grid_center[1] + (col - col_center) * grid_spacing_y

                orig_x, orig_y = x_pos, y_pos
                x_pos = orig_x * math.cos(theta) - orig_y * math.sin(theta)
                y_pos = orig_x * math.sin(theta) + orig_y * math.cos(theta)

                target_name = f"target_{button_index}"

                # Create rotation quaternion for vertical target
                angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))
                rotate = matrix_to_quaternion(
                    euler_angles_to_matrix(angles, convention="XYZ")
                )

                # Build purple and white target
                raised_indices = {0, 2, 4, 6, 8}
                z_pos = 0.01 if button_index in raised_indices else -0.01
                target = build_gray_white_target(
                    scene=self.scene,
                    radius=0.02,
                    thickness=0.01,
                    name=target_name,
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=sapien.Pose(p=[x_pos, y_pos, z_pos], q=rotate),
                )

                self.buttons_grid.append(target)
                # Note: purple_white_target doesn't have joints, so we append None
                self.button_joints_grid.append(None)
                logger.debug(f"Generated target {button_index} at position ({x_pos:.3f}, {y_pos:.3f})")
                button_index += 1

        self.targets_grid = self.buttons_grid

        # Spawn white cubes on specific targets to create fixed obstacles.
        target_cube_indices = [1, 3, 5,7]
        self.target_cube_indices = target_cube_indices
        self.target_cubes = {}
        self.cubes_on_targets = []

        for obstacle_order, target_idx in enumerate(target_cube_indices):
            if target_idx >= len(self.targets_grid):
                logger.debug(f"[SwingAvoid] Skip cube spawn for target {target_idx}: index out of range.")
                continue

            target_actor = self.targets_grid[target_idx]
            target_pos = (
                target_actor.pose.p
                if hasattr(target_actor, "pose")
                else target_actor.get_pose().p
            )

            if isinstance(target_pos, torch.Tensor):
                target_pos = target_pos.detach().cpu().numpy()

            target_pos = np.asarray(target_pos, dtype=np.float64).reshape(-1)

            cube_position = [float(target_pos[0]), float(target_pos[1])]

            cylinder_radius = layout_cfg["cylinder_radius"]
            cylinder_height = layout_cfg["cylinder_height"]
            cylinder_half_length = cylinder_height / 2.0

            # Keep the cylinder centered so that it stands upright on the table surface.
            cylinder_angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))
            builder = self.scene.create_actor_builder()

            cylinder_material = sapien.render.RenderMaterial()
            if spec is None:
                random_rgb = self._spec.value(
                    f"layout.obstacle_rgb.{obstacle_order}",
                    torch.rand(3, generator=generator).tolist(),
                )
            else:
                random_rgb = [float(v) for v in spec["layout"]["obstacle_rgb"][obstacle_order]]
            cylinder_material.set_base_color((*random_rgb, 1))

            # Rotate upright then around its own z-axis to align with the target line.
            z_twist_mat = euler_angles_to_matrix(
                torch.tensor([0.0, 0.0, theta], dtype=torch.float32), convention="XYZ"
            )
            base_upright_mat = euler_angles_to_matrix(
                cylinder_angles, convention="XYZ"
            )
            final_rot_mat = z_twist_mat @ base_upright_mat
            cylinder_quat = matrix_to_quaternion(final_rot_mat)

            builder.set_initial_pose(
                sapien.Pose(
                    p=[
                        cube_position[0],
                        cube_position[1],
                        cylinder_half_length,
                    ],
                    q=cylinder_quat.detach().cpu().numpy(),
                )
            )

            rect_length = 0.03 #0.03
            rect_width = 0.015
            # Keep height the same as the previous cylinder; rotation keeps height along world z.
            builder.add_box_visual(
                half_size=[cylinder_half_length, rect_width, rect_length],
                material=cylinder_material,
            )
            builder.add_box_collision(
                half_size=[cylinder_half_length, rect_width, rect_length],
            )

            cube_actor = builder.build_kinematic(name=f"target_cube_{target_idx}")

            self.cubes_on_targets.append(cube_actor)
            self.target_cubes[target_idx] = cube_actor
            setattr(self, f"target_cube_{target_idx}", cube_actor)

        tasks = []



        tasks=[]

        # Use the actual button actors corresponding to indices 0,2,4,6,8
        walk_cfg = self._sampling["parameters"]["walk"]
        button_indices = list(walk_cfg["node_indices"])
        self.route_button_indices = button_indices

        sampling_configs = self._sampling["parameters"]["configs"]
        fallback_difficulty = self._sampling["parameters"]["configs_fallback_difficulty"]
        if is_newvalue_difficulty(getattr(self, "difficulty", "easy")):
            # V6（计划 2.12）：新值族缺键直接抛错，不静默回退到 easy
            if self.difficulty not in sampling_configs:
                raise ValueError(
                    f"RouteStick {self.difficulty}: sampling_config.parameters.configs 缺少 {self.difficulty} 档"
                )
            cfg = sampling_configs[self.difficulty]
        else:
            # 原三档：保持原有的静默回退行为逐字不变
            cfg = sampling_configs.get(getattr(self, "difficulty", "easy"), sampling_configs[fallback_difficulty])
        length_min, length_max = cfg.get("length")
        length_decision_key = f"configs.{getattr(self, 'difficulty', 'easy')}.length"
        if is_newvalue_difficulty(self.difficulty):
            # V5（计划 2.11 / L38）：xhard 的段数范围从 decision（规格 header 冻结的那份）读；V6 新值族按本局档位读
            length_min, length_max = self._xhard_segment_count_range()
            length_decision_key = f"{self.difficulty}.segment_count_range"
        allow_backtracking = bool(cfg.get("backtrack", True))
        if spec is None:
            steps = self._spec.value(
                "objects.L",
                int(torch.randint(length_min, length_max + 1, (1,), generator=generator).item()),
                decision_key=length_decision_key,
            )
            # 游走函数内部的抽样照常发生；这里只冻结最终节点序列
            traj = self._spec.value(
                "actions.nodes",
                list(generate_dynamic_walk(button_indices, steps=steps, allow_backtracking=allow_backtracking, generator=generator, walk_config=walk_cfg)),
            )
            if is_newvalue_difficulty(self.difficulty):
                # V5（N17 精神）：回放冻结规格时 value() 直接返回冻结值、不复核，这里复核段数与节点数
                if not length_min <= int(steps) <= length_max or len(traj) != int(steps) + 1:
                    raise _EpisodeSpecError(
                        f"RouteStick {self.difficulty}: 段数 {steps} / 节点数 {len(traj)} 与 "
                        f"segment_count_range [{length_min}, {length_max}] 不符（应为 L 在范围内、节点数 L+1）"
                    )
        else:
            # 规格定死路线：先按 generate_dynamic_walk 的线性邻接语义校验拓扑，
            # 再直接使用给定路线，不再抽随机数。
            steps = int(spec["objects"]["L"])
            traj = [int(v) for v in spec["actions"]["nodes"]]
            slots = [int(v) for v in spec["actions"]["node_slots"]]
            if not length_min <= steps <= length_max:
                raise ValueError(f"规格的段数 {steps} 超出难度 {self.difficulty} 的 [{length_min}, {length_max}]")
            if len(traj) != steps + 1 or len(slots) != steps + 1:
                raise ValueError(f"规格的节点数 {len(traj)} 与段数 {steps} 不符（应为 steps+1）")
            if [button_indices[i] for i in slots] != traj:
                raise ValueError("规格的 node_slots 与 nodes 不一致")
            for i in range(steps):
                if abs(slots[i + 1] - slots[i]) != 1:
                    raise ValueError(f"规格第 {i} 段不是相邻按钮：{slots[i]} → {slots[i+1]}")
                if not allow_backtracking and i > 0 and slots[i + 1] == slots[i - 1] and 0 < slots[i] < len(button_indices) - 1:
                    raise ValueError(f"难度 {self.difficulty} 不允许在非端点主动回退：第 {i} 段")
            self._injection_evidence = {
                "spec_sha256": spec.get("spec_sha256"),
                "episode": spec.get("episode"),
                "rotation_deg": float(spec["layout"]["rotation_deg"]),
                "nodes": list(traj),
                "node_slots": list(slots),
            }
        self.selected_buttons = [self.buttons_grid[i] for i in traj]

        def _stick_side(actor, ref_actor=None):
            """
            Determine whether a target lies on the left or right of a reference
            target based on their y positions. When no reference is provided,
            fall back to the workspace center (y=0).
            """
            def _get_y(a):
                pos = a.pose.p if hasattr(a, "pose") else a.get_pose().p
                pos_flat = np.asarray(pos).reshape(-1)
                return pos_flat[1] if pos_flat.size >= 2 else None

            y_val = _get_y(actor)
            ref_y = _get_y(ref_actor) if ref_actor is not None else 0.0
            if y_val is None or ref_y is None:
                # Fallback to right to avoid indexing errors; should not happen.
                return "right"
            return "left" if y_val > ref_y else "right"# Reverse according to robot perspective!

        # Randomly decide and record clockwise/counterclockwise direction for each solve_swingonto_withDirection
        self.swing_directions = []
        direction_cfg = walk_cfg["direction"]
        legal_directions = (direction_cfg["less_than"], direction_cfg["otherwise"])
        if spec is None:
            for order, _ in enumerate(self.selected_buttons[1:]):
                dir_flag = self._spec.value(
                    f"actions.directions.{order}",
                    direction_cfg["less_than"] if torch.rand(*direction_cfg["shape"], generator=generator).item() < direction_cfg["threshold"] else direction_cfg["otherwise"],
                )
                self.swing_directions.append(dir_flag)
        else:
            # 逐段绕行方向由规格定死；演示与执行两轮任务都读同一份列表，绑定天然一致
            spec_directions = list(spec["actions"]["directions"])
            if len(spec_directions) != len(self.selected_buttons) - 1:
                raise ValueError(f"规格的方向数 {len(spec_directions)} 与段数 {len(self.selected_buttons) - 1} 不符")
            for dir_flag in spec_directions:
                if dir_flag not in legal_directions:
                    raise ValueError(f"规格出现非法绕行方向 {dir_flag!r}")
                self.swing_directions.append(dir_flag)
            self._injection_evidence["directions"] = list(self.swing_directions)
        logger.debug(f"[RouteStick] swing direction list: {self.swing_directions}")

        current_target=self.selected_buttons[0]
        tasks.append({
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t, distance_threshold=0.03, z_threshold=self.z_threshold),
            "name":  "NO RECORD",
            "subgoal_segment":f"NO RECORD",
            "choice_label": "pick up the stick",
            "demonstration": True,
            "failure_func":  None,
            "solve": lambda env, planner, t=current_target: solve_swingonto(env, planner, target=t,record_swing_qpos=True),

        })  
        for i, current_target in enumerate(self.selected_buttons[1:]):
            direction = self.swing_directions[i]
            prev_target = self.selected_buttons[i]
            stick_side = _stick_side(current_target, prev_target)
             #task_name=f"rotate around the {stick_side} stick {direction}"
            task_name=f"move to the nearest {stick_side} target by circling around the stick {direction}"
            tasks.append({# Decrease threshold to see if replay appears
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t, distance_threshold=0.03, z_threshold=self.z_threshold),
            "name": task_name,
            "subgoal_segment":task_name,
            "choice_label": task_name,
            "demonstration": True,
            "failure_func":  (lambda expected=current_target, last=prev_target: self._wrong_button_touch(expected_button=expected, last_button=last)),
            "expected_dir": direction,
            "solve": lambda env, planner, t=current_target, d=direction: solve_swingonto_withDirection(env, planner, target=t,radius=0.2,direction=d),
                })  
        tasks.append({
                    "func": lambda:reset_check(self,gripper="stick"),
                    "name": "NO RECORD",
                    "subgoal_segment":"NO RECORD",
                    "choice_label": "place the stick into the tube",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [solve_strong_reset(env,planner,timestep=200,gripper="stick")],
                    },),
        
        current_target=self.selected_buttons[0]
        tasks.append({
            "func":   lambda:reset_check(self,gripper="stick",target_qpos=self.swing_qpos),
            "name":  "NO RECORD",
            "subgoal_segment":f"NO RECORD",
            "choice_label": "pick up the stick",
            "demonstration": True,
            "failure_func":  None,
            "solve": lambda env, planner, t=current_target: [solve_strong_reset(env, planner,gripper="stick",action=self.swing_qpos)],
        })  
        for i, current_target in enumerate(self.selected_buttons[1:]):
            direction = self.swing_directions[i]
            prev_target = self.selected_buttons[i]
            stick_side = _stick_side(current_target, prev_target)
            #task_name=f"rotate around the {stick_side} stick {direction}"
            task_name=f"move to the nearest {stick_side} target by circling around the stick {direction}"
            tasks.append({
            "func":   lambda t=current_target,list=[current_target, prev_target,direction]: is_obj_swing_onto(self, obj=self.agent.tcp, target=t, distance_threshold=0.03, z_threshold=self.z_threshold,judge_direction_list=list),
            "name": task_name,
            "subgoal_segment":task_name,
            "choice_label": task_name,
            "demonstration": False,
            "failure_func":  (lambda expected=current_target, last=prev_target: [self._wrong_button_touch(expected_button=expected, last_button=last)]),
            "expected_dir": direction,
            "solve": lambda env, planner, t=current_target, d=direction: solve_swingonto_withDirection(env, planner, target=t,radius=0.2,direction=d),
        })  
            # Store task list for RecordWrapper use
        self.task_list = tasks



    def _xhard_segment_count_range(self):
        """V5（计划 2.11 / L38）：xhard 的段数 L 范围，取自 ``decision.xhard.segment_count_range``。

        抽签时它等于 ``config_xhard4.length`` 的默认值；回放时 sampling_config 来自规格 header，
        因此读到的是冻结值。缺键说明传入的是 V4 或更早的快照（V4 header 没冻结 L 范围），
        V4 已作废（口径 13），直接报错而不是回退到类属性。
        """
        # V6：按本局档位取子树（新值族四档同结构）
        xhard_cfg = self._sampling["decision"].get(self.difficulty)
        value = xhard_cfg.get("segment_count_range") if isinstance(xhard_cfg, dict) else None
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
                or not 1 <= value[0] <= value[1]):
            raise ValueError(
                f"RouteStick {self.difficulty}: sampling_config.decision.{self.difficulty}.segment_count_range 缺失或不是 "
                f"[下界, 上界] 正整数对（收到 {value!r}；V4 及更早的快照在 V5 代码上不可用）"
            )
        return int(value[0]), int(value[1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):

            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos",gripper="stick")
            self.agent.reset(qpos)
            self.failureflag = torch.tensor([False])

     


    def _get_obs_extra(self, info: Dict):
        return dict()




    def evaluate(self,solve_complete_eval=False):
        previous_failure = getattr(self, "failureflag", torch.tensor([False]))
        if isinstance(previous_failure, torch.Tensor):
            failure_latched = bool(previous_failure.detach().cpu().item())
        else:
            failure_latched = bool(previous_failure)
        self.successflag = torch.tensor([False])
        self.failureflag = torch.tensor([True]) if failure_latched else torch.tensor([False])
        had_latched_fail = failure_latched



        # Record gripper xy position of current step (only record after demonstration is enabled)
        if  self.current_task_demonstration == False:
            gripper_xy = torch.as_tensor(self.agent.tcp.pose.p[0][:2]).detach().cpu()
            self._gripper_xy_trace.append((self.elapsed_steps, gripper_xy))

        # Use encapsulated sequence task check function
        if(self.use_demonstrationwrapper==False):# change subgoal after planner ends during recording
            if solve_complete_eval==True:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False
        else:# during demonstration, video needs to call evaluate(solve_complete_eval) video ends and flag changes in demonstrationwrapper
            if solve_complete_eval==True or self.demonstration_record_traj==False:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False
        all_tasks_completed, current_task_name, task_failed,self.current_task_specialflag = sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)
     
        

        # If task failed, mark as failed immediately, and keep fail thereafter
        if task_failed:
            self.failureflag = torch.tensor([True])
            if not had_latched_fail:
                logger.debug(f"Task failed: {current_task_name}")

        # If static_check succeeds or all tasks completed, set success flag
        if all_tasks_completed and not task_failed:
            self.successflag = torch.tensor([True])


        # # Check if a swing task has just been completed
        # if after_demo_active:
        #     # If current_task_name jumps, record adjacent two targets in order
        #     last_logged_name = getattr(self, "_last_logged_task_name", None)
        #     if current_task_name != last_logged_name:
        #         change_idx = getattr(self, "_swing_task_change_idx", 0)
        #         buttons = getattr(self, "selected_buttons", [])
        #         if change_idx + 1 < len(buttons):
        #             first_target = buttons[change_idx]
        #             second_target = buttons[change_idx + 1]
        #             expected_dir = (
        #                 self.swing_directions[change_idx]
        #                 if change_idx < len(getattr(self, "swing_directions", []))
        #                 else None
        #             )
        #             self._swing_success_history = [
        #                 {"step": cur_step, "target": first_target, "expected_dir": None},
        #                 {"step": cur_step, "target": second_target, "expected_dir": expected_dir},
        #             ]
        #             self._swing_task_change_idx = change_idx + 1
        #         self._last_logged_task_name = current_task_name

        #     # When there are two recent successful swings (different targets), judge gripper trajectory on left/right side of line between two targets and print
        #     if len(self._swing_success_history) == 2:
        #         first, second = self._swing_success_history
        #         pair_key = (first["step"], second["step"])
        #         if first["target"] is not second["target"] and self._last_swing_pair_reported_step != pair_key:
        #             start, end = first["step"], second["step"]
        #             segment = [(s, xy) for s, xy in self._gripper_xy_trace if start <= s <= end]
        #             # Only keep trajectory from first timestamp, avoid list growing infinitely
        #             self._gripper_xy_trace = [(s, xy) for s, xy in self._gripper_xy_trace if s >= start]

        #             t1 = torch.as_tensor(first["target"].pose.p[0][:2]).detach().cpu()
        #             t2 = torch.as_tensor(second["target"].pose.p[0][:2]).detach().cpu()
        #             line_vec = t2 - t1
        #             cross_vals = []
        #             for _, xy in segment:
        #                 rel = xy - t1
        #                 cross_vals.append(float(line_vec[0] * rel[1] - line_vec[1] * rel[0]))

        #             if cross_vals:
        #                 avg_cross = sum(cross_vals) / len(cross_vals)
        #                 # According to current coordinate system, positive cross product direction should be considered clockwise
        #                 side = "clockwise" if avg_cross > 0 else "counterclockwise" if avg_cross <0 else "on the line"
        #                 expected_dir = second.get("expected_dir")
        #                 if expected_dir and side != "on the line" and side != expected_dir:
        #                     print("direction mistake!!!")
        #                 print(f"Gripper path from step {start} to {end} stayed on the {side} side of the directed line between the last two targets.")
        #             self._last_swing_pair_reported_step = pair_key

        return {
            "success": self.successflag,
            "fail": self.failureflag,
        }

    def direction_fail(self,judge_direction_list=None):
        if judge_direction_list is None:
            return True

        # judge_direction_list format is [current_target, prev_target, expected_dir]
        try:
            current_target, prev_target, expected_dir = judge_direction_list
        except Exception:
            # Unable to judge direction when parameter is abnormal, keep unfinished status
            self.failureflag = torch.tensor([True])
            return False

        trace = getattr(self, "_gripper_xy_trace", [])

        # Calculate vector from previous target to current target
        prev_xy = torch.as_tensor(prev_target.pose.p[0][:2]).detach().cpu()
        curr_xy = torch.as_tensor(current_target.pose.p[0][:2]).detach().cpu()
        line_vec = curr_xy - prev_xy

        # If two targets coincide or no trajectory, unable to judge direction
        if torch.norm(line_vec) < 1e-6 or len(trace) == 0:
            self.failureflag = torch.tensor([True])
            return False

        cross_vals = []
        for _, xy in trace:
            xy = torch.as_tensor(xy).detach().cpu()
            rel = xy - prev_xy
            cross_vals.append(float(line_vec[0] * rel[1] - line_vec[1] * rel[0]))

        # Clear trajectory immediately after consumption, convenient for next segment judgment
        self._gripper_xy_trace = []

        if len(cross_vals) == 0:
            self.failureflag = torch.tensor([True])
            return False

        avg_cross = sum(cross_vals) / len(cross_vals)
        # Positive cross product in current coordinate system is considered clockwise
        side = "clockwise" if avg_cross > 0 else "counterclockwise" if avg_cross < 0 else "on the line"

        # Unable to judge direction along line; need to retry
        if side == "on the line":
            self.failureflag = torch.tensor([True])
            return False

        expected_dir = str(expected_dir).lower()
        if side != expected_dir:
            logger.debug(f"direction mistake: expected {expected_dir}, got {side}")
            self.failureflag = torch.tensor([True])
            return False

        return True





    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        reward=torch.tensor([0])
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):

        obs, reward, terminated, truncated, info = super().step(action)

        cur_step = int(self.elapsed_steps[0].item())
        trail_cfg = self._sampling["positions"].get(
            "tcp_trail", {"end_offset_steps": 40, "disk_radius": 0.005}
        )
        highlight_position(
            self,
            self.agent.tcp.pose.p,
            start_step=cur_step,
            # 原值 40（官方 dataset-gen）；2026-09-12 的 10 步方案改为传 sampling_config 覆盖，
            # 不再写死在源码里——尾迹进相机观测，写死会让 A↔B 永远不可能逐位相同。
            end_step=cur_step + int(trail_cfg["end_offset_steps"]),
            cur_step=cur_step,
            disk_radius=float(trail_cfg["disk_radius"]),
        )


        for idx, button in enumerate(self.buttons_grid):
            if is_obj_swing_onto(self, obj=self.agent.tcp, target=button,distance_threshold=0.03):
                # Update start time to refresh highlight effect when triggered repeatedly
                self.highlight_starts[idx] = cur_step

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
    

    def _wrong_button_touch(self, expected_button, last_button=None):
        # Judge as error when touched button is neither current expected target nor previous button (debounce)
        for button in self.buttons_grid:
            if button is expected_button:
                continue
            if last_button is not None and button is last_button:
                continue
            if is_obj_swing_onto(self, obj=self.agent.tcp, target=button):
                return True
        return False
