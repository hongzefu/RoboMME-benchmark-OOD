from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import SO100, Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Actor
#Robomme
import matplotlib.pyplot as plt

from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
import copy
from .utils import *
from .utils.difficulty import normalize_robomme_difficulty, require_xhard4_only
from .utils.subgoal_evaluate_func import static_check
from .utils.episode_spec import SpecRecorder
from .utils.sampling_config import SamplingConfigError, assert_native_decision, split_sampling_config
from .utils.SceneGenerationError import SceneGenerationError
from .utils.episode_spec import EpisodeSpecError
from .utils import subgoal_language
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole, point_segment_distance_xy
from .utils import reset_panda
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


# ── decision／native 两块的原值（newtaskRelease-v3 步 3，映射见方案第二节 2.13）────────
NATIVE_SAMPLING = {
    "parameters": {
        "peg_size": {
            "length_expression": "0.1 + (0.05 - 0.05) * rand()",
            "radius_expression": "0.01 + (0.005 - 0.005) * rand()",
            "note": "两次 rand 结果被乘 0 消掉，但必须保留以免平移随机流（红线 R8）",
        },
        "way_selection": {"sampler": "torch.randint(len(self.ways))"},
        "obj_selection": {"sampler": "torch.randint(0, 2)", "mapping": [-1, 1]},
        "dir_sample": {"sampler": "torch.randint(0, 2)", "consumed": False,
                        "note": "抽了但未使用，保留为 sampling_trace"},
        "direction_rule": "evaluate 按两套布局的实际 y 差给 ±1，不再随机抽",
        "reset_rule": "step 切换到已生成的执行位姿",
        "recovery": "本环境没有 inject_fail_grasp，只接收入口恢复模式，实际恢复动作为 null",
    },
    "positions": {
        "goal_demo": {"region_center": [0.0, 0.0], "region_half_size": 0.15,
                       "radius_factor": 2, "min_gap_factor": 1},
        "goal_execution": {"region_center": [0.0, 0.0], "region_half_size": 0.1,
                            "radius_factor": 2, "min_gap_factor": 1},
        "cube_rejection": {"max_trials": 128, "min_distance_factor": 5},
        "peg_color": {"head": "#EC7357", "tail": "#EC7357"},
    },
}


def native_blocks(cls):
    """本环境的 ``(decision, native)`` 原值块；外部导出与内部解析共用同一份。"""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


def _native_decision(cls):
    """按方案第二节 2.13 切出 decision 块（原值阶段等于原值）。

    V4（计划 2.17）：原 xhard 档的新值一律挂在名为 ``xhard4`` 的子键下（守卫只放行这些键偏离），
    默认值取自 ``cls.configs["xhard4"]``；原三档可见部分与 V3 逐字相同。

    V5（计划 2.9，L30/L33）：V4 的 ``corner_bias`` 已删除。
    V6（计划 2.6）：V5 的 ``center_exclusion`` 已删除，演示段与执行段各自暴露一份 ``region``
    （统一区域 U，两段各自声明、各自消费）。
    """
    xhard4 = cls.configs["xhard4"]
    return {
        # 演示阶段的方块与杆位置采样规则（原值：杆基位 y=±0.2、xy 各抖动 ±0.05；
        # 方块候选中心 xy 各 ±0.1，再在 half_size 0.05 的小区里生成）。
        "demo_layout": {
            "peg_position_policy": {"base_y_abs": 0.2, "base_y_threshold": 0.5, "jitter_span": 0.1},
            "cube_position_policy": {"center_span": 0.2, "center_offset": -0.1, "region_half_size": 0.05},
            # V6 xhard4：统一区域 U（计划 2.6；两段各自声明、各自消费，取值相同）
            "xhard4": {"region": copy.deepcopy(xhard4["region"])},
        },
        # 执行阶段另抽一套，原规则与演示相同但必须分开，不能误合并。
        "execution_layout": {
            "peg_position_policy": {"base_y_abs": 0.2, "base_y_threshold": 0.5, "jitter_span": 0.1},
            "cube_position_policy": {"center_span": 0.2, "center_offset": -0.1, "region_half_size": 0.05},
            # V6 xhard4：执行段的统一区域 U，与演示段各自声明、各自消费（两套不可合并）
            "xhard4": {"region": copy.deepcopy(xhard4["region"])},
        },
        # 杆在桌面内的转角范围：原值 ±π/4（表达式为 u*span - offset）。
        # V4 xhard4：±π（A1，仍只绕世界 z；joint7 冲突按等价朝向归约，见 B11）。
        "peg_yaw_range": {"span_rad": np.pi / 2, "offset_rad": np.pi / 4,
                          "xhard4": dict(xhard4["peg_yaw_range"])},
    }


def _resolve_sampling_config(cls, override):
    """拆出本实例专属的 decision／native 副本；不抽随机数，必须在 Generator 之前调用。"""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    native["decision"] = decision
    return native


# ── V6 xhard4 统一区域 U（计划 2.6）：纯数值判据，不抽随机数，只在 xhard4 分支被调用 ──────────
# 机械臂基座的桌面 xy（与 ``MoveCube._load_agent`` 的 ``sapien.Pose(p=[-0.615, 0, 0])`` 同值）
ROBOT_BASE_XY = (-0.615, 0.0)
# 推起点几何（``solve_push_to_target`` / ``solve_push_to_target_with_peg``）：方块沿推方向后退 0.10 起推，
# 带杆推时再沿法向侧移 0.10；U 的成对约束要求这三个推起点离基座也落在 ``base_dist`` 区间内
PUSH_BACKOFF_M = 0.10
PEG_PUSH_LATERAL_M = 0.10


def _peg_axis_extent(length):
    """杆轴线段在杆根坐标系里沿杆朝向 u 的区间 ``(t_min, t_max)``（米）。

    由 ``utils/object_generation.py::build_peg`` 的几何推出：head link 以杆根为中心，tail link
    经固定关节挂在 ``−length·u``；两段的碰撞盒半长 ``0.45·length``、可视盒半长 ``0.5·length``。
    取碰撞与可视外形的并集（即可视外形），``length=0.1`` 时为 ``(−0.15, +0.05)``，与 P2 实测一致。
    xhard4 每次建杆后由 ``MoveCube._xhard4_verify_peg_extent`` 用实际形状复核，二者不符即报错。
    """
    head_half = 0.5 * float(length)          # 可视盒半长（碰撞盒 0.45·length 被它包住）
    tail_center = -float(length)             # tail 固定关节的 pose_in_parent
    return (tail_center - head_half, head_half)


def _peg_root_xy(base_y, x_jitter, y_jitter):
    """与 ``_load_scene`` 建杆时的 float32 平移逐位同算法，得到杆根 xy（float64）。"""
    translation = np.array([0.0, base_y, 0.0], dtype=np.float32)
    translation[1] = base_y
    translation[:2] += np.array([x_jitter, y_jitter], dtype=np.float32)
    return translation[:2].astype(np.float64)


def _peg_geometry(root, yaw, length, extent):
    """杆根 ``root``、朝向 ``yaw`` → ``(抓取点 xy, 杆身线段端点 a, b)``（float64）。

    抓取点 = tail link 中心 = ``root − length·u``（``evaluate`` 里 ``grasp_target = peg_tail``，
    ``grasp_and_lift_peg_side`` 抓的就是它）；杆身线段 = ``root + t·u``，``t ∈ extent``。
    """
    u = np.array([np.cos(float(yaw)), np.sin(float(yaw))], dtype=np.float64)
    root = np.asarray(root, dtype=np.float64)
    return root - float(length) * u, root + extent[0] * u, root + extent[1] * u


def _in_region_u(xy, region):
    """点（物体中心／抓取点）是否在统一区域 U 内：圆环 ∧ 离基座区间。违反返回说明文字，否则 None。"""
    xy = np.asarray(xy, dtype=np.float64)
    rc = float(np.linalg.norm(xy - region["center"]))
    if not (region["r_in"] <= rc <= region["r_out"]):
        return f"离圆环圆心 {rc:.6f} 不在 [{region['r_in']}, {region['r_out']}]"
    rb = float(np.linalg.norm(xy - np.asarray(ROBOT_BASE_XY, dtype=np.float64)))
    if not (region["base_lo"] <= rb <= region["base_hi"]):
        return f"离基座 {rb:.6f} 不在 [{region['base_lo']}, {region['base_hi']}]"
    return None


def _peg_region_violation(root, yaw, length, extent, region):
    """杆规则：抓取点在 U 内，且杆身线段离圆环圆心 ≥ r_in（杆整体不进圆环内孔）。违反返回说明文字。"""
    grasp, a, b = _peg_geometry(root, yaw, length, extent)
    why = _in_region_u(grasp, region)
    if why is not None:
        return f"抓取点{why}"
    d = point_segment_distance_xy(region["center"], a, b)
    if d < region["r_in"]:
        return f"杆身线段离圆环圆心 {d:.6f} < r_in {region['r_in']}"
    return None


def _assert_peg_in_region(root, yaw, length, extent, region, spec_prefix):
    """N17：回放冻结规格时杆位姿不经拒绝循环，按同一规则复核，违反抛 ``EpisodeSpecError``。"""
    why = _peg_region_violation(root, yaw, length, extent, region)
    if why is not None:
        raise EpisodeSpecError(
            f"MoveCube xhard4：{spec_prefix}.peg_offsets/peg_yaw 违反统一区域 U（冻结或注入值）：{why}")


@register_env("MoveCube", override=True)
class MoveCube(BaseEnv):

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
    _clearance = 0.01

    # V4（A4/A6）：本环境原本没有难度分档，现有全局常量即 hard；三档同值，
    # difficulty 只在 xhard4 生效。消费点读的是 decision（可被 sampling_config 覆盖），
    # 这里是 decision 的默认值来源。
    config_native = {
        "peg_yaw_range": {"span_rad": np.pi / 2, "offset_rad": np.pi / 4},
        "corner_bias": 0.0,
    }
    config_xhard4 = {
        # ±180°：u*2π - π（A1）
        "peg_yaw_range": {"span_rad": 2 * np.pi, "offset_rad": np.pi},
        # V6（计划 2.6，M8；用户 2026-09-25 定圆环版）：统一区域 U，替代 V5 的 (0,0)、R=0.05 中心圆禁区与
        # 三个互不相同的小框。方块中心、goal 中心、杆抓取点（杆尾 = 杆根 − 0.10·u）共用：
        #   圆环 r_in ≤ |p − center| ≤ r_out（center = 可达环带 [0.31, 0.80] 的中点 (−0.06, 0)），
        #   保险 base_dist[0] ≤ |p − 基座(−0.615, 0)| ≤ base_dist[1]；
        # 成对：0.10（原 cube_rejection 的 5×半边长）≤ |方块 − goal| ≤ push_len_max，三个推起点离基座也在 base_dist；
        # 杆：杆身线段离 center ≥ r_in；方块离杆身 ≥ peg_gap，goal 离杆身 ≥ goal_peg_gap。
        # 依据：artifacts/newtask-v6/plan-probes/reach/（A/B/C 三路实测）与 reach/U/region.py 离线定义。
        # 三个 *_max_trials 是杆／goal／方块各自拒绝循环的预算（方块受推距与推起点约束，单次接受率最低约 0.5%，
        # 故给 4096；离线估计最坏段耗尽概率约 1e-9），超出抛 SceneGenerationError。
        # V9（计划 1002-newtask-v9-movecube-region-800-plan.md §2，用户 2026-10-02 定最终版）：圆环由 V6 的
        # 0.12–0.20 扩到 r_in 0.24、r_out 0.42；base_dist 由 [0.35, 0.76]（两头各留 4 cm）改为 [0.31, 0.80]，
        # 正好是可达边界——依据 docs/validation/newtask-v6/records/legacy/plan-probes/reach/{A,B} 探针实测：
        # 末端全 yaw 可达 0.31–0.80 m，抓杆 0.27–0.80 m 全部成功（0.80–0.85 m 仅 48%，不在 V9 范围内）。
        # 区域 U = 圆环 0.24–0.42 ∩ 离基座 0.31–0.80，只剩左右两块；center、push_len_max、两个 gap 与三个预算不变。
        "region": {"center": [-0.06, 0.0], "r_in": 0.24, "r_out": 0.42, "base_dist": [0.31, 0.80],
                   "push_len_max": 0.30, "peg_gap": 0.04, "goal_peg_gap": 0.02,
                   "peg_max_trials": 128, "goal_max_trials": 256, "cube_max_trials": 4096},
    }
    configs = {
        "easy": config_native,
        "medium": config_native,
        "hard": config_native,
        "xhard4": config_xhard4,
    }

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     sampling_config=None,
                     native_episode_spec=None,
                     **kwargs):
        # 必须落在任何随机数调用与 super().__init__() 之前
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._spec = SpecRecorder(native_episode_spec, "MoveCube", {"seed": seed},
                                  difficulty=kwargs.get("difficulty"))
        # 初始化序号从 -1 起，_initialize_episode 每次进来先加一；
        # _load_scene 里的取值点用不带序号的路径，所以这里只作兜底。
        self._native_init_index = -1
        self.reset_in_proecess=False
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
        self._hb_generator = torch.Generator()
        self._hb_generator.manual_seed(int(self.seed))

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
            else:
                self.difficulty = "hard"
        require_xhard4_only(self.difficulty, "MoveCube")
        if self.difficulty == "xhard4":
            # V4 B11：抓杆时按等价朝向归约（只改夹爪姿态，并同步补偿抓杆后的推杆路点）；
            # 求解器按这个开关分叉，原三档不设此属性、走原路径
            self._xhard_peg_yaw_reduction = True

        self.restore_flag=False
        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
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
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=0
        )
        self.table_scene.build()

        length_tensor = torch.rand(1, generator=self._hb_generator)
        radius_tensor = torch.rand(1, generator=self._hb_generator)
        self.length = (0.1 + (0.05 - 0.05) * length_tensor).item()
        self.radius = (0.01 + (0.005 - 0.005) * radius_tensor).item()

        # Create a single peg
        #peg_spawn_translation = np.array([self.length / 2, 0.0, self.radius], dtype=np.float32)
        demo_layout = self._sampling["decision"]["demo_layout"]
        exec_layout = self._sampling["decision"]["execution_layout"]
        peg_yaw_range = self._sampling["decision"]["peg_yaw_range"]
        native_pos = self._sampling["positions"]
        # V6 xhard4（计划 2.6）：统一区域 U 走独立分支（随机调用次序在分支内重排，xhard4 重冻）；
        # 原三档不经过该分支，下面的原路径与 V3 原值逐字相同（V1 闸门）。
        if self.difficulty == "xhard4":
            self._load_scene_xhard4_region(demo_layout, exec_layout, peg_yaw_range, native_pos)
            return
        yaw_policy = peg_yaw_range
        demo_peg = demo_layout["peg_position_policy"]
        base_y = -demo_peg["base_y_abs"] if torch.rand(1, generator=self._hb_generator).item() < demo_peg["base_y_threshold"] else demo_peg["base_y_abs"]

        peg_spawn_translation = np.array([0.0, base_y, 0.0], dtype=np.float32)

        # Generate [-0.05, 0.05] random offset (using torch generator)
        x_jitter = (torch.rand(1, generator=self._hb_generator).item() - 0.5) * demo_peg["jitter_span"]
        y_jitter = (torch.rand(1, generator=self._hb_generator).item() - 0.5) * demo_peg["jitter_span"]

        # Apply offset
        base_y, x_jitter, y_jitter = self._spec.value(
            "layout.demo.peg_offsets", [base_y, x_jitter, y_jitter], decision_key=None
        )
        peg_spawn_translation[1] = base_y
        peg_spawn_translation[:2] += np.array([x_jitter, y_jitter], dtype=np.float32)
        self.peg1_basex=peg_spawn_translation[0]
        self.peg1_basey=peg_spawn_translation[1]

        initial_yaw = torch.rand(1, generator=self._hb_generator).item() * (yaw_policy["span_rad"]) - (yaw_policy["offset_rad"])
        initial_yaw = self._spec.value("layout.demo.peg_yaw", initial_yaw, decision_key=None)
        yaw_angles = torch.tensor([[0.0, 0.0, initial_yaw]], dtype=torch.float32)
        yaw_matrix = euler_angles_to_matrix(yaw_angles, convention="XYZ")
        yaw_quat = matrix_to_quaternion(yaw_matrix)[0].detach().cpu().numpy().tolist()

        peg_initial_pose = sapien.Pose(
            p=peg_spawn_translation.tolist(),
            q=yaw_quat,
        )

        self.peg, self.peg_head, self.peg_tail = build_peg(
            self,
            length=self.length,
            radius=self.radius,
            initial_pose=peg_initial_pose,
            name='peg',
            head_color= "#EC7357",
            tail_color= "#EC7357",
        )

        # Create lists for backward compatibility
        self.pegs = [self.peg]
        self.peg_heads = [self.peg_head]
        self.peg_tails = [self.peg_tail]



        # Store initial poses for all pegs
        self.peg_init_poses = [peg.pose for peg in self.pegs]
        self.peg_init_pose = self.pegs[0].pose  # Keep backward compatibility

        #generate another set of pose for another reset
        exec_peg = exec_layout["peg_position_policy"]
        base_y = -exec_peg["base_y_abs"] if torch.rand(1, generator=self._hb_generator).item() < exec_peg["base_y_threshold"] else exec_peg["base_y_abs"]

        peg_spawn_translation = np.array([0.0, base_y, 0.0], dtype=np.float32)
        x_jitter = (torch.rand(1, generator=self._hb_generator).item() - 0.5) * exec_peg["jitter_span"]
        y_jitter = (torch.rand(1, generator=self._hb_generator).item() - 0.5) * exec_peg["jitter_span"]
        base_y, x_jitter, y_jitter = self._spec.value(
            "layout.execution.peg_offsets", [base_y, x_jitter, y_jitter], decision_key=None
        )
        peg_spawn_translation[1] = base_y
        peg_spawn_translation[:2] += np.array([x_jitter, y_jitter], dtype=np.float32)

        initial_yaw = torch.rand(1, generator=self._hb_generator).item() * (yaw_policy["span_rad"]) - (yaw_policy["offset_rad"])
        initial_yaw = self._spec.value("layout.execution.peg_yaw", initial_yaw, decision_key=None)
        yaw_angles = torch.tensor([[0.0, 0.0, initial_yaw]], dtype=torch.float32)
        yaw_matrix = euler_angles_to_matrix(yaw_angles, convention="XYZ")
        yaw_quat = matrix_to_quaternion(yaw_matrix)[0].detach().cpu().numpy().tolist()

        self.peg2_basex=peg_spawn_translation[0]
        self.peg2_basey=peg_spawn_translation[1]
        peg_initial_pose = sapien.Pose(
            p=peg_spawn_translation.tolist(),
            q=yaw_quat,
        )
        self.peg_init_poses_2=[peg_initial_pose]
        
        self.finish_return_flag=False

                # Define task list, each task contains a dictionary with function, name, demonstration flag, and optional failure_func
        self._sample_obj_and_dir()

        self.goal_site = spawn_random_target(
                    self,
                    avoid=None,  # Use current avoidance list, containing all spawned cubes
                    include_existing=False,  # Manually maintain list
                    include_goal=False,  # Manually maintain list
                    region_center=list(native_pos["goal_demo"]["region_center"]),
                    region_half_size=native_pos["goal_demo"]["region_half_size"],
                    radius=self.cube_half_size*native_pos["goal_demo"]["radius_factor"],  # Use radius instead of half_size
                    thickness=0.005,  # target thickness
                    min_gap=self.cube_half_size*1,  # Gap requirement same as cube
                    name_prefix=f"goal_site",
                    recorder=self._spec,
                    spec_path="layout.demo.goal_xy",
                    generator=self._hb_generator,
                    )
        self.goal_site_2 = spawn_random_target(
            self,
            avoid=None,  # Use current avoidance list, containing all spawned cubes
            include_existing=False,  # Manually maintain list
            include_goal=False,  # Manually maintain list
            region_center=list(native_pos["goal_execution"]["region_center"]),
            region_half_size=native_pos["goal_execution"]["region_half_size"],
            radius=self.cube_half_size*native_pos["goal_execution"]["radius_factor"],  # Use radius instead of half_size
            thickness=0.005,  # target thickness
            min_gap=self.cube_half_size*1,  # Gap requirement same as cube
            name_prefix=f"goal_site_2",
            recorder=self._spec,
            spec_path="layout.execution.goal_xy",
            generator=self._hb_generator,
            )
        


        max_cube_spawn_trials = native_pos["cube_rejection"]["max_trials"]

        goal_pos = self.goal_site.pose.p
        goal_xy = np.asarray(goal_pos)
        goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(-1)[:2]

        def _sample_cube_center(required_distance: float):
            for _ in range(max_cube_spawn_trials):
                demo_cube = demo_layout["cube_position_policy"]
                sampled_x = torch.rand(1, generator=self._hb_generator).item() * demo_cube["center_span"] + demo_cube["center_offset"]
                #direction = -1.0 if -self.peg1_basey < 0 else 1.0
                #sampled_y = torch.rand(1, generator=self._hb_generator).item() * 0.2 * direction
                sampled_y = torch.rand(1, generator=self._hb_generator).item() * demo_cube["center_span"] + demo_cube["center_offset"]
                candidate_xy = np.array([sampled_x, sampled_y], dtype=np.float64)
                if np.linalg.norm(candidate_xy - goal_xy) > required_distance:
                    return candidate_xy
            return None

        cube_center = _sample_cube_center(self.cube_half_size*native_pos["cube_rejection"]["min_distance_factor"])

        cube_x, cube_y = float(cube_center[0]), float(cube_center[1])

        self.cube = spawn_random_cube(
                        self,
                        region_center=[cube_x, cube_y],
                        color=(1, 0, 0, 1),
                        name_prefix="fixed_cube",
                        recorder=self._spec,
                        spec_path="layout.demo.cube_pose",
                        region_half_size=demo_layout["cube_position_policy"]["region_half_size"],
                        generator=self._hb_generator,
                        half_size=self.cube_half_size,
                    )
        
        self.cube_init_pose=self.cube.pose



        goal_pos = self.goal_site_2.pose.p
        goal_xy = np.asarray(goal_pos)
        goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(-1)[:2]
        def _sample_cube_center(required_distance: float):
            for _ in range(max_cube_spawn_trials):
                exec_cube = exec_layout["cube_position_policy"]
                sampled_x = torch.rand(1, generator=self._hb_generator).item() * exec_cube["center_span"] + exec_cube["center_offset"]
                #direction = -1.0 if -self.peg2_basey < 0 else 1.0
                #sampled_y = torch.rand(1, generator=self._hb_generator).item() * 0.2 * direction
                sampled_y = torch.rand(1, generator=self._hb_generator).item()  * exec_cube["center_span"] + exec_cube["center_offset"]
                candidate_xy = np.array([sampled_x, sampled_y], dtype=np.float64)
                if np.linalg.norm(candidate_xy - goal_xy) > required_distance:
                    return candidate_xy
            return None

        cube_center = _sample_cube_center(self.cube_half_size*native_pos["cube_rejection"]["min_distance_factor"])

        cube_x, cube_y = float(cube_center[0]), float(cube_center[1])
        self.cube_2 = spawn_random_cube(
                        self,
                        region_center=[cube_x, cube_y],
                        color=(1, 0, 0, 1),
                        name_prefix="fixed_cube_2",
                        recorder=self._spec,
                        spec_path="layout.execution.cube_pose",
                        region_half_size=exec_layout["cube_position_policy"]["region_half_size"],
                        generator=self._hb_generator,
                        half_size=self.cube_half_size,
                    )
        
        self.cube_init_pose_2=self.cube_2.pose
        #only need the pose! teleport away in 
        self._store_goal_poses()

    def _sample_obj_and_dir(self):
        """obj_sample 与 dir_sample 两次抽样（原三档与 xhard4 共用；次序与原代码相同）。"""
        obj_sample = self._spec.value(
            "objects.obj_sample",
            int(torch.randint(0, 2, (1,), generator=self._hb_generator).item()),
        )
        self.obj_flag = -1 if obj_sample == 0 else 1
        # 这次抽样原本就没被消费；记进 sampling_trace 以证明它照常发生（红线 R8）
        dir_sample = self._spec.value(
            "objects.sampling_trace.dir_sample",
            int(torch.randint(0, 2, (1,), generator=self._hb_generator).item()),
        )
        #self.direction = -1 if dir_sample.item() == 0 else 1

    def _store_goal_poses(self):
        """两段 goal 圆盘位姿的 float64 副本（不抽随机数；原三档与 xhard4 共用）。"""
        goal2_p = np.array(self.goal_site_2.pose.p.detach().cpu().numpy(), dtype=np.float64, copy=True)
        self.goal_site_2_pose_p = goal2_p

        goal2_q = np.array(self.goal_site_2.pose.q.detach().cpu().numpy(), dtype=np.float64, copy=True)
        self.goal_site_2_pose_q = goal2_q

        goal1_p = np.array(self.goal_site.pose.p.detach().cpu().numpy(), dtype=np.float64, copy=True)
        self.goal_site_1_pose_p = goal1_p

        goal1_q = np.array(self.goal_site.pose.q.detach().cpu().numpy(), dtype=np.float64, copy=True)
        self.goal_site_1_pose_q = goal1_q

    # ── V6 xhard4：统一区域 U（计划 2.6）─────────────────────────────────────────────
    def _load_scene_xhard4_region(self, demo_layout, exec_layout, peg_yaw_range, native_pos):
        """V6 xhard4 布局：方块中心、goal 中心、杆抓取点三者共用统一区域 U（圆环 ∧ 离基座区间）。

        随机调用次序（只在本分支内，xhard4 重冻）：
        演示段杆（每次试验 3 次 ``torch.rand``：抓取点 x、y、yaw；拒绝就整组重抽）→ 执行段杆（同）→
        obj_sample、dir_sample → 演示段 goal → 执行段 goal → 演示段方块 → 执行段方块。
        V5 的 base_y 抽样、方块候选中心两级采样、``center_exclusion`` 圆禁区、演示／执行段 goal 不同框全部删除。

        杆：抓取点（杆尾 = 杆根 − length·u）在 U 内均匀、yaw ∈ ±π，杆根由二者推出；杆身线段离圆环圆心 ≥ r_in。
        goal：中心在 U 内，离本段杆身 ≥ goal_peg_gap。
        方块：中心在 U 内，离本段杆身 ≥ peg_gap，离本段 goal ∈ [min_cg, push_len_max]，三个推起点离基座在区间内。
        规格键沿用原路径：``layout.<seg>.peg_offsets = [0.0, 杆根 x, 杆根 y]``（base_y 恒 0、两个「抖动」即杆根 xy）。
        回放冻结值不经拒绝循环，杆由本方法、goal／方块由 spawn 函数按同一规则复核（N17）。
        """
        yaw_policy = peg_yaw_range["xhard4"]
        dk_yaw = "peg_yaw_range.xhard4"
        peg_extent = _peg_axis_extent(self.length)
        regions = {"demo": self._xhard4_region(demo_layout, "demo_layout"),
                   "execution": self._xhard4_region(exec_layout, "execution_layout")}
        trials = {"demo": {}, "execution": {}}
        pegs = {}
        for seg, label in (("demo", "演示段"), ("execution", "执行段")):
            region = regions[seg]
            root_x, root_y, yaw, trials[seg]["peg_trials"] = self._xhard4_sample_peg_in_region(
                region, yaw_policy, peg_extent, label)
            base_y, root_x, root_y = self._spec.value(
                f"layout.{seg}.peg_offsets", [0.0, root_x, root_y], decision_key=f"{region['key']}.xhard4.region")
            yaw = self._spec.value(f"layout.{seg}.peg_yaw", yaw, decision_key=dk_yaw)
            root = _peg_root_xy(base_y, root_x, root_y)
            # N17：回放时上面两处 value 返回冻结值、不经拒绝循环，必须按同一规则复核
            _assert_peg_in_region(root, yaw, self.length, peg_extent, region, f"layout.{seg}")
            translation = np.array([0.0, base_y, 0.0], dtype=np.float32)
            translation[1] = base_y
            translation[:2] += np.array([root_x, root_y], dtype=np.float32)
            yaw_angles = torch.tensor([[0.0, 0.0, yaw]], dtype=torch.float32)
            yaw_quat = matrix_to_quaternion(euler_angles_to_matrix(yaw_angles, convention="XYZ"))[0].detach().cpu().numpy().tolist()
            pose = sapien.Pose(p=translation.tolist(), q=yaw_quat)
            _, seg_a, seg_b = _peg_geometry(root, yaw, self.length, peg_extent)
            pegs[seg] = {"translation": translation, "pose": pose, "segment": (seg_a, seg_b)}
            if seg == "demo":
                self.peg1_basex = translation[0]
                self.peg1_basey = translation[1]
                self.peg, self.peg_head, self.peg_tail = build_peg(
                    self, length=self.length, radius=self.radius, initial_pose=pose, name='peg',
                    head_color="#EC7357", tail_color="#EC7357",
                )
                # 杆规则所用的杆轴线段必须与刚建好的杆的实际碰撞/可视几何一致，否则判据形同虚设
                self._xhard4_verify_peg_extent(peg_extent)
                self.pegs = [self.peg]
                self.peg_heads = [self.peg_head]
                self.peg_tails = [self.peg_tail]
                self.peg_init_poses = [peg.pose for peg in self.pegs]
                self.peg_init_pose = self.pegs[0].pose
            else:
                self.peg2_basex = translation[0]
                self.peg2_basey = translation[1]
                self.peg_init_poses_2 = [pose]
        self.finish_return_flag = False

        self._sample_obj_and_dir()

        base_xy = tuple(ROBOT_BASE_XY)
        goals = {}
        for seg, name, attr, goal_key, label in (
                ("demo", "goal_site", "goal_site", "goal_demo", "演示段"),
                ("execution", "goal_site_2", "goal_site_2", "goal_execution", "执行段")):
            region = regions[seg]
            radius = self.cube_half_size * native_pos[goal_key]["radius_factor"]
            try:
                goal = spawn_random_target(
                    self,
                    avoid=None,
                    include_existing=False,
                    include_goal=False,
                    # 采样框 = 圆环外接正方形（spawn 内部会把框收缩一个圆盘半径）
                    region_center=[float(region["center"][0]), float(region["center"][1])],
                    region_half_size=region["r_out"] + radius,
                    radius=radius,
                    thickness=0.005,
                    min_gap=self.cube_half_size * 1,
                    name_prefix=name,
                    max_trials=region["goal_max_trials"],
                    recorder=self._spec,
                    spec_path=f"layout.{seg}.goal_xy",
                    generator=self._hb_generator,
                    annulus=region["annulus"],
                    base_band=(base_xy, region["base_lo"], region["base_hi"]),
                    segment_clearance=[(*pegs[seg]["segment"], region["goal_peg_gap"])],
                )
            except RuntimeError as exc:
                if not isinstance(exc, SceneGenerationError):
                    raise SceneGenerationError(f"MoveCube xhard4：{label} goal 生成失败：{exc}") from exc
                raise
            setattr(self, attr, goal)
            goals[seg] = np.asarray(goal.pose.p, dtype=np.float64).reshape(-1)[:2]

        min_cg = self.cube_half_size * native_pos["cube_rejection"]["min_distance_factor"]
        for seg, name, attr, label in (("demo", "fixed_cube", "cube", "演示段"),
                                       ("execution", "fixed_cube_2", "cube_2", "执行段")):
            region = regions[seg]
            try:
                cube = spawn_random_cube(
                    self,
                    region_center=[float(region["center"][0]), float(region["center"][1])],
                    color=(1, 0, 0, 1),
                    name_prefix=name,
                    recorder=self._spec,
                    spec_path=f"layout.{seg}.cube_pose",
                    # 采样框 = 圆环外接正方形（spawn 内部会把框收缩一个方块半边长）
                    region_half_size=region["r_out"] + self.cube_half_size,
                    generator=self._hb_generator,
                    half_size=self.cube_half_size,
                    max_trials=region["cube_max_trials"],
                    # 两段方块从不同时在场、goal 无碰撞体：不以任何 actor 作障碍，约束全部由下面的区域规则给出
                    include_existing=False,
                    include_goal=False,
                    annulus=region["annulus"],
                    base_band=(base_xy, region["base_lo"], region["base_hi"]),
                    segment_clearance=[(*pegs[seg]["segment"], region["peg_gap"])],
                    push_feasible=(tuple(goals[seg]), min_cg, region["push_len_max"], PUSH_BACKOFF_M,
                                   PEG_PUSH_LATERAL_M, base_xy, region["base_lo"], region["base_hi"]),
                )
            except RuntimeError as exc:
                if not isinstance(exc, SceneGenerationError):
                    raise SceneGenerationError(f"MoveCube xhard4：{label}方块生成失败：{exc}") from exc
                raise
            setattr(self, attr, cube)
        self.cube_init_pose = self.cube.pose
        self.cube_init_pose_2 = self.cube_2.pose
        self._store_goal_poses()

        # 只读记录本局实际生效的区域规则与杆循环尝试次数（N18；不抽随机数，排在全部取值点之后）
        for seg in ("demo", "execution"):
            self._spec.record(f"layout.{seg}.region", dict(
                regions[seg]["decision"], peg_axis_extent_m=list(peg_extent), robot_base_xy=list(ROBOT_BASE_XY),
                push_backoff_m=PUSH_BACKOFF_M, peg_push_lateral_m=PEG_PUSH_LATERAL_M, min_cube_goal_m=min_cg))
            self._spec.record(f"layout.{seg}.region_trials", dict(trials[seg]))

    def _xhard4_region(self, layout, key):
        """取并校验 xhard4 的统一区域 U（计划 2.6）；缺字段或数值不合法时拒绝，不许静默放宽。"""
        cfg = layout["xhard4"]["region"]
        where = f"MoveCube xhard4：decision.{key}.xhard4.region"
        if not isinstance(cfg, dict):
            raise SamplingConfigError(f"{where} 必须是字典，收到 {cfg!r}")
        try:
            center = np.asarray(cfg["center"], dtype=np.float64).reshape(-1)
            r_in, r_out = float(cfg["r_in"]), float(cfg["r_out"])
            base_lo, base_hi = (float(v) for v in cfg["base_dist"])
            nums = {k: float(cfg[k]) for k in ("push_len_max", "peg_gap", "goal_peg_gap")}
            budgets = {k: cfg[k] for k in ("peg_max_trials", "goal_max_trials", "cube_max_trials")}
        except (KeyError, TypeError, ValueError) as exc:
            raise SamplingConfigError(f"{where} 缺字段或类型不对：{exc}") from exc
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise SamplingConfigError(f"{where}.center 必须是两个有限数，收到 {cfg['center']!r}")
        if not (np.isfinite(r_in) and np.isfinite(r_out) and 0.0 <= r_in < r_out):
            raise SamplingConfigError(f"{where} 须满足 0 ≤ r_in < r_out，收到 ({cfg['r_in']!r}, {cfg['r_out']!r})")
        if not (np.isfinite(base_lo) and np.isfinite(base_hi) and 0.0 <= base_lo < base_hi):
            raise SamplingConfigError(f"{where}.base_dist 须为 [lo, hi] 且 0 ≤ lo < hi，收到 {cfg['base_dist']!r}")
        for k, v in nums.items():
            if not (np.isfinite(v) and v >= 0.0):
                raise SamplingConfigError(f"{where}.{k} 必须是 ≥0 的有限数，收到 {cfg[k]!r}")
        for k, v in budgets.items():
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 1:
                raise SamplingConfigError(f"{where}.{k} 必须是 ≥1 的整数，收到 {v!r}")
        return {
            "key": key,
            "center": center,
            "r_in": r_in,
            "r_out": r_out,
            "annulus": ((float(center[0]), float(center[1])), r_in, r_out),
            "base_lo": base_lo,
            "base_hi": base_hi,
            **nums,
            **{k: int(v) for k, v in budgets.items()},
            "decision": copy.deepcopy(cfg),
        }

    def _xhard4_sample_peg_in_region(self, region, yaw_policy, extent, seg_label):
        """V6 xhard4：每次试验按 (抓取点 x, 抓取点 y, yaw) 各抽一次 ``torch.rand``，推出杆根后按杆规则判，
        违反就整组重抽。抓取点在圆环外接正方形里均匀抽（拒绝后即 U 内均匀）。

        返回 ``(杆根 x, 杆根 y, yaw, 尝试次数)``；判定用的杆根与建杆时一样经 float32 平移（``_peg_root_xy``），
        保证回放复核与生成时逐位同判。超过 ``peg_max_trials`` 抛真 ``SceneGenerationError``。
        """
        cx, cy = float(region["center"][0]), float(region["center"][1])
        side = 2.0 * region["r_out"]
        for trial in range(1, region["peg_max_trials"] + 1):
            gx = torch.rand(1, generator=self._hb_generator).item() * side + (cx - region["r_out"])
            gy = torch.rand(1, generator=self._hb_generator).item() * side + (cy - region["r_out"])
            yaw = torch.rand(1, generator=self._hb_generator).item() * (yaw_policy["span_rad"]) - (yaw_policy["offset_rad"])
            root_x = gx + self.length * float(np.cos(yaw))
            root_y = gy + self.length * float(np.sin(yaw))
            root = _peg_root_xy(0.0, root_x, root_y)
            if _peg_region_violation(root, yaw, self.length, extent, region) is None:
                return float(root_x), float(root_y), float(yaw), trial
        raise SceneGenerationError(
            f"MoveCube xhard4：{seg_label}杆 {region['peg_max_trials']} 次重抽全部违反统一区域 U")

    def _xhard4_verify_peg_extent(self, extent):
        """用刚建好的杆的实际碰撞盒与可视盒复核杆规则（V6 统一区域 U）所用的轴线段区间（取二者并集）。

        读 head/tail 两个 link 的 box 形状半长、形状局部位姿与 tail 固定关节的 ``pose_in_parent``／``pose_in_child``，
        得到沿杆朝向（link 局部 x 轴）的实际区间；与 ``_peg_axis_extent`` 不一致就抛 RuntimeError
        （属代码类错误：说明 build_peg 的几何变了而判据没跟上）。
        """
        lo, hi = np.inf, -np.inf
        for link in (self.peg_head, self.peg_tail):
            comp = link._objs[0]
            joint = comp.get_joint()
            # 固定关节只有沿 x 的平移（build_peg）：link 原点在父 link 系的 x = pose_in_parent.x − pose_in_child.x
            offset = 0.0 if comp.get_parent() is None else (
                float(joint.get_pose_in_parent().p[0]) - float(joint.get_pose_in_child().p[0]))
            shapes = [(float(sh.half_size[0]), float(sh.local_pose.p[0])) for sh in comp.get_collision_shapes()]
            for c in comp.get_entity().get_components():
                for rs in getattr(c, "render_shapes", []) or []:
                    if hasattr(rs, "half_size"):
                        shapes.append((float(rs.half_size[0]), float(rs.local_pose.p[0])))
            if not shapes:
                raise RuntimeError(f"MoveCube xhard4：杆 link {link.name} 没有可读的 box 形状，无法复核杆规则所用的杆轴线段")
            for half, local_x in shapes:
                lo = min(lo, offset + local_x - half)
                hi = max(hi, offset + local_x + half)
        if abs(lo - extent[0]) > 1e-6 or abs(hi - extent[1]) > 1e-6:
            raise RuntimeError(
                f"MoveCube xhard4：杆规则用的杆轴线段 {tuple(extent)} 与实际几何 ({lo}, {hi}) 不一致")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 每次初始化各自记一份规格，不复用上一次的结果
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "pegs"):
                return

            # Initialize all 3 pegs

            qpos = np.array(
            [
                0.0,
                0,
                0,
                -np.pi * 4 / 8,
                0,
                np.pi * 2 / 4,
                np.pi / 4,
                0.04,
                0.04,
            ],
            dtype=np.float32,
            )
            self.ways=["peg_push","gripper_push","grasp_putdown"]
            way_idx = self._spec.value(
                f"initializations.{getattr(self, '_native_init_index', 0)}.way_idx",
                torch.randint(len(self.ways), (1,), generator=self._hb_generator).item(),
            )
            self.way = self.ways[way_idx]
            #self.way="gripper_push"

            self.agent.reset(qpos)            
            if self.difficulty == "xhard4":
                # B11：每次初始化清掉上一次抓杆的归约标记（由 grasp_and_lift_peg_side 重新设置）
                self._peg_grasp_flipped = False
                self._peg_grasp_flip_log = []
            self.cube_2.set_pose(sapien.Pose(p=[10,10,1]))#only need the pose!
            self.goal_site_2.set_pose(sapien.Pose(p=[10, -10, 1]))

            

    def evaluate(self,solve_complete_eval=False):
        timestep = self.elapsed_steps
        # flag=is_A_pickup_notB(self,self.peg_head,self.peg_tail)
        # flag2=is_A_pickup_notB(self,self.peg_tail,self.peg_head)
        # flag=is_A_insert_notB(self,self.peg_head,self.peg_tail,self.box)

        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([False])
        

        self.obj_flag=-1
        if self.obj_flag==-1:
            self.grasp_target=self.peg_tail
            self.grasp_target_false=self.peg_head

        else:
            self.grasp_target=self.peg_head
            self.grasp_target_false=self.peg_tail

        self.direction1 = 1 if self.cube_init_pose.p[0][1]-self.goal_site_1_pose_p[0][1] > 0 else -1# relative position
        self.direction2 = 1 if self.cube_init_pose_2.p[0][1]-self.goal_site_2_pose_p[0][1]  > 0 else -1
        # direction -1 push from left 
        # direction 1 push from right +y side / table right side from camera view -> treated as push from right


        if self.way=="peg_push":
            tasks = [
                {
                "func": lambda: is_any_obj_pickup_flag_currentpickup(self, objects=[self.grasp_target,self.grasp_target_false]),
                "name": f"Pick up the peg",
                "subgoal_segment":f"Pick up the peg at <>",
                "choice_label": "pick up the peg",
                "demonstration": True,
                "failure_func":   lambda:[
                                           is_obj_pickup(self, obj=self.cube), 
                                           is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2),],
                "solve": lambda env, planner:grasp_and_lift_peg_side(env, planner, env.grasp_target),
                "segment":self.grasp_target
                },
                {
                "func": lambda:  is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2,must_gripper_open=True),
                "name": f"Hook the cube to the target with the peg",
                "subgoal_segment":f"Hook the cube at <> to the target at <> with the peg",
                "choice_label": "hook the cube to the target with the peg",
                "demonstration": True,
                "failure_func": lambda:None,
                "solve": lambda env, planner:solve_push_to_target_with_peg(env,planner,self.cube,self.goal_site,self.direction1,self.obj_flag),
                "segment":[self.cube,self.goal_site],
                },
                                                {
                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=30),
                "name": "static",
                "subgoal_segment":f"static",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [solve_hold_obj(env, planner, static_steps=30)],
            },
                  {
                    "func": lambda:reset_check(self),
                    "name": "NO RECORD",
                    "subgoal_segment":f"NO RECORD",
                    "demonstration": True,
                    "failure_func": None,
                    "specialflag":"reset pegs",
                    "solve": lambda env, planner: [solve_strong_reset(env,planner)],
                    },
                
                {
                "func": lambda: is_any_obj_pickup_flag_currentpickup(self, objects=[self.grasp_target,self.grasp_target_false]),
                "name": "Pick up the peg",
                "subgoal_segment":f"Pick up the peg at <>",
                "choice_label": "pick up the peg",
                "demonstration": False,
                "failure_func":   lambda:[
                                           is_obj_pickup(self, obj=self.cube), 
                                           is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2),],
                "solve": lambda env, planner:grasp_and_lift_peg_side(env, planner, env.grasp_target),
                "segment":self.grasp_target
                },
                {
                "func": lambda:  is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2,must_gripper_open=True),
                "name": f"Hook the cube to the target with the peg",
                "subgoal_segment":f"Hook the cube at <> to the target at <> with the peg",
                "choice_label": "hook the cube to the target with the peg",
                "demonstration": False,
                "failure_func": lambda:None,
                "solve": lambda env, planner:solve_push_to_target_with_peg(env,planner,self.cube,self.goal_site,self.direction2,self.obj_flag),
                "segment":[self.cube,self.goal_site],
                },
                ]
            
            #test using gripper/grasp =false

        if self.way=="gripper_push":
            tasks = [{
                                "func": lambda: is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2,must_gripper_open=True),
                                "name": "Close the gripper and push the cube to the target",
                                "subgoal_segment":f"Close the gripper and push the cube at <> to the target at <>",
                                "choice_label": "close gripper and push the cube to the target",
                                "demonstration": True,
                                "failure_func":  lambda: [is_obj_pickup(self, obj=self.cube),
                                                          is_obj_pickup(self, obj=self.grasp_target),
                                                          is_obj_pickup(self, obj=self.grasp_target_false)],
                                "solve": lambda env, planner:solve_push_to_target(env,planner,self.cube,self.goal_site),
                                "segment":[self.cube,self.goal_site],
                                },
                                                                {
                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=60),
                "name": "static",
                "subgoal_segment":f"static",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [solve_hold_obj(env, planner, static_steps=60)],
            },
                                {
                                "func": lambda:reset_check(self),
                                "name": "NO RECORD",
                                "subgoal_segment":f"NO RECORD",
                                "demonstration": True,
                                "failure_func": None,
                                "specialflag":"reset pegs",
                                "solve": lambda env, planner: [solve_strong_reset(env,planner)],
                                },
                                {
                                "func": lambda: is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2,must_gripper_open=True),
                                "name": "Close the gripper and push the cube to the target",
                                "subgoal_segment":f"Close the gripper and push the cube at <> to the target at <>",
                                "choice_label": "close gripper and push the cube to the target",
                                "demonstration": False,
                                "failure_func":  lambda: [is_obj_pickup(self, obj=self.cube),
                                                          is_obj_pickup(self, obj=self.grasp_target),
                                                          is_obj_pickup(self, obj=self.grasp_target_false)],
                                "solve": lambda env, planner:solve_push_to_target(env,planner,self.cube,self.goal_site),
                                "segment":[self.cube,self.goal_site],
                                },
                                
                                
                                ]
            
        if self.way=="grasp_putdown":
            tasks = [
                {
                        "func": lambda: is_obj_pickup(self, obj=self.cube),
                        "name": "Pick up the cube",
                        "subgoal_segment":f"Pick up the cube at <>",
                        "choice_label": "pick up the cube",
                        "demonstration": True,
                        "failure_func": lambda: [is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2), 
                                                 is_obj_pickup(self, obj=self.grasp_target),
                                                 is_obj_pickup(self, obj=self.grasp_target_false)],
                        "solve": lambda env, planner:[solve_pickup(env, planner, obj=self.cube),],
                        "segment":[self.cube],
                        },
                        {
                    "func": (lambda: is_obj_dropped_onto(self,obj=self.cube,target=self.goal_site)),
                    "name": "place the cube onto the target",
                    "subgoal_segment":f"place the cube onto the target at <>",
                    "choice_label": "place the cube onto the target",
                    "demonstration": True,
                    "failure_func":  None, 
                    "solve": lambda env, planner: [solve_putonto_whenhold(env, planner,target=self.goal_site)],
                                        "segment":[self.goal_site],
                                        },

                                {
                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=60),
                "name": "static",
                "subgoal_segment":f"static",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: [solve_hold_obj(env, planner, static_steps=60)],
            },
                                                    {
                                "func": lambda:reset_check(self),
                                "name": "NO RECORD",
                                "subgoal_segment":f"NO RECORD",
                                "demonstration": True,
                                "failure_func": None,
                                "specialflag":"reset pegs",
                                "solve": lambda env, planner: [solve_strong_reset(env,planner)],
                                },
                {
                        "func": lambda: is_obj_pickup(self, obj=self.cube),
                        "name": "Pick up the cube",
                        "subgoal_segment":f"Pick up the cube at <>",
                        "choice_label": "pick up the cube",
                        "demonstration": False,
                        "failure_func": lambda: [is_obj_pushed_onto(self,self.cube,self.goal_site,distance_threshold=self.cube_half_size*2*1.2), 
                                                 is_obj_pickup(self, obj=self.grasp_target),
                                                 is_obj_pickup(self, obj=self.grasp_target_false)],
                        "solve": lambda env, planner:[solve_pickup(env, planner, obj=self.cube),],
                        "segment":[self.cube],
                        },
                        {
                    "func": (lambda: is_obj_dropped_onto(self,obj=self.cube,target=self.goal_site)),
                    "name": "place the cube onto the target",
                    "subgoal_segment":f"place the cube onto the target at <>",
                    "choice_label": "place the cube onto the target",
                    "demonstration": False,
                    "failure_func":  None, 
                    "solve": lambda env, planner: [solve_putonto_whenhold(env, planner,target=self.goal_site)],
                                        "segment":[self.goal_site],
                                        },

            ]


                            


        # Store task list for RecordWrapper use
        self.task_list = tasks

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

        #allow_subgoal_change_this_timestep=True
        all_tasks_completed, current_task_name, task_failed,_ = sequential_task_check(self, tasks,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

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
        timestep = int(info["elapsed_steps"])

            
        if self.reset_in_proecess==True:
            for i, peg in enumerate(self.pegs):
                peg.set_pose(self.peg_init_poses_2[i])
                if peg.dof > 0:
                    zero = np.zeros(peg.dof)
                    peg.set_qpos(zero)
                    peg.set_qvel(zero)

            self.cube.set_pose(self.cube_init_pose_2)
            #self.goal_site_2.set_pose(sapien.Pose(p=self.goal_site_2_pose_p[0],q=self.goal_site_2_pose_q[0]))
            goal2_p = np.array(self.goal_site_2_pose_p, copy=True)
            goal2_q = np.array(self.goal_site_2_pose_q, copy=True)
            self.goal_site.set_pose(sapien.Pose(p=goal2_p[0],q=goal2_q[0]))
            #print("reset goal site to",goal2_p[0],goal2_q[0])



        return obs, reward, terminated, truncated, info
