import copy
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

from .utils import *
from .utils.difficulty import is_newvalue_difficulty, normalize_robomme_difficulty, require_xhard4_only
from .utils.subgoal_evaluate_func import static_check
from .utils.episode_spec import EpisodeSpecError, SpecRecorder
from .utils.sampling_config import SamplingConfigError, assert_native_decision, fill_missing_newvalue, split_sampling_config
from .utils.SceneGenerationError import SceneGenerationError
from .utils import subgoal_language
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils import reset_panda
from .utils import subgoal_evaluate_func
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


# ── decision／native 两块的原值（newtaskRelease-v3 步 3，映射见方案第二节 2.14）────────
NATIVE_SAMPLING = {
    "parameters": {
        "peg_size": {
            "length_expression": "0.05 + (0.01 - 0.01) * rand()",
            "radius_expression": "0.01 + (0.005 - 0.005) * rand()",
            "note": "两次 rand 被乘 0 消掉，仍须保留以免平移随机流（红线 R8）",
        },
        "head_color": {"sampler": "torch.rand", "shape": [3]},
        "tail_color_rule": "1 - head",
        "target_peg": {
            "sampler": "torch.randint(0, 3)",
            "overridden_to": 0,
            "note": "抽完立刻被 0 覆盖，保留抽样次数",
        },
        "obj_selection": {"sampler": "torch.randint(0, 2)", "mapping": [-1, 1]},
        "direction_selection": {"sampler": "torch.randint(0, 2)", "mapping": [-1, 1]},
        "recovery": "本环境没有 inject_fail_grasp，只接收入口恢复模式，实际恢复动作为 null",
    },
    "positions": {
        "construction_peg": {
            "y_base": -0.15,
            "initial_yaw": 0,
            "note": "构造期临时位姿；真实输入是每次初始化重新采样的位姿",
        },
        "box": {
            "base_translation": [0, 0],
            "jitter_span": 0.2,
            "yaw_center_rad": "np.pi / 2",
            "yaw_half_span_deg": 20,
            "z_factor": 4,
            "inner_radius_factor": 1.7,
            "outer_radius_factor": 4,
        },
        "peg_sampling": {
            "x_span": 0.4, "x_offset": -0.2,
            "y_span": 0.6, "y_offset": -0.3,
            "min_distance_to_box_factor": 6,
            "min_distance_between_pegs_factor": 1.5,
            "max_attempts": 512,
        },
    },
}


# ── V5 xhard：杆与孔板的桌面轮廓几何（计划 2.8 / L24～L26）──────────────────────────
# 几何按实际建模核实（utils/object_generation.py）：
#   * build_peg：根链接 = 杆头，位姿 p 即采样的 xy（下称 root），杆轴 u = (cos yaw, sin yaw)；
#     杆尾链接在 root − length·u。两段视觉盒半尺寸各为 (length/2, radius)，合起来整根杆轮廓是
#     中心 root − (length/2)·u、半尺寸 (length, radius) 的有向矩形（length=0.05, radius=0.01 时即
#     中心 root − 0.025u、半尺寸 (0.05, 0.01)，杆身从 root − 0.075u 到 root + 0.025u）。
#     两段碰撞盒半长 0.9·length/2，被视觉轮廓完全包住，按视觉轮廓量更保守。
#   * build_box_with_hole：四块板局部 x 半长 = depth（这里传的是 length），局部 y 外沿 = outer_radius
#     （= radius·outer_radius_factor）⇒ 孔板轮廓半尺寸 (length, radius·4) = (0.05, 0.04)，朝向 box_yaw。
def _rect_corners(center, yaw, half):
    """有向矩形 (中心, 朝向, 半尺寸) → 4 个顶点（逆时针），float64。"""
    c = np.asarray(center, dtype=np.float64).reshape(2)
    u = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    v = np.array([-u[1], u[0]], dtype=np.float64)
    a = u * float(half[0])
    b = v * float(half[1])
    return np.stack([c + a + b, c - a + b, c - a - b, c + a - b])


def _point_segment_distance(p, a, b):
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom <= 0.0 else min(1.0, max(0.0, float(np.dot(p - a, ab)) / denom))
    return float(np.linalg.norm(p - (a + t * ab)))


def _rects_overlap(ca, cb):
    """分离轴定理：两矩形（含边界接触）有公共点即返回 True。"""
    for corners in (ca, cb):
        for k in range(2):
            edge = corners[k + 1] - corners[k]
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            pa = ca @ axis
            pb = cb @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def footprint_gap(rect_a, rect_b):
    """两个有向矩形轮廓的精确最小距离（米）。

    ``rect = (center_xy, yaw, half_xy)``。SAT 判为相交（含接触）时返回 0；否则两矩形不相交，
    最小距离必在某个顶点到对方某条边之间取到，取 4×4×2 = 32 个「顶点到边」距离的最小值。
    ⚠ 调用方的判据必须用严格不等号（``gap > g``）：相交时恒为 0，写成 ``≥ 0`` 等于没有约束。
    """
    ca = _rect_corners(*rect_a)
    cb = _rect_corners(*rect_b)
    if _rects_overlap(ca, cb):
        return 0.0
    best = float("inf")
    for src, dst in ((ca, cb), (cb, ca)):
        for p in src:
            for j in range(4):
                best = min(best, _point_segment_distance(p, dst[j], dst[(j + 1) % 4]))
    return best


def peg_footprint(root_xy, yaw, length, radius):
    """整根杆的桌面轮廓：中心 root − (length/2)·u，半尺寸 (length, radius)。"""
    root = np.asarray(root_xy, dtype=np.float64).reshape(2)
    u = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    return (root - 0.5 * float(length) * u, float(yaw), (float(length), float(radius)))


def box_footprint(box_xy, box_yaw, depth, outer_radius):
    """孔板的桌面轮廓：中心 box_xy，半尺寸 (depth, outer_radius)，朝向 box_yaw。"""
    return (np.asarray(box_xy, dtype=np.float64).reshape(2), float(box_yaw), (float(depth), float(outer_radius)))


def native_blocks(cls):
    """本环境的 ``(decision, native)`` 原值块；外部导出与内部解析共用同一份。"""
    return _native_decision(cls), copy.deepcopy(NATIVE_SAMPLING)


def _native_decision(cls):
    """按方案第二节 2.14 切出 decision 块（原值阶段等于原值）。

    V4（计划 2.18）：xhard 新值整体挂在顶层 ``xhard`` 子键下（守卫只放行这些键偏离），
    默认值取自 ``cls.configs["xhard4"]``；原三档可见的四个键与 V3 逐字相同。
    V5（计划 2.8）：``xhard`` 子键去掉 ``near_target_distractor``，新增 ``peg_min_pair_gap_m`` /
    ``peg_box_min_gap_m`` / ``peg_x_max_m``；顶层原三档可见部分不动。
    """
    return {
        # 场上杆的总数：原值 3 根（offsets 决定构造期的排布）。
        "peg_count": 3,
        "peg_offsets": [0.1, 0, -0.1],
        # 新增干扰杆相对目标杆的距离与方位约束：本轮不启用。
        "near_target_distractor": None,
        # 杆在桌面内的转角范围：原值 ±45°（表达式 (u*2-1)*radians(45)）。
        "peg_yaw_range": {"half_span_deg": 45},
        # xhard：4 根杆、±180°（V4 A1/B11）；V5 四根同一采样器 + 轮廓间隔（计划 2.8）
        "xhard4": copy.deepcopy(cls.configs["xhard4"]),
    }


def _resolve_sampling_config(cls, override):
    """拆出本实例专属的 decision／native 副本；不抽随机数，必须在 Generator 之前调用。"""
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    assert_native_decision(decision, decision_default, cls.__name__)
    # V6：旧快照缺的新值档子树从源码申报补齐（本环境不加档，只可能补 xhard）；原三档不读这些键
    fill_missing_newvalue(decision, decision_default)
    native["decision"] = decision
    return native


@register_env("InsertPeg", override=True)
class InsertPeg(BaseEnv):

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
    # difficulty 只在 xhard 生效。消费点读 decision（可被 sampling_config 覆盖），这里是默认值来源。
    config_native = {
        "peg_count": 3,
        "peg_offsets": [0.1, 0, -0.1],
        "near_target_distractor": None,
        "peg_yaw_range": {"half_span_deg": 45},
    }
    config_xhard4 = {
        # peg_offsets 的长度决定杆数（构造期临时排布，第 4 根放在 y_base+0.2 处，随后被重采样覆盖）；
        # peg_count 同步为 4：会改变 _load_scene 里那次 randint 的取值域（结果仍被 overridden_to=0 覆盖）
        "peg_count": 4,
        "peg_offsets": [0.1, 0, -0.1, -0.2],
        "peg_yaw_range": {"half_span_deg": 180},
        # V5（计划 2.8）：删除 V4 的 near_target_distractor（L28）；4 根杆由 _xhard_sample_pegs 一个循环抽，
        # 原生两条杆根判据（离孔板中心 > radius*6、杆根两两 > length*1.5）保留，另加两条轮廓间隔：
        # 任意两杆轮廓 > peg_min_pair_gap_m（L24，严格不等号），杆轮廓与孔板轮廓 > peg_box_min_gap_m（L26）。
        "peg_min_pair_gap_m": 0.03,
        "peg_box_min_gap_m": 0.01,
        # L52：杆区 x 上界由原生的 x_offset + x_span = 0.2 收到 0.1（下界 −0.2 与 y 范围不动）
        "peg_x_max_m": 0.1,
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
        self._spec = SpecRecorder(native_episode_spec, "InsertPeg", {"seed": seed},
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
        # V6（计划 2.13 / M2）：本环境原版无梯度、不加档，传入 xhard1/2/3 明确报错
        require_xhard4_only(self.difficulty, "InsertPeg")
        if is_newvalue_difficulty(self.difficulty):
            # V4 B11：抓杆按等价朝向归约，并同步补偿 insert_peg 的局部平移；原三档不设此属性
            self._xhard_peg_yaw_reduction = True

        self.restore_flag=False
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
        # self.length = (0.05 + (0.0 - 0.0) * length_tensor).item()
        # self.radius = (0.01 + (0.0 - 0.0) * radius_tensor).item()
        self.length = (0.05 + (0.01 - 0.01) * length_tensor).item()
        self.radius = (0.01 + (0.005 - 0.005) * radius_tensor).item()

        # Create 3 identical pegs with different x-axis coordinates
        self.pegs = []
        self.peg_heads = []
        self.peg_tails = []
        self._peg_initial_poses = []

        decision_cfg = self._sampling["decision"]
        if is_newvalue_difficulty(self.difficulty):
            # V4：xhard 的 peg_count / peg_offsets / peg_yaw_range 全部改读 xhard 子键（V5 起另有两条间隔与 x 上界，只在 _xhard_sample_pegs 里读）
            decision_cfg = decision_cfg[self.difficulty]
        native_pos = self._sampling["positions"]
        offsets = list(decision_cfg["peg_offsets"])  # X-axis differences for the 3 pegs
        # Sample a single pair of colors so all pegs share the same appearance per seed.
        peg_head_color = self._spec.value(
            "objects.head_rgb", torch.rand(3, generator=self._hb_generator).tolist()
        )
        # Use complementary tail color so head/tail are contrasting.
        peg_tail_color = [1.0 - c for c in peg_head_color]

        for offset in offsets:
            peg_spawn_translation = np.array([self.length / 2 , native_pos["construction_peg"]["y_base"]-offset, self.radius], dtype=np.float32)

            #initial_yaw = (torch.rand(1, generator=self._hb_generator).item() * 2 * np.pi) - np.pi
            initial_yaw =  native_pos["construction_peg"]["initial_yaw"]
            yaw_angles = torch.tensor([[0.0, 0.0, initial_yaw]], dtype=torch.float32)
            yaw_matrix = euler_angles_to_matrix(yaw_angles, convention="XYZ")
            yaw_quat = matrix_to_quaternion(yaw_matrix)[0].detach().cpu().numpy().tolist()

            peg_initial_pose = sapien.Pose(
                p=peg_spawn_translation.tolist(),
                q=yaw_quat,
            )

            peg, peg_head, peg_tail = build_peg(
                self,
                length=self.length,
                radius=self.radius,
                initial_pose=peg_initial_pose,
                name=f"peg_{len(self.pegs)}",
                head_color=peg_head_color,
                tail_color=peg_tail_color,
            )

            self.pegs.append(peg)
            self.peg_heads.append(peg_head)
            self.peg_tails.append(peg_tail)
            self._peg_initial_poses.append(peg_initial_pose)

        # Randomly select one peg from the 3 pegs
        target_cfg = self._sampling["parameters"]["target_peg"]
        # 这次抽样的结果原本就被 0 覆盖；记进 sampling_trace 以证明它照常发生（红线 R8）
        random_peg_idx = self._spec.value(
            "objects.sampling_trace.random_peg_idx",
            int(torch.randint(0, decision_cfg["peg_count"], (1,), generator=self._hb_generator).item()),
            decision_key=f"{self.difficulty}.peg_count" if is_newvalue_difficulty(self.difficulty) else None,
        )
        random_peg_idx=target_cfg["overridden_to"]
        self.peg = self.pegs[random_peg_idx]

        self.peg_head = self.peg_heads[random_peg_idx]
        self.peg_tail = self.peg_tails[random_peg_idx]
        self._peg_initial_pose = self._peg_initial_poses[random_peg_idx]


        box_cfg = native_pos["box"]
        self.box=build_box_with_hole(self,inner_radius=self.radius*box_cfg["inner_radius_factor"],outer_radius=self.radius*box_cfg["outer_radius_factor"],depth=self.length,center=list(box_cfg["base_translation"]))
        
        self.reset_in_proecess=False
   

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 每次初始化各自记一份规格，不复用上一次的结果
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        with torch.device(self.device):
            self.end_steps=None
            self.table_scene.initialize(env_idx)
            # Reset highlight state at the start of each episode
            self._insert_highlight_start = None
            self._insert_highlight_active = False

            if not hasattr(self, "pegs"):
                return

            box_cfg = self._sampling["positions"]["box"]
            peg_sampling = self._sampling["positions"]["peg_sampling"]
            base_translation = tuple(box_cfg["base_translation"])
            x_jitter_2 = (torch.rand(1, generator=self._hb_generator).item() - 0.5) * box_cfg["jitter_span"]
            y_jitter_2 = (torch.rand(1, generator=self._hb_generator).item() - 0.5) * box_cfg["jitter_span"]
            # x_jitter_2=0
            # y_jitter_2=0
            x_jitter_2, y_jitter_2 = self._spec.value(
                f"initializations.{self._native_init_index}.box_jitter", [x_jitter_2, y_jitter_2]
            )
            box_translation = [base_translation[0] + x_jitter_2, base_translation[1] + y_jitter_2, self.radius * box_cfg["z_factor"]]
            box_yaw = np.pi / 2 + (torch.rand(1, generator=self._hb_generator).item() * 2 - 1) * np.radians(box_cfg["yaw_half_span_deg"])
            box_yaw = self._spec.value(f"initializations.{self._native_init_index}.box_yaw", box_yaw)
            box_angles = torch.tensor([[0.0, 0.0, box_yaw]], dtype=torch.float32)
            box_matrix = euler_angles_to_matrix(box_angles, convention="XYZ")
            box_quat = matrix_to_quaternion(box_matrix)[0].detach().cpu().numpy().tolist()
            self.box.set_pose(sapien.Pose(p=box_translation, q=box_quat))

            box_xy = np.array(box_translation[:2], dtype=np.float32)
            sampled_xy_positions = []
            max_sampling_attempts = peg_sampling["max_attempts"]

            xhard = is_newvalue_difficulty(self.difficulty)
            if xhard:
                xhard_cfg = self._sampling["decision"][self.difficulty]
                yaw_half_span_deg = xhard_cfg["peg_yaw_range"]["half_span_deg"]
                # V5（计划 2.8 / L27a）：4 根杆全部交给 _xhard_sample_pegs 一个循环抽（紧接在下面这个原生
                # 循环之后、obj/dir 之前），原生循环在 xhard 下一根都不跑；循环体逐字保留不动。
                uniform_pegs = []
                yaw_dk = f"{self.difficulty}.peg_yaw_range"
            else:
                yaw_half_span_deg = self._sampling["decision"]["peg_yaw_range"]["half_span_deg"]
                uniform_pegs = self.pegs
                yaw_dk = None

            #Initialize all 3 pegs with constrained random placements
            for i, peg in enumerate(uniform_pegs):
                candidate_xy = None
                for _ in range(max_sampling_attempts):
                    x_sample = (torch.rand(1, generator=self._hb_generator).item() * peg_sampling["x_span"]) + peg_sampling["x_offset"]
                    y_sample = (torch.rand(1, generator=self._hb_generator).item() * peg_sampling["y_span"]) + peg_sampling["y_offset"]
                    sampled_xy = np.array([x_sample, y_sample], dtype=np.float32)

                    if np.linalg.norm(sampled_xy - box_xy) <= self.radius * peg_sampling["min_distance_to_box_factor"]:
                        continue

                    if any(np.linalg.norm(sampled_xy - prev_xy) <= self.length * peg_sampling["min_distance_between_pegs_factor"] for prev_xy in sampled_xy_positions):
                        continue

                    candidate_xy = sampled_xy
                    break

                if candidate_xy is None:
                    if xhard:
                        # 2.2④：xhard 放不下即判该局失败（任务性失败，可换 seed），不静默截断
                        raise SceneGenerationError(
                            f"InsertPeg xhard：peg_{i} 在 {max_sampling_attempts} 次内找不到满足判据的位置")
                    raise RuntimeError("Failed to sample peg positions satisfying placement constraints.")

                yaw_value = (torch.rand(1, generator=self._hb_generator).item() * 2 - 1) * np.radians(yaw_half_span_deg)

                # 拒绝采样的失败尝试照常发生；这里冻结被接受的位姿
                candidate_xy_list, yaw_value = self._spec.value(
                    f"initializations.{self._native_init_index}.pegs.{i}",
                    [[float(candidate_xy[0]), float(candidate_xy[1])], yaw_value],
                    decision_key=yaw_dk,
                )
                candidate_xy = np.array(candidate_xy_list, dtype=np.float32)
                yaw_angles = torch.tensor([[0.0, 0.0, yaw_value]], dtype=torch.float32)
                yaw_matrix = euler_angles_to_matrix(yaw_angles, convention="XYZ")
                yaw_quat = matrix_to_quaternion(yaw_matrix)[0].detach().cpu().numpy().tolist()

                pose = sapien.Pose(p=[float(candidate_xy[0]), float(candidate_xy[1]), 0.0], q=yaw_quat)
                peg.set_pose(pose)
                sampled_xy_positions.append(candidate_xy)

            if xhard:
                self._xhard_sample_pegs(box_xy, box_yaw, xhard_cfg)

            # Store initial poses for all pegs
            self.peg_init_poses = []
            for peg in self.pegs:
                pose = peg.pose
                pose_p = np.asarray(pose.p, dtype=np.float32).reshape(-1).copy()
                pose_q = np.asarray(pose.q, dtype=np.float32).reshape(-1).copy()
                self.peg_init_poses.append(sapien.Pose(p=pose_p, q=pose_q))


            # robomme-v2.7/robomme/robomme_env/PickPeg.py:243
            pose = self.peg.pose
            pose_p = np.asarray(pose.p, dtype=np.float32).reshape(-1).copy()
            pose_q = np.asarray(pose.q, dtype=np.float32).reshape(-1).copy()
            self.peg_init_pose = sapien.Pose(p=pose_p, q=pose_q)
            self.peg_init_pose = sapien.Pose(p=pose_p, q=pose_q)


                    # Define task list, each task contains a dictionary with function, name, demonstration flag, and optional failure_func
            obj_sample = self._spec.value(
                f"initializations.{self._native_init_index}.obj_sample",
                int(torch.randint(0, 2, (1,), generator=self._hb_generator).item()),
            )
            dir_sample = self._spec.value(
                f"initializations.{self._native_init_index}.dir_sample",
                int(torch.randint(0, 2, (1,), generator=self._hb_generator).item()),
            )
            # 两个采样值现在是规格里的整数（原为张量），映射语义不变
            self.obj_flag = -1 if obj_sample == 0 else 1
            self.direction = -1 if dir_sample == 0 else 1

            if xhard:
                # V5：V4 的「第 4 根杆贴近目标杆」已删除（L28），这里只重置抓取翻转记录（B11）
                self._peg_grasp_flipped = False
                self._peg_grasp_flip_log = []
            
            # if self.seed<30:
            #     self.obj_flag=-1
            #     self.direction=1
            # elif self.seed<60:
            #     self.obj_flag=-1
            #     self.direction=-1
            # elif self.seed<90:
            #     self.obj_flag=1
            #     self.direction=1
            # #elif self.seed<60:
            # else:
            #     self.obj_flag=1
            #     self.direction=-1


            # self.obj_flag=1
            # self.direction=-1


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

            self.agent.reset(qpos)            
            if self.obj_flag==-1:
                self.grasp_target=self.peg_head
                self.insert_target=self.peg_tail
            else:
                self.grasp_target=self.peg_tail
                self.insert_target=self.peg_head

            agent_x = self.agent.robot.pose.p.tolist()[0][0]
            head_x = float(self.peg_head.pose.p.tolist()[0][0])
            tail_x = float(self.peg_tail.pose.p.tolist()[0][0])
            logger.debug(f"agent_x: {agent_x}, head_x: {head_x}, tail_x: {tail_x}")
            if xhard:
                # V6 审查修复 D6（用户 K10「d6修」）：near/far 按机器人基座到杆两端的 XY 欧氏距离判定，
                # 不再只比 x 轴（审查 xhard4 ep5 第二布局：x 轴判头近、欧氏距离头 0.61363 m > 尾 0.63401 m 反向）；原三档不动
                base_xy = np.asarray(self.agent.robot.pose.p.tolist()[0][:2], dtype=np.float64)
                head_xy = np.asarray(self.peg_head.pose.p.tolist()[0][:2], dtype=np.float64)
                tail_xy = np.asarray(self.peg_tail.pose.p.tolist()[0][:2], dtype=np.float64)
                head_dist = float(np.linalg.norm(head_xy - base_xy))
                tail_dist = float(np.linalg.norm(tail_xy - base_xy))
                logger.debug(f"xhard near/far by euclidean XY: head_dist={head_dist:.5f}, tail_dist={tail_dist:.5f}")
                near_link = self.peg_head if head_dist <= tail_dist else self.peg_tail
            else:
                near_link = self.peg_head if abs(head_x - agent_x) <= abs(tail_x - agent_x) else self.peg_tail

            self.grasp_target_distance = "near" if self.grasp_target is near_link else "far"
            logger.debug(f"grasp_target_distance: {self.grasp_target_distance}")

            self.insert_way="left" if self.direction == -1 else "right"
            tasks = [
                {
                    "func": lambda: is_A_pickup_notB(self, self.grasp_target, self.insert_target),
                    "name": f"Pick up the peg by grasping the {self.grasp_target_distance} end",
                    "subgoal_segment":f"Pick up the peg by grasping the {self.grasp_target_distance} end at <>",
                    "choice_label": "pick up the peg by grasping one end",
                    "demonstration": True,
                    "failure_func": lambda: is_A_pickup_notB(self, self.insert_target, self.grasp_target),
                    "solve": lambda env, planner: grasp_and_lift_peg_side(env, planner, env.grasp_target),
                    "segment":self.grasp_target
                },
                {
                    "func": lambda: is_A_insert_notB(self, self.insert_target, self.grasp_target, self.box,direction=self.direction),
                    "name": f"Insert the peg from the {self.insert_way} side of the box",
                    "subgoal_segment":f"Insert the peg from the {self.insert_way} side of the box at <>",
                    "choice_label": f"insert the peg from the {self.insert_way} side",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: insert_peg(env, planner,  direction=self.direction,obj=self.obj_flag,insert_obj=self.insert_target),
                    "segment":self.box
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
                    "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=100),
                    "name": "NO RECORD",
                    "subgoal_segment":"NO RECORD",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [solve_hold_obj(env, planner, static_steps=100,close=False)],
                },

            {
                    "func": lambda: is_A_pickup_notB(self, self.grasp_target, self.insert_target),
                    "name": f"Pick up the peg by grasping the {self.grasp_target_distance} end",
                    "subgoal_segment":f"Pick up the peg by grasping the {self.grasp_target_distance} end at <>",
                    "choice_label": "pick up the peg by grasping one end",
                    "demonstration": False,
                    "failure_func": lambda: [
                        is_A_pickup_notB(self, self.insert_target, self.grasp_target),
                        is_any_obj_pickup(self, [head for i, head in enumerate(self.peg_heads) if self.pegs[i] is not self.peg] +
                                            [tail for i, tail in enumerate(self.peg_tails) if self.pegs[i] is not self.peg])
                    ],
                    "solve": lambda env, planner: grasp_and_lift_peg_side(env, planner, env.grasp_target),
                    "segment":self.grasp_target
                },
                {
                    "func": lambda: is_A_insert_notB(self, self.insert_target, self.grasp_target,self.box,direction=self.direction,mark_end_flag=True),
                    "name": f"Insert the peg from the {self.insert_way} side",
                    "subgoal_segment":f"Insert the peg from the {self.insert_way} side at <>",
                    "choice_label": f"insert the peg from the {self.insert_way} side",
                    "demonstration": False,
                    "failure_func": lambda: [
                        is_A_insert_notB(self, self.grasp_target, self.insert_target, self.box),
                        is_A_insert_notB(self, self.insert_target, self.grasp_target, self.box, direction=-self.direction),
                        is_any_obj_pickup(self, [head for i, head in enumerate(self.peg_heads) if self.pegs[i] is not self.peg] +
                                            [tail for i, tail in enumerate(self.peg_tails) if self.pegs[i] is not self.peg])
                    ],
                    "solve": lambda env, planner: insert_peg(env, planner,  direction=self.direction,obj=self.obj_flag,insert_obj=self.insert_target,cut_retreat=True),
                    "segment":self.box
                },
            ]

            # Store task list for RecordWrapper use
            self.task_list = tasks

    def _xhard_sample_pegs(self, box_xy, box_yaw, xhard_cfg):
        """V5 xhard（计划 2.8 / L24～L29 / L52）：4 根杆一个循环、同一套规则抽位姿。

        每根杆最多 ``max_attempts``（512）次尝试，每次：
          1. 按原生取法抽 x、y 各一次 rand（x 上界收到 ``peg_x_max_m``，L52；y 与原生相同）；
          2. 原生两条杆根判据（离孔板中心 ≤ radius*6 或离任一已放杆根 ≤ length*1.5 → 重抽，L25 保留）；
          3. 通过后才抽 yaw（lazy，±half_span_deg）；
          4. 杆轮廓与孔板轮廓 ``footprint_gap ≤ peg_box_min_gap_m`` → 重抽（L26）；
          5. 与任一已放杆 ``footprint_gap ≤ peg_min_pair_gap_m`` → 重抽（L24，严格不等号）。
        耗尽即抛 ``SceneGenerationError``（任务性失败，可换 seed）。只在被接受的那次调用 ``_spec.value``，
        尝试次数与实测最小间隔用 ``record`` 留痕（N18）；回放冻结规格时对冻结值复核两种间隔，
        不合格抛 ``EpisodeSpecError``（N17 / L29）。目标恒为 peg_0（只是第一个被抽的）。
        """
        peg_sampling = self._sampling["positions"]["peg_sampling"]
        max_attempts = peg_sampling["max_attempts"]
        x_lo = peg_sampling["x_offset"]
        x_span = float(xhard_cfg["peg_x_max_m"]) - x_lo
        if not x_span > 0:
            raise SamplingConfigError(f"InsertPeg xhard：peg_x_max_m 必须大于 x 下界 {x_lo}")
        yaw_half = np.radians(xhard_cfg["peg_yaw_range"]["half_span_deg"])
        pair_gap = float(xhard_cfg["peg_min_pair_gap_m"])
        box_gap = float(xhard_cfg["peg_box_min_gap_m"])
        box_to_root = self.radius * peg_sampling["min_distance_to_box_factor"]
        root_to_root = self.length * peg_sampling["min_distance_between_pegs_factor"]
        box_cfg = self._sampling["positions"]["box"]
        box_fp = box_footprint(box_xy, box_yaw, self.length, self.radius * box_cfg["outer_radius_factor"])
        prefix = f"initializations.{self._native_init_index}"

        placed_xy = []
        placed_fp = []
        attempts_log = []
        for i, peg in enumerate(self.pegs):
            accepted = None
            attempts = 0
            for attempts in range(1, max_attempts + 1):
                x_sample = (torch.rand(1, generator=self._hb_generator).item() * x_span) + x_lo
                y_sample = (torch.rand(1, generator=self._hb_generator).item() * peg_sampling["y_span"]) + peg_sampling["y_offset"]
                sampled_xy = np.array([x_sample, y_sample], dtype=np.float32)
                if np.linalg.norm(sampled_xy - box_xy) <= box_to_root:
                    continue
                if any(np.linalg.norm(sampled_xy - prev_xy) <= root_to_root for prev_xy in placed_xy):
                    continue
                yaw_try = (torch.rand(1, generator=self._hb_generator).item() * 2 - 1) * yaw_half
                fp = peg_footprint(sampled_xy, yaw_try, self.length, self.radius)
                if footprint_gap(fp, box_fp) <= box_gap:
                    continue
                if any(footprint_gap(fp, prev_fp) <= pair_gap for prev_fp in placed_fp):
                    continue
                accepted = (sampled_xy, yaw_try)
                break
            if accepted is None:
                raise SceneGenerationError(
                    f"InsertPeg xhard：peg_{i} 在 {max_attempts} 次内找不到满足判据（杆根 + 轮廓间隔）的位置")
            attempts_log.append(attempts)
            candidate_xy, yaw_value = accepted
            # 拒绝采样的失败尝试照常发生；这里冻结被接受的位姿
            candidate_xy_list, yaw_value = self._spec.value(
                f"{prefix}.pegs.{i}",
                [[float(candidate_xy[0]), float(candidate_xy[1])], yaw_value],
                decision_key=f"{self.difficulty}.peg_yaw_range",
            )
            candidate_xy = np.array(candidate_xy_list, dtype=np.float32)
            fp = peg_footprint(candidate_xy, yaw_value, self.length, self.radius)
            # N17：回放时这里是冻结值，必须重查两种间隔（导出时恒成立，复查不抽随机数）
            gap_box = footprint_gap(fp, box_fp)
            if not gap_box > box_gap:
                raise EpisodeSpecError(
                    f"InsertPeg xhard：peg_{i} 与孔板轮廓间隔 {gap_box:.4f} m 不大于 {box_gap} m（冻结规格违反 V5 规则）")
            for j, prev_fp in enumerate(placed_fp):
                gap_pair = footprint_gap(fp, prev_fp)
                if not gap_pair > pair_gap:
                    raise EpisodeSpecError(
                        f"InsertPeg xhard：peg_{i} 与 peg_{j} 轮廓间隔 {gap_pair:.4f} m 不大于 {pair_gap} m（冻结规格违反 V5 规则）")
            yaw_angles = torch.tensor([[0.0, 0.0, yaw_value]], dtype=torch.float32)
            yaw_quat = matrix_to_quaternion(euler_angles_to_matrix(yaw_angles, convention="XYZ"))[0].detach().cpu().numpy().tolist()
            peg.set_pose(sapien.Pose(p=[float(candidate_xy[0]), float(candidate_xy[1]), 0.0], q=yaw_quat))
            placed_xy.append(candidate_xy)
            placed_fp.append(fp)

        self._spec.record(f"{prefix}.peg_attempts", attempts_log)
        self._spec.record(f"{prefix}.min_pair_gap_m", min((
            footprint_gap(placed_fp[a], placed_fp[b])
            for a in range(len(placed_fp)) for b in range(a + 1, len(placed_fp))), default=None))
        self._spec.record(f"{prefix}.min_box_gap_m", min((footprint_gap(item, box_fp) for item in placed_fp), default=None))

    def evaluate(self,solve_complete_eval=False):
        timestep = self.elapsed_steps


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

        all_tasks_completed, current_task_name, task_failed,_ = sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)


        if self.end_steps!=None:# truncate tail, also truncate tail in planner
            logger.debug(
                "elapsed_steps=%s, end_steps=%s",
                self.elapsed_steps,
                self.end_steps,
            )
            if int(getattr(self, "elapsed_steps", 0))>=self.end_steps+3:
                 self.successflag = torch.tensor([True])

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
        cur_step = int(self.elapsed_steps[0].item())



        if self.reset_in_proecess==True:
            for i, peg in enumerate(self.pegs):
                peg.set_pose(self.peg_init_poses[i])
                if peg.dof > 0:
                    zero = np.zeros(peg.dof)
                    peg.set_qpos(zero)
                    peg.set_qvel(zero)
              
            logger.debug("reset peg!")



        if is_A_insert_notB(self, self.insert_target, self.grasp_target, self.box,direction=self.direction):
            self.start_step=cur_step


        color=sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5)
        if getattr(self, "start_step", None) is not None :
            if cur_step <=  self.start_step + 20 and cur_step>= self.start_step:
               color=[1.0, 0.0, 0.0, 1.0]


        highlight_obj(
                self,
                self.box,
                start_step= 0,
                end_step= 99999,
                cur_step=cur_step,
                disk_radius=0.015,
                disk_half_length=0.055,
                highlight_color=color,)
        return obs, reward, terminated, truncated, info
