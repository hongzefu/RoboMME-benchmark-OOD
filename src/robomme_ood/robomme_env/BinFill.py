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
from mani_skill.utils.structs import Actor, Link
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
from .utils.object_generation import _build_new_cube_obb2d, _obb2d_intersect
from .utils.xhard import cube_obb2d_exact
from .utils.episode_spec import EpisodeSpecError
from .utils import reset_panda
from .utils import subgoal_language
from .utils.difficulty import normalize_robomme_difficulty, is_newvalue_difficulty
from .utils.episode_spec import SpecRecorder
from .utils.sampling_config import (
    SamplingConfigError,
    assert_native_decision,
    fill_missing_newvalue,
    split_sampling_config,
)
from .utils.SceneGenerationError import SceneGenerationError

from ..logging_utils import logger


# ── 原版采样输入的原值快照（newtask-v2 10.0）────────────────────────────────────
# 本字典就是不传 sampling_config 时的运行默认值，同时也是
# `scripts/generate_dataset_newseed.py --extract-config` 的 AST 提取目标，
# 因此「提取到的原值」与「实际跑的默认值」永远是同一处，不存在双真值。
# 难度字典不在这里重复，仍以类属性 config_easy / config_medium / config_hard 为准。
# 表达式与来源说明字段只作核查用，运行时不消费。
# ── decision／native 两块的原值（newtaskRelease-v3 步 3）──────────────────────
# decision 是第二节字段表里标「是，拟修改」的参数；原值对拍阶段它也必须等于原值（红线 R7）。
# 取值全部来自类属性 config_easy / config_medium / config_hard，这里不另起一套数值，
# 避免两份真值漂移；native 保留原随机规则与常量。
def native_blocks(cls):
    """本环境的 ``(decision, native)`` 原值块；外部导出与内部解析共用同一份，杜绝两套真值。"""
    native = copy.deepcopy(NATIVE_SAMPLING)
    native["parameters"]["put_in_color"] = {
        difficulty: list(cfg["put_in_color"]) for difficulty, cfg in cls.configs.items()
    }
    return _native_decision(cls), native


def _native_decision(cls):
    """从类属性的难度配置里切出 decision 块：色数、生成总数、投入总数。

    V4：xhard 档另带 ``layout_mode``（``"clutter"``），落在 ``configs.xhard`` 之下——
    守卫 ``assert_native_decision`` 会把任意深度名为 ``xhard`` 的子树剥掉再与原值比，
    所以原三档可见的部分（含顶层 ``layout_mode``）逐字不变。
    """
    def _entry(cfg):
        entry = {
            "color": cfg["color"],
            "spawn_cubes": list(cfg["spawn_cubes"]),
            "put_in_numbers": list(cfg["put_in_numbers"]),
        }
        if "layout_mode" in cfg:
            # 只有 config_xhard4 带这个键；原三档的字典里没有，输出与改动前逐字相同。
            entry["layout_mode"] = cfg["layout_mode"]
        if "color_mix" in cfg:
            # V5（L41）：只有 config_xhard4 带同色成团上限；原三档没有这个键，输出不变。
            entry["color_mix"] = dict(cfg["color_mix"])
        return entry

    return {
        # 方块摆放模式（原三档）：原值＝由 native.parameters.dynamic 的 randint 决定。
        # xhard 的摆放模式单独写在 configs.xhard.layout_mode（V4 新增 clutter）。
        "layout_mode": "native_dynamic",
        "configs": {
            difficulty: _entry(cfg)
            for difficulty, cfg in cls.configs.items()
        },
    }


# V4 xhard 支持的摆放模式 → 是否动态出现（D6：clutter＝全部方块开局即在场，dynamic 固定 False）。
# 这里只列已实现的模式；外部配置给出表外的值直接拒绝，不静默回退。
XHARD_LAYOUT_DYNAMIC = {"clutter": False}


def _board_strips_obb2d(board):
    """孔板四条边的 2D 障碍（pad=0），与 ``spawn_random_cube`` 内对 ``board_with_hole`` 的特判逐式相同。

    只读位姿、不抽随机数；供 V5 xhard 的槽位几何采样使用（槽位阶段不建 actor，不能借 spawn 函数组障碍）。
    """
    board_side = board._board_side
    hole_side = board._hole_side
    actor_pos = board.pose.p
    if isinstance(actor_pos, torch.Tensor):
        actor_pos = actor_pos[0].detach().cpu().numpy()
    board_center = np.array(actor_pos[:2], dtype=np.float64)
    board_half = board_side / 2
    hole_half = hole_side / 2
    out = []
    if board_half > hole_half:
        top_height = board_half - hole_half
        A_top = np.eye(2)
        h_top = np.array([board_half, top_height / 2])
        out.append((board_center + np.array([0, hole_half + top_height / 2]), A_top, h_top))
        out.append((board_center + np.array([0, -(hole_half + top_height / 2)]), A_top, h_top))
        left_width = board_half - hole_half
        h_left = np.array([left_width / 2, hole_half])
        out.append((board_center + np.array([-(hole_half + left_width / 2), 0]), A_top, h_left))
        out.append((board_center + np.array([hole_half + left_width / 2, 0]), A_top, h_left))
    return out


def _max_same_color_component(xy, labels, link):
    """V5（L40 b / L41）：最大同色连通团的块数。纯函数，不抽随机数。

    两块同色且中心距 ``<= link``（米）即相连；返回所有同色连通分量中最大的块数（空输入为 0）。
    ``xy`` 形状 ``(n, 2)``，``labels`` 长度 ``n``（任意可比较的颜色标签）。
    """
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n == 0:
        return 0
    labels = list(labels)
    if len(labels) != n:
        raise ValueError(f"_max_same_color_component: 坐标 {n} 个，颜色标签 {len(labels)} 个")
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    seen = [False] * n
    best = 0
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack, size = [start], 0
        while stack:
            u = stack.pop()
            size += 1
            for v in range(n):
                if not seen[v] and labels[v] == labels[u] and dist[u, v] <= link:
                    seen[v] = True
                    stack.append(v)
        best = max(best, size)
    return best


def _validate_color_mix(color_mix):
    """V5（L41）：校验 ``decision.configs.xhard.color_mix`` 的取值（键结构已由守卫的形状检查保证）。"""
    if not isinstance(color_mix, dict):
        raise SamplingConfigError(f"BinFill: xhard 的 color_mix 必须是字典，收到 {color_mix!r}")
    max_component = color_mix.get("max_component")
    link_m = color_mix.get("link_m")
    max_redraws = color_mix.get("max_redraws")
    if isinstance(max_component, bool) or not isinstance(max_component, int) or max_component < 1:
        raise SamplingConfigError(f"BinFill: color_mix.max_component 必须是 ≥1 的整数，收到 {max_component!r}")
    if isinstance(link_m, bool) or not isinstance(link_m, (int, float)) or not np.isfinite(link_m) or link_m < 0:
        raise SamplingConfigError(f"BinFill: color_mix.link_m 必须是 ≥0 的有限数，收到 {link_m!r}")
    if isinstance(max_redraws, bool) or not isinstance(max_redraws, int) or max_redraws < 0:
        raise SamplingConfigError(f"BinFill: color_mix.max_redraws 必须是 ≥0 的整数，收到 {max_redraws!r}")


NATIVE_SAMPLING = {
    "parameters": {
        # put_in_color 属于 native（第二节：投入哪些颜色的规则不改，只外部生成本局值），
        # 按难度取自同一套类属性配置。
        "put_in_color": "FROM_CLASS_CONFIGS",
        "dynamic": {
            "sampler": "torch.randint",
            "low": 0,
            "high_exclusive": 2,
            "shape": [1],
            "cast": "bool",
        },
    },
    "positions": {
        "button": {
            "center_xy": [-0.2, 0],
            "randomize": True,
            "randomize_range": [0.1, 0.4],
            "sampling_expression": "(torch.rand(2, generator=generator) - 0.5) * randomize_range",
            "scale": 1.5,
            "randomize_range_origin": "原值取自 build_button 形参默认值；原调用点未传，现由本快照显式传入",
        },
        "board": {
            "base_position": [0.15, 0, 0],
            "x_offset": {"scale": 0.2, "subtract": 0.2},
            "y_offset": {"scale": 0.4, "subtract": 0.2},
            "yaw_deg": {"scale": 40, "subtract": 20},
            "x_expression": "0.15 + (u * 0.2 - 0.2)",
            "y_expression": "u * 0.4 - 0.2",
            "yaw_expression": "u * 40.0 - 20.0",
            "board_side": 0.1,
            "hole_side": 0.08,
            "thickness": 0.05,
            "consumer": "build_board_with_hole 只是位置接收方，内部无随机",
        },
        "cubes": {
            "region_center": [-0.1, 0],
            "region_half_size": [0.2, 0.25],
            "random_yaw": True,
            "yaw_range_rad": [0, 6.283185307179586],
            "yaw_expression": "yaw_sample * 2 * np.pi",
            "min_gap": "self.cube_half_size",
            "min_gap_value": 0.02,
            "include_existing": False,
            "include_goal": False,
            "rng_per_trial": ["x", "y", "yaw"],
        },
    },
}


def _resolve_sampling_config(cls, override):
    """准备本实例专属的采样配置副本。

    只做取值、字段校验与深拷贝：全程不调用任何随机数，且必须在 torch.Generator()
    创建之前完成 —— 在这里多抽或少抽一次会平移其后全部取值。
    gymnasium 会把 kwargs 字典的引用存进 env.unwrapped.spec.kwargs，
    因此独立副本只能由这里的 deepcopy 保证。
    """
    decision_default, native_default = native_blocks(cls)
    decision, native = split_sampling_config(override, native_default, decision_default)
    # 第一轮只做原值导出／消费：decision 必须逐键等于原值，否则就是没申报的新用户决策。
    assert_native_decision(decision, decision_default, "BinFill")
    # V6：旧快照（V5 没有 xhard1/2/3 子树）从源码申报补齐缺的新值档，已有的不动
    fill_missing_newvalue(decision, decision_default)
    if decision.get("layout_mode") != "native_dynamic":
        raise SamplingConfigError("BinFill: 本轮只支持原布局模式 native_dynamic")
    # V4：硬守卫只对 xhard 放开，且只放行已实现的模式（clutter）；原三档仍只认顶层 native_dynamic。
    # V6：新值族四档（xhard1/2/3/xhard）逐档做同样的校验；原三档没有新值条目，不进这里。
    for tier_name, xhard_decision in decision.get("configs", {}).items():
        if not is_newvalue_difficulty(tier_name):
            continue
        if xhard_decision.get("layout_mode") not in XHARD_LAYOUT_DYNAMIC:
            raise SamplingConfigError(
                f"BinFill: {tier_name} 的 layout_mode 只支持 {sorted(XHARD_LAYOUT_DYNAMIC)}，"
                f"收到 {xhard_decision.get('layout_mode')!r}"
            )
        # V5（L41）：同色成团上限的取值校验
        _validate_color_mix(xhard_decision.get("color_mix"))
    native["parameters"].setdefault("put_in_color", native_default["parameters"]["put_in_color"])
    # V6：旧快照的 native.put_in_color 同样缺新值档（V5 只有 xhard），按源码补齐，否则下面合并时 KeyError；
    # 只补缺的新值档键，原三档与已有档的值不动。
    fill_missing_newvalue(native["parameters"]["put_in_color"], native_default["parameters"]["put_in_color"])
    # 消费侧仍按难度读一份合并后的配置：decision 出色数／生成数／投入数，
    # native 出投入颜色数范围，合并结果与改动前的 cls.configs[difficulty] 逐键相同。
    native["parameters"]["configs"] = {
        difficulty: {
            **decision["configs"][difficulty],
            "put_in_color": list(native["parameters"]["put_in_color"][difficulty]),
        }
        for difficulty in decision["configs"]
    }
    native["decision"] = decision
    return native


def _actor_xyz(actor):
    """只读取 actor 当前世界位置，供注入证据用；不改状态、不抽随机数。"""
    pos = actor.pose.p if hasattr(actor, "pose") else actor.get_pose().p
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    return [float(v) for v in np.asarray(pos).reshape(-1)[:3]]


def _actor_quat(actor):
    """只读取 actor 当前世界朝向（wxyz）。"""
    quat = actor.pose.q if hasattr(actor, "pose") else actor.get_pose().q
    if isinstance(quat, torch.Tensor):
        quat = quat.detach().cpu().numpy()
    return [float(v) for v in np.asarray(quat).reshape(-1)[:4]]


def _resolve_episode_spec(spec, task):
    """准备本实例专属的固定规格副本（新值注入）。

    与 :func:`_resolve_sampling_config` 同样的道理：gymnasium 会把 kwargs 字典的引用存进
    ``env.unwrapped.spec.kwargs``，两次初始化又要各自从规格重建工作态，所以这里必须
    deepcopy 出一份独立副本，任务内只改这份工作态，绝不碰调用方传进来的原对象。

    传 ``None``（即没传 ``--episode-specs``）时返回 ``None``，之后每个消费点都走原随机
    路径，链路与改动前逐字相同——``DEFAULT_PARITY`` 靠的就是这一条。
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("episode_spec 必须是字典")
    if spec.get("task") != task:
        raise ValueError(f"episode_spec 是 {spec.get('task')} 的规格，不能用于 {task}")
    return copy.deepcopy(spec)


@register_env("BinFill", override=True)
class BinFill(BaseEnv):

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

    # config_hard = {
    # 'color': 3, 
    # 'spawn_cubes':4,
    #     "put_in_color":3,
    # }

    # config_easy = {
    #     'color': 1, 
    # 'spawn_cubes':8,
    #     "put_in_color":1,
    # }

    # config_medium = {
    #     'color': 3, 
    # 'spawn_cubes':4,
    #     "put_in_color":1,
    # }

    config_easy = {
    'color': 1, 
    'spawn_cubes':[4,6],
    "put_in_color":[1,1],
    "put_in_numbers":[1,3]
    }

    config_medium = {
    'color': 2, 
    'spawn_cubes':[8,10],
    "put_in_color":[1,2],
    "put_in_numbers":[2,4]
    }


    config_hard = {
    'color': 3, 
    'spawn_cubes':[10,12],
    "put_in_color":[2,3],
    "put_in_numbers":[3,5]
    }




    # Combine into a dictionary
    # V4 xhard（派生自 hard，计划 2.3）：全部 clutter（D6：dynamic 固定 False，12 块开局即在场）、
    # 12 块、3 色、投入总数 [5,7]；投入颜色数沿用 hard 的 [2,3]（native 规则不改）。
    # 方块区域与间距不动（B1）。put_in 单色最多 7，序数表已扩到 20（E2）。
    # V5（计划 2.12，L41）：color_mix＝同色成团上限——最大同色连通团（中心距 ≤ link_m 相连）超过
    # max_component 块时，在末尾追加 randperm 重排颜色，最多 max_redraws 次，失败取最优。
    config_xhard4 = {
    'color': 3,
    'spawn_cubes':[12,12],
    "put_in_color":[2,3],
    "put_in_numbers":[9,9],
    "layout_mode": "clutter",
    "color_mix": {"max_component": 3, "link_m": 0.09, "max_redraws": 64},
    }

    # V6：四档沿用杂乱布局、精确 OBB 与同色团限制，只按档设置投入总数 6/7/8/9；总块固定 12。
    config_xhard1 = {
    'color': 3,
    'spawn_cubes':[12,12],
    "put_in_color":[2,3],
    "put_in_numbers":[6,6],
    "layout_mode": "clutter",
    "color_mix": {"max_component": 3, "link_m": 0.09, "max_redraws": 64},
    }

    config_xhard2 = {
    'color': 3,
    'spawn_cubes':[12,12],
    "put_in_color":[2,3],
    "put_in_numbers":[7,7],
    "layout_mode": "clutter",
    "color_mix": {"max_component": 3, "link_m": 0.09, "max_redraws": 64},
    }

    config_xhard3 = {
    'color': 3,
    'spawn_cubes':[12,12],
    "put_in_color":[2,3],
    "put_in_numbers":[8,8],
    "layout_mode": "clutter",
    "color_mix": {"max_component": 3, "link_m": 0.09, "max_redraws": 64},
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
                     episode_spec=None,
                     native_episode_spec=None,
                     **kwargs):
        # 必须落在任何随机数调用与 super().__init__() 之前
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._episode_spec = _resolve_episode_spec(episode_spec, "BinFill")
        # 步 4：只读导出（native_episode_spec=None）或原值回注（传入冻结规格）。
        # 与上面的旧注入通道相互独立：本记录器只挂在原随机分支上（红线 R9）。
        self._spec = SpecRecorder(native_episode_spec, "BinFill", {"seed": seed},
                                  difficulty=kwargs.get("difficulty"))
        # 初始化序号从 -1 起，_initialize_episode 每次进来先加一；
        # _load_scene 里的取值点用不带序号的路径，所以这里只作兜底。
        self._native_init_index = -1
        # 注入生效的只读证据，供对拍观察器核对「创建输入 vs 创建后位姿」
        self._injection_evidence = {}
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
        normalized_robomme_difficulty = normalize_robomme_difficulty(
            kwargs.pop("difficulty", None)
        )
        self.robomme_failure_recovery = bool(
            kwargs.pop("robomme_failure_recovery", False)
        )
        self.robomme_failure_recovery_mode = kwargs.pop(
            "robomme_failure_recovery_mode", None
        )
        if isinstance(self.robomme_failure_recovery_mode, str):
            self.robomme_failure_recovery_mode = self.robomme_failure_recovery_mode.lower()

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
        self.generator.manual_seed(seed)
        dynamic_cfg = self._sampling["parameters"]["dynamic"]
        if self._episode_spec is None and is_newvalue_difficulty(self.difficulty):
            # V4 xhard（D6）：摆放模式由 decision.configs.xhard.layout_mode 决定，clutter ⇒ dynamic 固定
            # False。**不抽**原来那次 randint：xhard 是新档，其自身随机流整体前移一位可接受；
            # 原三档走下面的 else 分支，随机调用序列逐字不变（N5/H2）。
            layout_mode = self._sampling["parameters"]["configs"][self.difficulty]["layout_mode"]
            self._spec.record("layout.mode", layout_mode)
            self.dynamic = bool(self._spec.value(
                "layout.dynamic", XHARD_LAYOUT_DYNAMIC[layout_mode],
                decision_key=f"configs.{self.difficulty}.layout_mode",
            ))
            self._spec.identity.setdefault("difficulty", getattr(self, "difficulty", None))
        elif self._episode_spec is None:
            self.dynamic=bool(self._spec.value("layout.dynamic", bool(torch.randint(dynamic_cfg["low"], dynamic_cfg["high_exclusive"], tuple(dynamic_cfg["shape"]), generator=self.generator).item())))
            self._spec.identity.setdefault("difficulty", getattr(self, "difficulty", None))
        else:
            # 规格定死 dynamic：不抽这一次 randint。开启态不要求随机流位置与原版对齐——
            # 凡规格覆盖的量都由规格决定，规格没覆盖的量（如 inject_fail_grasp）继续用
            # self.generator，取值会因流位置平移而与原版不同，这些量不在验收范围内。
            self.dynamic=bool(self._episode_spec["layout"]["dynamic"])

        # Track the color order and counts used to describe the language goal.
        self.binfill_language_sequence = []

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
        camera_eye=[1,0,0.4]
        camera_target =[0,0,0.4]
        pose = sapien_utils.look_at(
            eye=camera_eye, target=camera_target
        )

        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Create generator for all randomization
        generator = self.generator

        spec = self._episode_spec
        button_cfg = self._sampling["positions"]["button"]
        if spec is None:
            button_obb = build_button(
                self,
                center_xy=tuple(button_cfg["center_xy"]),
                scale=button_cfg["scale"],
                generator=generator,
                randomize=button_cfg["randomize"],
                randomize_range=tuple(button_cfg["randomize_range"]),
                recorder=self._spec,
                spec_path="layout.button_xy",
            )
        else:
            # 规格给的是**最终**中心；关掉 randomize 后 build_button 不抽随机数，
            # 其余（缩放、travel、连杆、OBB）全走原路径
            button_obb = build_button(
                self,
                center_xy=tuple(spec["layout"]["button_xy"]),
                scale=button_cfg["scale"],
                generator=generator,
                randomize=False,
                randomize_range=tuple(button_cfg["randomize_range"]),
            )
        avoid = [button_obb]

        # Create square board with square hole
        board_cfg = self._sampling["positions"]["board"]
        board_x = board_cfg["x_offset"]
        board_y = board_cfg["y_offset"]
        board_yaw = board_cfg["yaw_deg"]
        board_base = board_cfg["base_position"]
        if spec is None:
            x_var = torch.rand(1, generator=generator).item() * board_x["scale"] - board_x["subtract"]  # [-0.25, 0.25]
            y_var = torch.rand(1, generator=generator).item() * board_y["scale"] - board_y["subtract"]  # [-0.25, 0.25]
            z_rot_deg = (torch.rand(1, generator=generator).item() * board_yaw["scale"] - board_yaw["subtract"])  # [-20, 20] degrees
            # 三次抽样照常发生；回注模式下真正用于建板的是冻结值
            x_var, y_var, z_rot_deg = self._spec.value("layout.board.offsets", [x_var, y_var, z_rot_deg])
        else:
            # 规格存的是最终位置，这里反解回原公式里的偏移量，位置计算仍走下面同一行
            board_spec = spec["layout"]["board"]
            x_var = float(board_spec["xy"][0]) - float(board_base[0])
            y_var = float(board_spec["xy"][1]) - float(board_base[1])
            z_rot_deg = float(board_spec["yaw_deg"])
        z_rot_rad = torch.deg2rad(torch.tensor(z_rot_deg))
        # Create rotation quaternion for z-axis rotation
        rot_mat = euler_angles_to_matrix(torch.tensor([[0.0, 0.0, z_rot_rad]]), convention="XYZ")
        rot_quat = matrix_to_quaternion(rot_mat)[0]  # [w, x, y, z]
        self.board_with_hole = build_board_with_hole(
            self,
            board_side=board_cfg["board_side"],  # Side length of square board
            hole_side=board_cfg["hole_side"],   # Side length of square hole, slightly larger than cube for passing
            thickness=board_cfg["thickness"],   # Board thickness
            position=[float(board_base[0]) + x_var, float(board_base[1]) + y_var, float(board_base[2])],  # Board position
            rotation_quat=rot_quat.tolist(),  # z-axis rotation
            name="board_with_hole"
        )
        avoid += [self.board_with_hole]

        ###
        ###
        ###
        ###
        ###
        # First generate target_number (put_in):
        # If put_in_color == 1: Randomly select a color, assign target count in range [put_in_range[0], put_in_range[1]]
        # If put_in_color == 3:
            # First generate total target count total_target from put_in_range
            # Start from [0, 0, 0], randomly distribute to three colors (no requirement for min 1 per color)

        # Then generate spawn_number:
        # If num_colors == 1: Only the color with target will spawn cube, spawn count = max(total_spawn, target count)
        # If num_colors == 3: Spawn count for each color at least equals target, remaining spawn count distributed randomly
        # This ensures spawn >= target for each color.


        # Get configuration for current difficulty
        config = self._sampling["parameters"]["configs"][self.difficulty]
        num_colors = config['color']  # 1 or 3
        spawn_range = config['spawn_cubes']  # [min, max]
        put_in_color_range = config['put_in_color']
        spec_counts = None
        if spec is not None:
            # 规格直接定死每色的生成数与目标数，下面整段配额抽样一次随机数都不走
            order = ["red", "blue", "green"]
            spec_counts = (
                [int(spec["objects"]["spawn_count"].get(name, 0)) for name in order],
                [int(spec["objects"]["target_count"].get(name, 0)) for name in order],
            )
        color_pool = torch.randperm(3, generator=generator).tolist()[:num_colors]
        put_in_color = torch.randint(
            put_in_color_range[0], put_in_color_range[1] + 1, (1,), generator=generator
        ).item()
        put_in_color = max(1, min(3, put_in_color))
        put_in_color = min(put_in_color, max(1, num_colors))
        if spec is None:
            color_pool = self._spec.value("objects.color_pool", color_pool)
            put_in_color = self._spec.value("objects.put_in_color", put_in_color)
        active_color_indices = color_pool[:put_in_color]
        put_in_range = config['put_in_numbers']  # [min, max]

        # First generate target_number (put_in)
        target_numbers = [0, 0, 0]
        if spec_counts is not None:
            spawn_numbers, target_numbers = list(spec_counts[0]), list(spec_counts[1])
        elif put_in_color == 1:
            # Only one color needs to be put in bin
            selected_idx = active_color_indices[0]
            target_numbers[selected_idx] = torch.randint(put_in_range[0], put_in_range[1] + 1, (1,), generator=generator).item()
        else:
            # All 3 colors need to be put in bin, generate total number first then distribute
            total_target = torch.randint(put_in_range[0], put_in_range[1] + 1, (1,), generator=generator).item()
            # Randomly distribute target number to three colors
            for _ in range(total_target):
                idx = torch.randint(0, len(active_color_indices), (1,), generator=generator).item()
                target_numbers[active_color_indices[idx]] += 1

        self.red_cubes_target_number = target_numbers[0]
        self.blue_cubes_target_number = target_numbers[1]
        self.green_cubes_target_number = target_numbers[2]

        # Then generate spawn_number, ensure spawn >= target
        if spec_counts is not None:
            total_spawn = sum(spawn_numbers)
        else:
            total_spawn = torch.randint(spawn_range[0], spawn_range[1] + 1, (1,), generator=generator).item()

        if spec_counts is not None:
            pass  # spawn_numbers 已由规格定死
        elif num_colors == 1:
            # Only one color has cube, choose the one with target (if none, use first color in color_pool)
            spawn_numbers = [0, 0, 0]
            active_idx = next((i for i in color_pool if target_numbers[i] > 0), color_pool[0])
            # Spawn number at least equals target number
            spawn_numbers[active_idx] = max(total_spawn, target_numbers[active_idx])
        else:
            # num_colors controls 1/2/3 colors: ensure each selected color has at least 1 spawn, and spawn >= target
            spawn_numbers = [0, 0, 0]
            for i in color_pool:
                spawn_numbers[i] = max(target_numbers[i], 1)
            used_spawn = sum(spawn_numbers[i] for i in color_pool)
            remaining = total_spawn - used_spawn
            # Randomly distribute remaining spawn count
            for _ in range(max(0, remaining)):
                idx = torch.randint(0, len(color_pool), (1,), generator=generator).item()
                spawn_numbers[color_pool[idx]] += 1

        if spec is None:
            # decision_key 只在新值（xhard）回注出现不等时写进 mismatch 供归因，原值模式忽略、行为不变
            spawn_numbers = self._spec.value("objects.spawn_numbers", spawn_numbers,
                                             decision_key=f"configs.{self.difficulty}.spawn_cubes")
            target_numbers = self._spec.value("objects.target_numbers", target_numbers,
                                              decision_key=f"configs.{self.difficulty}.put_in_numbers")
        self.red_cubes_spawn_number = spawn_numbers[0]
        self.blue_cubes_spawn_number = spawn_numbers[1]
        self.green_cubes_spawn_number = spawn_numbers[2]

        logger.debug(f"Target numbers - Red: {self.red_cubes_target_number}, Blue: {self.blue_cubes_target_number}, Green: {self.green_cubes_target_number}")
        logger.debug(f"Spawn numbers - Red: {self.red_cubes_spawn_number}, Blue: {self.blue_cubes_spawn_number}, Green: {self.green_cubes_spawn_number}")

        ###
        ###
        ###
        ###
        ###
        self.all_cubes = []
        self.red_cubes, self.blue_cubes, self.green_cubes = [], [], []

        color_info = [
            {"color": (1, 0, 0, 1), "name": "red", "list": self.red_cubes, "spawn_num": self.red_cubes_spawn_number},
            {"color": (0, 0, 1, 1), "name": "blue", "list": self.blue_cubes, "spawn_num": self.blue_cubes_spawn_number},
            {"color": (0, 1, 0, 1), "name": "green", "list": self.green_cubes, "spawn_num": self.green_cubes_spawn_number}
        ]

        # Generate task list for all cubes and shuffle order
        cube_tasks = []
        for info in color_info:
            for idx in range(info["spawn_num"]):
                cube_tasks.append({"color": info["color"], "name": info["name"], "list": info["list"], "idx": idx})

        if spec is None:
            # Shuffle generation order
            shuffle_order = self._spec.value(
                "objects.spawn_order", torch.randperm(len(cube_tasks), generator=generator).tolist()
            )
            cube_tasks = [cube_tasks[i] for i in shuffle_order]
        else:
            # 规格的 cubes 列表本身就是生成顺序，直接按它重排并挂上固定位姿；
            # 名字仍由 name_prefix=f"cube_{名}_{序号}" 拼出，与规格里的 object_id 一致
            by_identity = {(item["name"], item["idx"]): item for item in cube_tasks}
            ordered = []
            for entry in spec["layout"]["cubes"]:
                key = (entry["color"], int(entry["color_index"]))
                if key not in by_identity:
                    raise ValueError(f"规格里的 {entry['object_id']} 不在按 spawn_count 展开的任务列表里")
                task = dict(by_identity[key])
                task["fixed_xy"] = [float(entry["xy"][0]), float(entry["xy"][1])]
                task["fixed_yaw"] = float(entry["yaw_rad"])
                ordered.append(task)
            if len(ordered) != len(cube_tasks):
                raise ValueError(
                    f"规格给了 {len(ordered)} 块方块，spawn_count 展开是 {len(cube_tasks)} 块"
                )
            cube_tasks = ordered

        # Spawn cubes in shuffled order
        cubes_cfg = self._sampling["positions"]["cubes"]
        # V4 xhard：min_gap 从调用点字面量外提到 positions.cubes.min_gap_value（值同为 0.02，B1 不动）；
        # 原三档仍用调用点原来的 self.cube_half_size，逐字不变。
        min_gap = float(cubes_cfg["min_gap_value"]) if is_newvalue_difficulty(self.difficulty) else self.cube_half_size
        if spec is None and is_newvalue_difficulty(self.difficulty):
            # V5 xhard（计划 2.12）：槽位 → 配色 → 建 actor 三段，已放方块以精确 OBB 作障碍。
            # 原三档与旧注入通道（spec 非 None）走下面 else 里的原循环，逐字不变。
            self._spawn_cubes_xhard(cube_tasks, avoid, cubes_cfg, min_gap, generator)
        else:
            for task in cube_tasks:
                try:
                    cube = spawn_random_cube(
                        self, color=task["color"], avoid=avoid,
                        include_existing=cubes_cfg["include_existing"], include_goal=cubes_cfg["include_goal"],
                        region_center=list(cubes_cfg["region_center"]), region_half_size=list(cubes_cfg["region_half_size"]),
                        half_size=self.cube_half_size, min_gap=min_gap,
                        random_yaw=cubes_cfg["random_yaw"], name_prefix=f"cube_{task['name']}_{task['idx']}",
                        generator=generator,
                        fixed_xy=task.get("fixed_xy"), fixed_yaw=task.get("fixed_yaw"),
                        recorder=None if spec is not None else self._spec,
                        spec_path=None if spec is not None else f"layout.cubes.{task['name']}_{task['idx']}",
                    )
                    self.all_cubes.append(cube)
                    task["list"].append(cube)
                    avoid.append(cube)
                except RuntimeError as e:
                    logger.debug(f"Failed to spawn {task['name']} cube {task['idx']}: {e}")

        logger.debug(f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})")

        if is_newvalue_difficulty(self.difficulty):
            # V4 D1（只在 xhard 修，H2）：原三档上面的 except RuntimeError 只记日志、不补生成，
            # 实际块数可能静默少于请求数（2.2④）。xhard 记「请求数 vs 实际数」，不等直接判本局失败。
            requested, actual = len(cube_tasks), len(self.all_cubes)
            self._spec.record("objects.spawn_requested", requested)
            self._spec.record("objects.spawn_actual", actual)
            if actual != requested:
                raise SceneGenerationError(
                    f"BinFill xhard: 方块只放下 {actual}/{requested} 块"
                    f"（red {len(self.red_cubes)}/{self.red_cubes_spawn_number}，"
                    f"blue {len(self.blue_cubes)}/{self.blue_cubes_spawn_number}，"
                    f"green {len(self.green_cubes)}/{self.green_cubes_spawn_number}）"
                )

        if spec is not None:
            # 只读证据：记录「创建输入 vs 创建后 actor 实际位姿」，供 INJECTION_BINDING 核对。
            # 这里只读位姿，不改任何状态、不抽随机数。
            self._injection_evidence = {
                "spec_sha256": spec.get("spec_sha256"),
                "episode": spec.get("episode"),
                "button_xy": [float(v) for v in spec["layout"]["button_xy"]],
                "board": dict(spec["layout"]["board"]),
                "spawn_count": dict(spec["objects"]["spawn_count"]),
                "target_count": dict(spec["objects"]["target_count"]),
                "cubes": [
                    {
                        "object_id": entry["object_id"],
                        "requested_xy": [float(v) for v in entry["xy"]],
                        "requested_yaw_rad": float(entry["yaw_rad"]),
                        "actual_p": _actor_xyz(cube),
                        "actual_q": _actor_quat(cube),
                    }
                    for entry, cube in zip(spec["layout"]["cubes"], self.all_cubes)
                ],
            }




    def _spawn_cubes_xhard(self, cube_tasks, avoid, cubes_cfg, min_gap, generator):
        """V5 xhard（计划 2.12，L40～L42；N17、N18）：先放槽位只做几何 → 按 V4 语义配色 → 建 actor。

        * **槽位**：逐个槽位做与 ``spawn_random_cube`` 拒绝循环逐次相同的抽样（每 trial 依次抽
          u1、u2、yaw 三个数，候选方块按 ``min_gap`` 外扩后与障碍做 OBB 相交判定），但**不建 actor**；
          已放槽位以 ``cube_obb2d_exact`` 预制三元组进障碍（修 2.0① 的退化），名义 2 cm 间距真正生效。
          被接受的位姿经 ``layout.slots.<k>`` 注入；回放时冻结值按同一规则复核（N17）。
        * **配色**：槽位 k 先放 ``cube_tasks[k]``（即 V4 的 ``spawn_order`` 语义）；最大同色连通团
          （中心距 ≤ ``link_m``）超过 ``max_component`` 时，**在末尾追加**一次 ``randperm(n)`` 重排，
          最多 ``max_redraws`` 次，失败取团最小的一次（并列取最早）。重排只换颜色标签、不动位置，
          只在被接受的那次调用 ``recorder.value("objects.slot_assignment")``，次数用 ``record()`` 留痕（N18）。
        * **建 actor**：按槽位顺序用 ``spawn_random_cube(fixed_xy, fixed_yaw)`` 建方块（不抽随机数），
          各色列表的顺序即槽位顺序（与 V4「列表顺序＝生成顺序」同义）；``layout.cubes.*`` 改由 ``record()`` 写入。

        槽位放不下直接抛 ``SceneGenerationError``（与 V4 D1「缺块即本局失败」同义）。
        """
        half = float(self.cube_half_size)
        max_trials = 256  # 与 V4 调用点一致（spawn_random_cube 默认值，V4 未显式传）
        # 障碍：按钮 OBB（预制三元组）+ 孔板四条边（与 spawn_random_cube 对 board_with_hole 的特判同式）
        obstacles = []
        for item in avoid:
            if isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], np.ndarray):
                obstacles.append(item)
            elif hasattr(item, "_board_side") and hasattr(item, "_hole_side"):
                obstacles.extend(_board_strips_obb2d(item))
            else:
                raise TypeError(f"BinFill xhard: 不认识的障碍 {item!r}（只应有按钮 OBB 与孔板）")
        # 采样区域：与 spawn_random_cube 同式（区域按方块半边长内缩）
        center = np.array(cubes_cfg["region_center"], dtype=np.float64)
        area_half = np.array(cubes_cfg["region_half_size"], dtype=np.float64)
        x_low, x_high = center[0] - area_half[0] + half, center[0] + area_half[0] - half
        y_low, y_high = center[1] - area_half[1] + half, center[1] + area_half[1] - half
        random_yaw = bool(cubes_cfg["random_yaw"])

        # ── 第一段：槽位（只做几何）──
        slots = []
        for k in range(len(cube_tasks)):
            placed = None
            for _trial in range(max_trials):
                u1 = torch.rand(1, generator=generator).item()
                u2 = torch.rand(1, generator=generator).item()
                x = float(x_low + u1 * (x_high - x_low))
                y = float(y_low + u2 * (y_high - y_low))
                if random_yaw:
                    yaw = float(torch.rand(1, generator=generator).item() * 2 * np.pi)
                else:
                    yaw = 0.0
                c_new, A_new, h_new = _build_new_cube_obb2d(x, y, half, yaw, pad_xy=float(min_gap))
                if any(_obb2d_intersect(c, A, h, c_new, A_new, h_new) for (c, A, h) in obstacles):
                    continue
                placed = [x, y, yaw]
                break
            if placed is None:
                raise SceneGenerationError(
                    f"BinFill xhard: 第 {k} 个槽位在 {max_trials} 次尝试内放不下（已放 {k}/{len(cube_tasks)}）"
                )
            x, y, yaw = (float(v) for v in self._spec.value(f"layout.slots.{k}", placed))
            # N17：回放时冻结值不经拒绝循环，按同一规则复核（导出模式下恒通过）
            if not (x_low <= x <= x_high and y_low <= y <= y_high):
                raise EpisodeSpecError(f"BinFill xhard: 槽位 {k} 的中心 ({x}, {y}) 超出方块区域")
            c_new, A_new, h_new = _build_new_cube_obb2d(x, y, half, yaw, pad_xy=float(min_gap))
            if any(_obb2d_intersect(c, A, h, c_new, A_new, h_new) for (c, A, h) in obstacles):
                raise EpisodeSpecError(
                    f"BinFill xhard: 槽位 {k} 的冻结位姿 ({x}, {y}, {yaw}) 与按钮／孔板／已放方块的间距小于 {min_gap}"
                )
            obstacles.append(cube_obb2d_exact((x, y, yaw), half))
            slots.append((x, y, yaw))

        # ── 第二段：配色（位置不动，只换颜色标签）──
        mix = self._sampling["parameters"]["configs"][self.difficulty]["color_mix"]
        max_component = int(mix["max_component"])
        link = float(mix["link_m"])
        max_redraws = int(mix["max_redraws"])
        n = len(cube_tasks)
        xy = np.array([[s[0], s[1]] for s in slots], dtype=np.float64).reshape(-1, 2)
        names = [task["name"] for task in cube_tasks]

        def _component(order):
            return _max_same_color_component(xy, [names[i] for i in order], link)

        best_order = list(range(n))  # V4 语义：槽位 k 放 spawn_order 洗牌后的第 k 个任务
        best_component = _component(best_order)
        redraws = 0
        while best_component > max_component and redraws < max_redraws:
            perm = torch.randperm(n, generator=generator).tolist()
            redraws += 1
            component = _component(perm)
            if component < best_component:
                best_order, best_component = perm, component
        fallback = best_component > max_component
        self._spec.record("objects.color_redraws", redraws)
        self._spec.record("objects.color_mix_fallback", bool(fallback))
        self._spec.record("objects.color_mix_max_component", int(best_component))
        by_identity = {f"{task['name']}_{task['idx']}": task for task in cube_tasks}
        assignment = [f"{cube_tasks[i]['name']}_{cube_tasks[i]['idx']}" for i in best_order]
        assignment = list(self._spec.value("objects.slot_assignment", assignment))
        # N17：冻结的配色必须恰是本局方块身份的一个排列，且成团不超过上限（兜底局不劣于本次实算的最优）
        if sorted(assignment) != sorted(by_identity):
            raise EpisodeSpecError(
                f"BinFill xhard: 冻结的 slot_assignment {assignment} 与本局方块身份 {sorted(by_identity)} 不是同一组"
            )
        frozen_component = _max_same_color_component(
            xy, [by_identity[ident]["name"] for ident in assignment], link
        )
        if frozen_component > max(max_component, best_component):
            raise EpisodeSpecError(
                f"BinFill xhard: 冻结的 slot_assignment 最大同色团 {frozen_component} 块，超过上限 {max_component}"
            )

        # ── 第三段：按槽位顺序建 actor（固定位姿，不抽随机数）──
        for k, ident in enumerate(assignment):
            task = by_identity[ident]
            x, y, yaw = slots[k]
            cube = spawn_random_cube(
                self, color=task["color"], avoid=None,
                include_existing=False, include_goal=False,
                region_center=list(cubes_cfg["region_center"]), region_half_size=list(cubes_cfg["region_half_size"]),
                half_size=self.cube_half_size, min_gap=min_gap,
                random_yaw=cubes_cfg["random_yaw"], name_prefix=f"cube_{task['name']}_{task['idx']}",
                generator=None, fixed_xy=[x, y], fixed_yaw=yaw,
            )
            self._spec.record(f"layout.cubes.{ident}", [x, y, yaw])
            self.all_cubes.append(cube)
            task["list"].append(cube)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 每次初始化各自记一份规格：颜色排列、恢复动作索引等都按序号分开存，
        # 绝不复用上一次的结果（方案 8.2 对 BinFill 两次初始化的明确要求）。
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)

            tasks=[]
            self.red_cubes_in_bin=0
            self.blue_cubes_in_bin=0
            self.green_cubes_in_bin=0
            self.binfill_language_sequence = []
            color_task_definitions = [
                ("blue", self.blue_cubes, self.blue_cubes_target_number),
                ("red", self.red_cubes, self.red_cubes_target_number),
                ("green", self.green_cubes, self.green_cubes_target_number),
            ]
            if self._episode_spec is None:
                color_order = self._spec.value(
                    f"initializations.{self._native_init_index}.color_order",
                    torch.randperm(len(color_task_definitions), generator=self.generator).tolist(),
                )
            else:
                # 规格的 initialize_color_order 直接定死定义表 (blue, red, green) 的遍历顺序，
                # 替代源码这次 randperm。两次初始化都从同一份规格重建，不复用上一次的结果。
                names = [item[0] for item in color_task_definitions]
                color_order = [names.index(name) for name in self._episode_spec["objects"]["initialize_color_order"]]
            for color_idx in color_order:
                color_name, cube_collection, target_number = color_task_definitions[color_idx]
                if target_number <= 0:
                    continue
                if is_newvalue_difficulty(self.difficulty) and len(cube_collection) < target_number:
                    # V4 D1：原三档这里缺块会在 cube_collection[i] 处 IndexError；xhard 改为明确的场景失败
                    raise SceneGenerationError(
                        f"BinFill xhard: {color_name} 只有 {len(cube_collection)} 块，目标要投 {target_number} 块"
                    )
                self.binfill_language_sequence.append((color_name, target_number))
                for i in range(target_number):
                    cube = cube_collection[i]
                    tasks.append({
                        "func": lambda c=self.all_cubes: is_any_obj_pickup_flag_currentpickup(self,objects=c),
                        "name": subgoal_language.get_subgoal_with_index(i, "pick up the {idx} {color} cube", color=color_name),
                        "subgoal_segment": subgoal_language.get_subgoal_with_index(i, "pick up the {idx} {color} cube at <>", color=color_name),
                        "choice_label": "pick up the cube",
                        "demonstration": False,
                        "failure_func":  lambda:is_button_pressed(self, obj=self.button),
                        "solve": lambda env, planner, c=cube: solve_pickup(env, planner, obj=c),
                        "segment":[cube_collection[i]]
                    })
                    tasks.append({
                        "func": lambda c=self.all_cubes: is_any_obj_dropped_onto_delete(self, objects=c, target=self.board_with_hole),
                        "name": f"put it into the bin",
                        "subgoal_segment":"put it into the bin at <>",
                        "choice_label": "put it into the bin",
                        "demonstration": False,
                        "failure_func":  lambda:is_button_pressed(self, obj=self.button),
                        "solve": lambda env, planner, c=cube: [
                            solve_putonto_whenhold_binspecial(env, planner, target=self.board_with_hole),
                        ],
                        "segment":[self.board_with_hole]
                    })
            tasks.append({
                "func": lambda: is_button_pressed(self, obj=self.button),
                "name": "press the button",
                "subgoal_segment":"press the button at <>",
                "choice_label": "press the button",
                "demonstration": False,
                "failure_func":lambda  c=self.all_cubes:[not check_in_bin_number(self,in_bin_list= [self.red_cubes_in_bin, self.blue_cubes_in_bin, self.green_cubes_in_bin],
                                                            total_number_list=[self.red_cubes_target_number, self.blue_cubes_target_number, self.green_cubes_target_number])
                ,is_any_obj_dropped_onto_delete(self, objects=c, target=self.board_with_hole)],
                "solve": lambda env, planner: [solve_button(env, planner, obj=self.button)],
                  "segment":self.cap_link 
            })
            self.task_list=tasks
            # Record pickup related task indices and items for recovery
            self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
            if self.robomme_failure_recovery:
                # Only inject an intentional failed grasp when recovery mode is enabled
                # 恢复动作的选择是一次真实抽样：原位置照常抽，回注模式下用冻结的索引
                self.fail_grasp_task_index = self._spec.value(
                    f"initializations.{self._native_init_index}.recovery_action_index",
                    inject_fail_grasp(
                    self.task_list,
                    generator=self.generator,
                    mode=self.robomme_failure_recovery_mode,
                ),
                )
            else:
                self.fail_grasp_task_index = None


    def _get_obs_extra(self, info: Dict):
        return dict()



    def evaluate(self,solve_complete_eval=False):
        self.successflag=torch.tensor([False])
        # Save current_task_failure state before calling sequential_task_check
        # This is because failure might be detected during step(), but sequential_task_check might reset it
        previous_failure = getattr(self, "current_task_failure", False)
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

        # If task failed, mark as failed immediately
        # Or if failure was detected previously (previous_failure), also mark as failed
        if task_failed or previous_failure:
            self.failureflag = torch.tensor([True])
            if task_failed:
                logger.debug(f"Task failed: {current_task_name}")
            elif previous_failure:
                # If marked failed due to previous_failure, ensure current_task_failure is also set
                self.current_task_failure = True

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
        self.vis_obj_id_list=[]
        
        timestep = self.elapsed_steps
        if self.dynamic:
            # Dynamically lift cubes for each color (starting from 2nd cube)
            for cube_list in [self.red_cubes, self.blue_cubes, self.green_cubes]:
                for idx in range(1, len(cube_list)):
                    lift_and_drop_objects_back_to_original(
                        self,
                        obj=cube_list[idx],
                        start_step=0,
                        end_step=idx * 100,
                        cur_step=timestep,
                    )
                
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
