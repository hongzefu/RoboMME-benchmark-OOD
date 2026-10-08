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

from .utils.SceneGenerationError import SceneGenerationError
from .utils import *
# V5 L3（仿 VideoPlaceOrder 的 K2 修法）：上一行的 `from .utils import *` 会把同名子模块
# `utils.SceneGenerationError` 盖到名字 `SceneGenerationError` 上（import 自省核实），原三档的
# raise / except 因此是 TypeError（按 H2 原三档保持现状）。xhard 用下面这个别名拿到真正的异常类。
from .utils.SceneGenerationError import SceneGenerationError as _RealSceneGenerationError
from .utils.subgoal_evaluate_func import static_check, is_static
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
from .utils import reset_panda
from .utils.difficulty import NEWVALUE_DIFFICULTIES, is_newvalue_difficulty, normalize_robomme_difficulty
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
# V5（计划 2.15，L48/L49/L54）：xhard 搭档 reset 规划用的扫掠判据与静止障碍（S2b 新增 API，只在 xhard 路径调用）
from .utils.bin_collision import (
    ObjectState,
    button_base_state,
    check_swap_sweep_prefiltered,
    cube_actor_pose,
    cube_shape_specs,
)
# V5 N17：回放冻结规格时复核规划搭档，违反抛 EpisodeSpecError（下划线别名，不经 from .utils import * 外泄）
from .utils.episode_spec import EpisodeSpecError as _EpisodeSpecError

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
# 注意 __init__ 里的 np.random.seed(seed) 是全仓唯一的进程级全局播种点，位置不得移动。
NATIVE_SAMPLING = {
    "parameters": {
        "num_repeats": {
            "sampler": "torch.randint",
            "low": 1,
            "high_exclusive": 4,
            "shape": [1],
        },
        "hard_spawn_rounds": 5,
        "object_selection": {
            "easy_medium_target_count": 1,
            "hard_target_low": 0,
            "swap_remaining_count": 2,
        },
        "swap_selection": {
            "initiator_mapping": "target_then_permuted_remaining_spawned_indices",
            "remaining_selection": "randperm_without_target",
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
        "button": {
            "center_xy": [-0.2, 0],
            "randomize": True,
            "randomize_range": [0.1, 0.1],
            "sampling_expression": "(torch.rand(2, generator=generator) - 0.5) * randomize_range",
            "scale": 1.5,
            "randomize_range_origin": "VideoRepick 调用点显式传入",
        },
        "easy_medium_cubes": {
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
            "random_yaw": True,
            "yaw_range_rad": [0, 6.283185307179586],
            "yaw_expression": "yaw_sample * 2 * np.pi",
            "rotation_center": [0, 0],
            "min_gap": "self.cube_half_size",
            "min_gap_value": 0.02,
            "include_existing": True,
            "include_goal": True,
            "include_flags_origin": "原值取自 spawn_random_cube 形参默认 True；原调用点未传，现由本快照显式传入",
            "region4_reachable": False,
        },
        "hard_cubes": {
            "region_center": [-0.1, 0],
            "region_half_size": [0.2, 0.25],
            "random_yaw": True,
            "yaw_range_rad": [0, 6.283185307179586],
            "yaw_expression": "yaw_sample * 2 * np.pi",
            "min_gap": "self.cube_half_size",
            "min_gap_value": 0.02,
            "include_existing": False,
            "include_goal": False,
        },
    },
}


from .utils.xhard import HSV_FLOOR_COLOR, cube_obb2d_exact, hsv_floor_rgb
from .utils import swap_uniform

# V4 xhard「block 颜色任意」（C2：每局全部方块仍同色，只是色值任意）。
# 色域按用户 2026-09-22 决定设饱和度/亮度下限：色相任意、S≥0.5、V≥0.4（utils/xhard.py::HSV_FLOOR_COLOR），
# alpha 固定 1；可经 sampling_config 的 decision.xhard.block_color 覆盖三个区间，不改源码。
NEWVALUE_BLOCK_COLOR = {
    "policy": "same_color_hsv_floor",
    "sampler": "torch.rand",
    **copy.deepcopy(HSV_FLOOR_COLOR),
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


def _solve_hold_obj_xhard(env, planner, static_steps):
    """V4 xhard 专用的原地等待：与 ``solve_hold_obj(close=False)`` 同语义，只把裸 ``except`` 收窄为 ``AttributeError``。

    原函数 ``utils/subgoal_planner_func.py::solve_hold_obj`` 用裸 ``except:`` 包住 ``planner.open_gripper()``，
    会吞掉 step 里抛出的 ``BinCollisionError``；此时 ``elapsed_steps`` 不前进，循环永不结束，且每轮都往
    ``_runtime_checks`` 追加一条拒绝证据，最终内存暴涨崩溃（本机实测 rc=139）。共享工具函数的既有缺陷
    按 N12 不就地修，这里另写 xhard 专用路径。
    """
    start_step = int(getattr(env, "elapsed_steps", 0))
    target_step = start_step + static_steps
    while int(getattr(env, "elapsed_steps", 0)) < target_step:
        try:
            planner.open_gripper()
        except AttributeError:
            pass
    return None


def _solve_hold_until_step_xhard(env, planner, target_step):
    """V6 审查修复 N11：xhard 交换段的原地等待改为等到**绝对步** ``target_step``（与 ``_solve_hold_obj_xhard`` 同语义，
    只把「相对当前步数 static_steps」换成绝对步）。

    根因（审查 xhard4 ep3 seed 6900300，12 次交换只有 11 个 static 边界）：交换段 static 任务的内部完成判定
    ``static_check(static_steps=50)`` 从**内部**任务切换那一步起计数，而演示 wrapper 执行的等待解法从**录像侧**任务切换
    （上一段解法结束）起再等 50 步并多消耗约 4 步开销；录像侧只在每段解法结束时采纳内部任务索引，两条时钟每段漂 4 步，
    累计超过一段长度后某个内部 static 任务在两次采纳之间整段完成，边界被吞掉。
    修法：新四档交换段的完成判定与等待解法都钉到同一条绝对时钟——第 k 段结束步 ``swap_schedule[k][3]``，漂移不再累计。
    """
    target_step = int(target_step)
    while int(getattr(env, "elapsed_steps", 0)) < target_step:
        try:
            planner.open_gripper()
        except AttributeError:
            pass
    return None


def _cube_index_of(name):
    """把规格里的 ``bin_<i>`` 还原成生成序号 ``i``（VideoRepick 的方块沿用 bin_ 前缀）。"""
    return int(str(name).rsplit("_", 1)[1])


# ── V5 xhard：搭档 reset 规划（计划 2.15，L47～L49、L54）──────────────────────────────────
# 下面三个纯函数/类只在 xhard 的 _plan_swaps_xhard 里调用，原三档不经过。


def _xhard_slot_states(slots, cube_half, margin):
    """名义槽位 ``[(x, y, yaw), ...]`` → 扫掠判据用的方块状态。

    形状用 ``cube_shape_specs(cube_half + margin)``（规划期余量，L49 的 5 mm），位姿复刻
    ``spawn_random_cube`` 的落点（中心高度仍取真实半边 ``cube_half``，与 P4 原型一致）。
    """
    shapes = cube_shape_specs(float(cube_half) + float(margin))
    states = []
    for index, (x, y, yaw) in enumerate(slots):
        p, q = cube_actor_pose((x, y), yaw, cube_half)
        states.append(ObjectState(name=f"slot_{index}", p=p, q=q, shapes=shapes))
    return states


class _XhardSlotSweepFeasibility:
    """名义槽位对的交换扫掠可行性，按无序槽位对缓存。

    ``swap_flat_two_lane`` 结束时两块互换 xy 与朝向，所以槽位集合整局不变、只有占用者在槽位间置换；
    两块的弯道路径只取决于无序槽位对（谁当发起者路径相同），所以可行性按无序对缓存。
    bystander = 其余槽位（同样按规划余量放大）+ 静止障碍（L54 的按钮底座）。
    判据用 S2b 的 ``check_swap_sweep_prefiltered``（判定与 ``check_swap_sweep`` 相同，只加认证预筛提速）。
    """

    def __init__(self, slot_states, statics=()):
        self.states = list(slot_states)
        self.statics = list(statics)
        self.cache = {}
        self.evidence = {}

    def feasible(self, slot_a, slot_b):
        key = (min(slot_a, slot_b), max(slot_a, slot_b))
        if key not in self.cache:
            bystanders = [state for index, state in enumerate(self.states) if index not in key] + self.statics
            _gap, rejection = check_swap_sweep_prefiltered(
                self.states[key[0]], self.states[key[1]], bystanders, stage="plan"
            )
            self.cache[key] = rejection is None
            self.evidence[key] = None if rejection is None else rejection.as_dict()
        return self.cache[key]


def _plan_swap_partners_xhard(slot_xy, seq, u, feasible, nearest_k, resolve=None):
    """按名义槽位推进，逐次给发起者 ``seq[k]`` 规划搭档（纯函数，不抽随机数）。

    * 候选 = 发起者当前槽位之外的其余槽位，按 XY 距离升序（稳定排序，等距按槽位序号）；
    * 最近 ``nearest_k`` 个里至少一个扫掠可行 ⇒ 候选池 = 按距离序的前 ``nearest_k`` 个可行者；
      否则（回退）⇒ 候选池 = 全部可行者；
    * ``u[k]`` 在候选池里均匀选一个（下标 ``floor(u·len)``）；候选池为空 ⇒ 抛真 ``SceneGenerationError``。
    * ``resolve(k, a, b)``：可选的取值钩子，返回**实际采用**的搭档方块号（回放时是冻结值）；
      实际搭档必须是另一块且其槽位对扫掠可行，否则抛 ``EpisodeSpecError``（N17）。
    * 名义对换后继续下一次（占用关系随之更新）。

    返回逐次记录 ``[{"initiator", "partner", "slot_a", "slot_b", "dist_m", "pool", "fallback"}, ...]``。
    """
    points = np.asarray(slot_xy, dtype=np.float64).reshape(-1, 2)
    count = len(points)
    occupant = list(range(count))  # occupant[槽位] = 方块号
    slot_of = list(range(count))   # slot_of[方块号] = 槽位
    plan = []
    for k, initiator in enumerate(seq):
        slot_a = slot_of[initiator]
        dist = np.linalg.norm(points - points[slot_a], axis=1)
        dist[slot_a] = np.inf
        order = [int(j) for j in np.argsort(dist, kind="stable")[: count - 1]]
        fallback = not any(feasible(slot_a, c) for c in order[:nearest_k])
        if not fallback:
            pool = []
            for c in order:
                if feasible(slot_a, c):
                    pool.append(c)
                    if len(pool) == nearest_k:
                        break
        else:
            pool = [c for c in order[nearest_k:] if feasible(slot_a, c)]
        if not pool:
            raise _RealSceneGenerationError(
                f"xhard: 第 {k} 次交换的发起者 bin_{initiator}（槽位 {slot_a}）没有扫掠可行的搭档"
            )
        slot_b = pool[min(int(float(u[k]) * len(pool)), len(pool) - 1)]
        partner = occupant[slot_b]
        if resolve is not None:
            partner = int(resolve(k, initiator, partner))
            if not 0 <= partner < count or partner == initiator:
                raise _EpisodeSpecError(f"xhard: 第 {k} 次交换的冻结搭档 bin_{partner} 非法（发起者 bin_{initiator}）")
            slot_b = slot_of[partner]
            if not feasible(slot_a, slot_b):
                raise _EpisodeSpecError(
                    f"xhard: 第 {k} 次交换的冻结搭档 bin_{partner} 与发起者 bin_{initiator} 的扫掠在规划口径下不可行"
                )
        plan.append({
            "initiator": initiator, "partner": partner, "slot_a": slot_a, "slot_b": slot_b,
            "dist_m": float(dist[slot_b]), "pool": len(pool), "fallback": fallback,
        })
        occupant[slot_a], occupant[slot_b] = partner, initiator
        slot_of[initiator], slot_of[partner] = slot_b, slot_a
    return plan


def native_blocks(cls, *, release="newtask-v6"):
    """本环境的 ``(decision, native)`` 原值块；外部导出与内部解析共用同一份，杜绝两套真值。"""
    native = copy.deepcopy(NATIVE_SAMPLING)
    if release in ("newtask-v4", "newtask-v5"):
        legacy_configs = {
            "easy": cls.configs["easy"], "medium": cls.configs["medium"], "hard": cls.configs["hard"],
            "xhard": {
                "cube": 6, "swap_min": 8, "swap_max": 12,
                "num_repeats_low": 4, "num_repeats_high_exclusive": 7,
                "layout_mode": "clutter", "region_center": [-0.1, 0.0],
                "region_half_size": [0.2, 0.25], "min_center_dist_m": 0.12,
                "partner_nearest_k": 3, "partner_sweep_margin_m": 0.005,
                "partner_button_obstacle": True,
            },
        }
        return _legacy_decision(legacy_configs, release), native
    # newtask-v7：与 v6 同一解析路径，取当前类常量（即 V7 定值；v6 值只存于包内 v6 规格 header，0928 方案 R3）
    if release not in ("newtask-v6", "newtask-v7"):
        raise ValueError(f"VideoRepick 不支持 sampling_config release {release!r}")
    native["parameters"]["configs"] = copy.deepcopy(cls.configs)
    return _native_decision(cls), native


def _legacy_decision(configs, release):
    decision = {
        "layout_mode": "native_by_difficulty",
        "num_repeats_range": {
            "low": NATIVE_SAMPLING["parameters"]["num_repeats"]["low"],
            "high_exclusive": NATIVE_SAMPLING["parameters"]["num_repeats"]["high_exclusive"],
        },
        "block_color_policy": "native_by_difficulty",
        "swap": {difficulty: {"swap_min": cfg["swap_min"], "swap_max": cfg["swap_max"]}
                 for difficulty, cfg in configs.items()},
    }
    xhard = configs["xhard"]
    decision["num_repeats_range"]["xhard"] = {
        "low": xhard["num_repeats_low"], "high_exclusive": xhard["num_repeats_high_exclusive"],
    }
    decision["xhard"] = {
        "layout": {
            "mode": xhard["layout_mode"], "cube_count": xhard["cube"],
            "region_center": list(xhard["region_center"]),
            "region_half_size": list(xhard["region_half_size"]),
        },
        "block_color": copy.deepcopy(NEWVALUE_BLOCK_COLOR),
    }
    if release == "newtask-v5":
        decision["xhard"]["layout"]["min_center_dist_m"] = xhard["min_center_dist_m"]
        decision["xhard"]["swap_plan"] = {
            "initiator_rule": "target_then_randperm_k_mod_cube_count",
            "partner_rule": "reset_plan_nearest_feasible",
            "nearest_k": xhard["partner_nearest_k"],
            "sweep_margin_m": xhard["partner_sweep_margin_m"],
            "button_obstacle": xhard["partner_button_obstacle"],
        }
    return decision


def _native_decision(cls):
    """按方案第二节字段表切出 decision 块（原值阶段等于原值）。"""
    # 第二节 2.10：decision 为布局模式、重复抓放次数范围、逐块颜色策略与是否交换／交换次数。
    # 原值阶段全部取原规则：布局模式沿用难度自带的锚点／区域，次数与交换次数取自 configs。
    # V6：四个新值档各有独立 decision 子树；原三档可见部分不变。
    decision = {
        "layout_mode": "native_by_difficulty",
        "num_repeats_range": {
            "low": NATIVE_SAMPLING["parameters"]["num_repeats"]["low"],
            "high_exclusive": NATIVE_SAMPLING["parameters"]["num_repeats"]["high_exclusive"],
        },
        "block_color_policy": "native_by_difficulty",
        "swap": {
            difficulty: {"swap_min": cfg["swap_min"], "swap_max": cfg["swap_max"]}
            for difficulty, cfg in cls.configs.items()
        },
    }
    for tier in NEWVALUE_DIFFICULTIES:
        tier_cfg = cls.configs[tier]
        # 原三档仍读 native.parameters.num_repeats；新值档按 tier 读半开重复次数区间。
        decision["num_repeats_range"][tier] = {
            "low": tier_cfg["num_repeats_low"],
            "high_exclusive": tier_cfg["num_repeats_high_exclusive"],
        }
        decision[tier] = {
            "layout": {
                "mode": tier_cfg["layout_mode"],
                "cube_count": tier_cfg["cube"],
                "region_center": list(tier_cfg["region_center"]),
                "region_half_size": list(tier_cfg["region_half_size"]),
            },
            "block_color": copy.deepcopy(NEWVALUE_BLOCK_COLOR),
        }
        # S5 使用全部可行槽位对，5 mm 余量与按钮障碍沿用原机制。
        decision[tier]["layout"]["min_center_dist_m"] = tier_cfg["min_center_dist_m"]
        decision[tier]["swap_plan"] = {
            # V6（计划 2.5，M6(a) S5）：发起者与搭档一起在「全部扫掠可行的槽位对」里按参与次数均衡贪心选
            #（先比两块参与次数之和、再比较大者；平局在局部流上均匀抽、再抽一次定谁当发起者），禁止立即换回，
            # 整条极差 > 1 重排 ≤ 20 次；不再限最近 3 个。可行图有孤立槽位（该块不可能参与）则本局 SceneGenerationError，
            # 与 V5「某发起者没有可行搭档」的失败条件同一口径（reset 成功率不降）。
            "initiator_rule": "s5_balanced_pair",
            "partner_rule": "s5_balanced_greedy",
            "s5": swap_uniform.inner_swap_plan_cfg(require_connected=False, score="sum_max"),
            # 以下两项沿用 V5：规划期方块半边 +5 mm 余量（L49）；按钮底座作为静止 bystander（L54）
            "sweep_margin_m": tier_cfg["partner_sweep_margin_m"],
            "button_obstacle": tier_cfg["partner_button_obstacle"],
        }
    return decision


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
            raise ValueError(f"VideoRepick.parameters.{key} 必须完整保留原版规则与类型")
    return resolved


@register_env("VideoRepick", override=True)
class VideoRepick(BaseEnv):

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
        "cube":3,
        "swap_min":1,
        "swap_max":2,
    }
    config_medium= {
        "cube":3,
        "swap_min":2,
        "swap_max":3,
    }
    config_hard = {
        "cluster":True,
        "swap":None,
        "swap_min":0,
        "swap_max":0,
    }
    # V6 新值档：块数、交换次数与 repick 次数按最终表分档；布局、颜色、中心距和 S5 机制共用。
    config_xhard1 = {
        "cube": 4, "swap_min": 4, "swap_max": 4,  # V7 定值：swap 4/6/8/10、repick 2/3/4/5（0928 方案 §3.2.2）
        "num_repeats_low": 2, "num_repeats_high_exclusive": 3,
        "layout_mode": "clutter",
        "region_center": [-0.1, 0.0],
        "region_half_size": [0.2, 0.25],
        "min_center_dist_m": 0.12,
        "partner_sweep_margin_m": 0.005,
        "partner_button_obstacle": True,
    }
    config_xhard2 = {
        "cube": 5, "swap_min": 6, "swap_max": 6,
        "num_repeats_low": 3, "num_repeats_high_exclusive": 4,
        "layout_mode": "clutter", "region_center": [-0.1, 0.0], "region_half_size": [0.2, 0.25],
        "min_center_dist_m": 0.12, "partner_sweep_margin_m": 0.005, "partner_button_obstacle": True,
    }
    config_xhard3 = {
        "cube": 6, "swap_min": 8, "swap_max": 8,
        "num_repeats_low": 4, "num_repeats_high_exclusive": 5,
        "layout_mode": "clutter", "region_center": [-0.1, 0.0], "region_half_size": [0.2, 0.25],
        "min_center_dist_m": 0.12, "partner_sweep_margin_m": 0.005, "partner_button_obstacle": True,
    }
    config_xhard4 = {
        "cube": 7, "swap_min": 10, "swap_max": 10,
        "num_repeats_low": 5, "num_repeats_high_exclusive": 6,
        "layout_mode": "clutter", "region_center": [-0.1, 0.0], "region_half_size": [0.2, 0.25],
        "min_center_dist_m": 0.12, "partner_sweep_margin_m": 0.005, "partner_button_obstacle": True,
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


    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     sampling_config=None,
                     episode_spec=None,
                     native_episode_spec=None,
                     **kwargs):
        # 必须落在任何随机数调用（含 np.random.seed）与 super().__init__() 之前
        self._sampling = _resolve_sampling_config(type(self), sampling_config)
        self._episode_spec = _resolve_episode_spec(episode_spec, "VideoRepick")
        self._spec = SpecRecorder(native_episode_spec, "VideoRepick", {"seed": seed},
                                  difficulty=kwargs.get("difficulty"))
        # 初始化序号从 -1 起，_initialize_episode 每次进来先加一；
        # _load_scene 里的取值点用不带序号的路径，所以这里只作兜底。
        self._native_init_index = -1
        self._injection_evidence = {}
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

        np.random.seed(seed)
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
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        if is_newvalue_difficulty(self.difficulty) and self._episode_spec is not None:
            # 新值规格含逐档 clutter 布局与 S5 交换计划，链路甲的原三块格式不能复用。
            raise ValueError("VideoRepick 新值档不接受链路甲的 episode_spec；请使用 native_episode_spec")
        repeats_cfg = self._sampling["parameters"]["num_repeats"]
        if self._episode_spec is None and is_newvalue_difficulty(self.difficulty):
            # Repick 次数由当前新值档的半开区间抽取。
            tier_repeats = self._sampling["decision"]["num_repeats_range"][self.difficulty]
            self.num_repeats = self._spec.value(
                "objects.num_repeats",
                torch.randint(tier_repeats["low"], tier_repeats["high_exclusive"], tuple(repeats_cfg["shape"]), generator=self.generator).item(),
                decision_key=f"num_repeats_range.{self.difficulty}",
            )
        elif self._episode_spec is None:
            self.num_repeats = torch.randint(repeats_cfg["low"], repeats_cfg["high_exclusive"], tuple(repeats_cfg["shape"]), generator=self.generator).item()
        else:
            self.num_repeats = int(self._episode_spec["objects"]["num_repeats"])
        logger.debug(f"Task will repeat {self.num_repeats} times (pickup-drop cycles)")

        difficulty_cfg = self._sampling["parameters"]["configs"][self.difficulty]
        if self._episode_spec is None and is_newvalue_difficulty(self.difficulty):
            # 交换次数从当前难度档的 decision 范围抽取。
            tier_swap = self._sampling["decision"]["swap"][self.difficulty]
            self.swap_times = self._spec.value(
                "objects.n_swaps",
                torch.randint(tier_swap["swap_min"], tier_swap["swap_max"] + 1, (1,), generator=self.generator).item(),
                decision_key=f"swap.{self.difficulty}",
            )
        elif self._episode_spec is None:
            self.swap_times = self._spec.value(
                "objects.n_swaps",
                torch.randint(difficulty_cfg['swap_min'], difficulty_cfg['swap_max']+1, (1,), generator=self.generator).item(),
            )
        else:
            self.swap_times = int(self._episode_spec["objects"]["n_swaps"])
        logger.debug(f"Task will swap {self.swap_times} times")


        self.static_flag=False
        self.start_step=99999
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
        try:
            self.table_scene = TableSceneBuilder(
                self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            self.table_scene.build()

            spec = self._episode_spec
            button_cfg = self._sampling["positions"]["button"]
            button_center = (
                tuple(button_cfg["center_xy"]) if spec is None else tuple(spec["layout"]["button_xy"])
            )
            button_obb_1 = build_button(
                self,
                center_xy=button_center,
                scale=button_cfg["scale"],
                generator=self.generator,
                name="button",
                # 规格给的是**最终**中心，关掉 randomize 后 build_button 不抽随机数
                randomize=button_cfg["randomize"] if spec is None else False,
                randomize_range=tuple(button_cfg["randomize_range"])
            )
            # Store first button before building second one
            self.button_left = self.button
            self.button_joint_1 = self.button_joint

            avoid = [button_obb_1]

            options = [
                {"color": (1, 0, 0, 1), "name": "red"},
                {"color": (0, 0, 1, 1), "name": "blue"},
                {"color": (0, 1, 0, 1), "name": "green"},
            ]
            if is_newvalue_difficulty(self.difficulty):
                # 四个新值档共用 clutter + S5 分支；原三档的分支与取值点不变。
                self._load_cubes_newvalue(avoid)
            elif self.difficulty == "hard":
                self.spawned_cubes = []

                hard_cfg = self._sampling["positions"]["hard_cubes"]
                for idx in range(self._sampling["parameters"]["hard_spawn_rounds"]):
                    shuffle_indices = self._spec.value(
                        f"objects.hard_round_order.{idx}",
                        torch.randperm(len(options), generator=self.generator).tolist(),
                    )
                    new_options = [options[i] for i in shuffle_indices]
                    for group in new_options:
                        try:
                            cube = spawn_random_cube(
                                self,
                                color=group["color"],
                                avoid=avoid,
                                include_existing=hard_cfg["include_existing"],
                                include_goal=hard_cfg["include_goal"],
                                region_center=list(hard_cfg["region_center"]),
                                region_half_size=list(hard_cfg["region_half_size"]),
                                half_size=self.cube_half_size,
                                min_gap=self.cube_half_size,
                                random_yaw=hard_cfg["random_yaw"],
                                name_prefix=f"cube_{group['name']}_{idx}",
                                generator=self.generator,
                            )
                        except RuntimeError as e:
                            raise SceneGenerationError(
                                f"Failed to generate {group['name']} cube {idx}"
                            ) from e

                        self.spawned_cubes.append(cube)
                        avoid.append(cube)

                if not self.spawned_cubes:
                    raise SceneGenerationError("Failed to generate any cube")

                selection_cfg = self._sampling["parameters"]["object_selection"]
                target_idx = self._spec.value(
                    "objects.target",
                    torch.randint(selection_cfg["hard_target_low"], len(self.spawned_cubes), (1,), generator=self.generator).item(),
                )
                logger.debug("target index: %s", target_idx)
                self.target_cube_1 = self.spawned_cubes[target_idx]

            else:
                if spec is None:
                    idx = self._spec.value(
                        "objects.color_idx",
                        torch.randint(0, len(options), (1,), generator=self.generator).item(),
                    )
                else:
                    # 三块同色，颜色由规格定死（原定义表顺序是 red / blue / green）
                    idx = [item["name"] for item in options].index(spec["objects"]["color"])
                chosen_color = options[idx]["color"]

                cube_colors = [chosen_color] * 4
                if spec is None:
                    shuffle_indices = self._spec.value(
                        "objects.color_order",
                        torch.randperm(len(cube_colors), generator=self.generator).tolist(),
                    )
                    cube_colors = [cube_colors[i] for i in shuffle_indices]
                # 四个元素全是同一个颜色，打乱与否结果相同；注入路径省掉这次抽样

                self.spawned_cubes = []

                plain_cfg = self._sampling["positions"]["easy_medium_cubes"]
                difficulty_cfg = self._sampling["parameters"]["configs"][self.difficulty]
                region4 = [list(point) for point in plain_cfg["region4"]]
                region3_tri = [list(point) for point in plain_cfg["region3_tri"]]
                region3_line = [list(point) for point in plain_cfg["region3_line"]]

                if spec is None:
                    choice_cfg = plain_cfg["region3_choice"]
                    region3_choice = torch.randint(choice_cfg["low"], choice_cfg["high_exclusive"], (1,), generator=self.generator).item()
                    region3 = region3_tri if region3_choice == 0 else region3_line

                    if difficulty_cfg['cube'] == 4:
                        region = region4
                    else:
                        region = region3
                    angle, region = rotate_points_random(region, tuple(plain_cfg["layout_rotation_range_rad"]), self.generator)
                else:
                    # 规格里的 cubes[i].xy 就是最终位置（锚点旋转 + 偏移在冻结前已算好并过了
                    # 碰撞检查），注入路径不再走 rotate_points_random，也就不抽随机数
                    angle = float(spec["layout"]["theta_rad"])
                    region = None

                for i in range(difficulty_cfg['cube']):
                    fixed_xy = fixed_yaw = None
                    if spec is not None:
                        entry = spec["layout"]["cubes"][i]
                        if entry["object_id"] != f"bin_{i}":
                            raise ValueError(f"规格第 {i} 块方块的 object_id 是 {entry['object_id']}，应为 bin_{i}")
                        fixed_xy = [float(entry["xy"][0]), float(entry["xy"][1])]
                        fixed_yaw = float(entry["yaw_rad"])
                    try:
                        cube_actor = spawn_random_cube(
                            self,
                            avoid=avoid,
                            region_center=region[i] if region is not None else plain_cfg["region3_tri"][0],
                            region_half_size=plain_cfg["region_half_size"],
                            min_gap=self.cube_half_size * 1,
                            half_size=self.cube_half_size,
                            name_prefix=f"bin_{i}",
                            max_trials=256,
                            color=cube_colors[i],
                            random_yaw=plain_cfg["random_yaw"],
                            include_existing=plain_cfg["include_existing"],
                            include_goal=plain_cfg["include_goal"],
                            generator=self.generator,
                            fixed_xy=fixed_xy,
                            fixed_yaw=fixed_yaw,

                        )
                    except RuntimeError as e:
                        raise SceneGenerationError(f"Failed to generate bin_{i}") from e

                    self.spawned_cubes.append(cube_actor)
                    setattr(self, f"bin_{i}", cube_actor)
                    avoid.append(cube_actor)

                if not self.spawned_cubes:
                    raise SceneGenerationError("Failed to generate any bin")

                selection_cfg = self._sampling["parameters"]["object_selection"]
                if spec is None:
                    target_indices = torch.randperm(len(self.spawned_cubes), generator=self.generator)[:selection_cfg["easy_medium_target_count"]].tolist()
                else:
                    target_indices = [_cube_index_of(spec["objects"]["target"])]
                self.target_cube_1 = self.spawned_cubes[target_indices[0]]

                if self.difficulty != "hard":
                    remaining_indices = [i for i in range(len(self.spawned_cubes)) if i not in target_indices]
                    if len(remaining_indices) < 2:
                        raise SceneGenerationError("Not enough cubes for swapping")

                    if spec is None:
                        selected_remaining = self._spec.value(
                            "objects.swap_initiators_remaining",
                            torch.randperm(len(remaining_indices), generator=self.generator)[:selection_cfg["swap_remaining_count"]].tolist(),
                        )
                        selected_indices = [remaining_indices[i] for i in selected_remaining]
                    else:
                        # ⚠ 源码无条件赋值 swap_pair{1,2,3}_idx1，所以规格存的是完整的 3 个发起者：
                        # 第一个必是目标方块，后两个是另外两块的一个排列
                        spec_initiators = [_cube_index_of(name) for name in spec["objects"]["swap_initiators"]]
                        if spec_initiators[0] != target_indices[0]:
                            raise ValueError(
                                f"规格的首个交换发起者 bin_{spec_initiators[0]} 不是目标方块 bin_{target_indices[0]}"
                            )
                        selected_indices = spec_initiators[1:]
                        if sorted(selected_indices) != sorted(remaining_indices):
                            raise ValueError(
                                f"规格的后续发起者 {selected_indices} 不是其余两块 {remaining_indices} 的排列"
                            )
                    swap_indices = target_indices + selected_indices

                    self.swap_pair1_idx1 = self.spawned_cubes[swap_indices[0]]
                    self.swap_pair2_idx1 = self.spawned_cubes[swap_indices[1]]
                    self.swap_pair3_idx1 = self.spawned_cubes[swap_indices[2]]
                    self.swap_pair1_idx2 = None
                    self.swap_pair2_idx2 = None
                    self.swap_pair3_idx2 = None
                    # xhard（swap 4～5 次）：第 k 次发起者循环沿用前 3 个（a,b,c,a,b）；swap ≤ 3 次时本循环不执行
                    for k in range(3, self.swap_times):
                        setattr(self, f"swap_pair{k+1}_idx1", self.spawned_cubes[swap_indices[k % 3]])
                        setattr(self, f"swap_pair{k+1}_idx2", None)
                    self._refresh_swap_schedule()

                if spec is not None:
                    # 只读证据：创建输入 vs 创建后 actor 实际位姿，供 INJECTION_BINDING 核对
                    self._injection_evidence = {
                        "spec_sha256": spec.get("spec_sha256"),
                        "episode": spec.get("episode"),
                        "theta_rad": float(spec["layout"]["theta_rad"]),
                        "layout_type": spec["layout"]["type"],
                        "n_swaps": self.swap_times,
                        "num_repeats": self.num_repeats,
                        "color": spec["objects"]["color"],
                        "target_index": target_indices[0],
                        "button_xy": [float(v) for v in spec["layout"]["button_xy"]],
                        "cubes": [
                            {
                                "object_id": entry["object_id"],
                                "requested_xy": [float(v) for v in entry["xy"]],
                                "requested_yaw_rad": float(entry["yaw_rad"]),
                                "actual_p": [float(v) for v in self._get_actor_position(actor)[:3]],
                            }
                            for entry, actor in zip(spec["layout"]["cubes"], self.spawned_cubes)
                        ],
                    }
        except _scene_gen_error(self.difficulty):  # V5 L3：xhard 用真类，原三档仍是被遮蔽的原名字
            raise
        except Exception as exc:
            if is_newvalue_difficulty(self.difficulty) and isinstance(exc, _EpisodeSpecError):
                # V5 N17：回放冻结规格违反几何规则（最小中心距、规划搭档）是规格／代码类错误，
                # 不包成可重试的 SceneGenerationError；原三档不进这个分支，行为逐字不变
                raise
            raise _scene_gen_error(self.difficulty)(
                f"Failed to load VideoRepick scene for seed {self.seed}"
            ) from exc

    def _load_cubes_newvalue(self, avoid):
        """新值档方块生成：clutter 布局、每局同色任意色值、最小中心距和 S5 reset 规划。

        取值顺序（原三档不经过这里）：颜色 → 逐块位姿 → 目标 → 剩余方块顺序 → 规划种子 →
        reset 规划搭档（不抽随机数）→ 逐次注入 ``actions.swap_pairs.<k>``。
        新值一律从 ``decision.<tier>`` 取布局、颜色和 S5 配置；判据参数沿用 hard 整片区域原值；
        ``min_gap`` 与原三档一样取 ``self.cube_half_size``。
        每个取值点都经 ``self._spec``，「请求块数 vs 实际块数」不等直接判本局失败（计划 2.2④）。

        V5（计划 2.15）：
        * L50：本档全部方块两两中心距 ≥ ``layout.min_center_dist_m``，经 ``spawn_random_cube(min_center_dist=...)``
          在拒绝循环内判（每次 trial 仍是 3 个 rand）；已放方块改用 ``cube_obb2d_exact`` 精确障碍，
          ``include_existing=False``（不再走会退化的 actor 路径）。回放冻结位姿时由 spawn 函数按同一规则复核（N17）。
        * L47 a'：``objects.swap_initiators_remaining`` 保留原取值点并按本档的其余方块数量抽排列；该排列不再决定 S5 发起者。
        * S5：由 ``_plan_swaps_newvalue_v6`` 规划全部扫掠可行槽位对。
        """
        tier_decision = self._sampling["decision"][self.difficulty]
        layout = tier_decision["layout"]
        if layout["mode"] != "clutter":
            raise ValueError(f"VideoRepick {self.difficulty} 只实现了 clutter 布局，收到 {layout['mode']!r}")
        region_cfg = self._sampling["positions"]["hard_cubes"]
        # V5 L50：最小中心距；按钮 OBB 是 _load_scene 放进 avoid 的第一个元素（L54 规划要用按钮最终中心）
        min_center_dist = float(layout["min_center_dist_m"])
        button_obb = avoid[0] if avoid else None

        color_cfg = tier_decision["block_color"]
        u = torch.rand(3, generator=self.generator).tolist()
        rgb = self._spec.value(
            "objects.color_rgb",
            hsv_floor_rgb(u, color_cfg),
            decision_key=f"{self.difficulty}.block_color",
        )
        chosen_color = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)

        requested = int(layout["cube_count"])
        self.spawned_cubes = []
        placed = []  # 已放方块的精确 OBB（同时进 avoid 与最小中心距的参考点集）
        for i in range(requested):
            try:
                cube_actor = spawn_random_cube(
                    self,
                    avoid=avoid,
                    region_center=list(layout["region_center"]),
                    region_half_size=list(layout["region_half_size"]),
                    min_gap=self.cube_half_size,
                    half_size=self.cube_half_size,
                    name_prefix=f"bin_{i}",
                    max_trials=256,
                    color=chosen_color,
                    random_yaw=region_cfg["random_yaw"],
                    include_existing=False,
                    include_goal=region_cfg["include_goal"],
                    generator=self.generator,
                    recorder=self._spec,
                    spec_path=f"layout.cubes.{i}.xy_yaw",
                    min_center_dist=(min_center_dist, placed),
                )
            except RuntimeError as e:
                raise _RealSceneGenerationError(f"{self.difficulty}: failed to generate bin_{i} of {requested}") from e
            self.spawned_cubes.append(cube_actor)
            setattr(self, f"bin_{i}", cube_actor)
            obb = cube_obb2d_exact(cube_actor, self.cube_half_size)
            placed.append(obb)
            avoid.append(obb)
        self._spec.record("objects.cube_count.requested", requested)
        self._spec.record("objects.cube_count.actual", len(self.spawned_cubes))
        if len(self.spawned_cubes) != requested:
            raise _RealSceneGenerationError(
                f"{self.difficulty}: requested {requested} cubes but spawned {len(self.spawned_cubes)}"
            )

        target_index = self._spec.value(
            "objects.target",
            int(torch.randint(0, len(self.spawned_cubes), (1,), generator=self.generator).item()),
        )
        self.target_cube_1 = self.spawned_cubes[target_index]
        remaining_indices = [i for i in range(len(self.spawned_cubes)) if i != target_index]
        # V5 L47 a'：其余 5 块的完整发起顺序（V4 同一次 randperm，只是不再截 [:2]）
        selected_remaining = self._spec.value(
            "objects.swap_initiators_remaining",
            torch.randperm(len(remaining_indices), generator=self.generator).tolist(),
        )
        if sorted(int(i) for i in selected_remaining) != list(range(len(remaining_indices))):
            # 回放 V4 规格（长度 2）或被篡改的规格：V5 要求其余块的完整排列（N17）
            raise _EpisodeSpecError(
                f"{self.difficulty}: objects.swap_initiators_remaining 应为 range({len(remaining_indices)}) 的完整排列，"
                f"收到 {selected_remaining}"
            )
        # V6（计划 2.5 S5）：上面 V5 的发起顺序取值点照旧抽（主流位置不变），但不再决定发起者；
        # V5 的 objects.swap_partner_u（rand(n_swaps)）换成一次规划种子（平局打破与重排走以它播种的局部流）
        plan_seed = int(self._spec.value(
            "objects.swap_plan_seed",
            int(torch.randint(0, int(tier_decision["swap_plan"]["s5"]["plan_seed_high_exclusive"]),
                              (1,), generator=self.generator).item()),
            decision_key=f"{self.difficulty}.swap_plan",
        ))
        pairs = self._plan_swaps_newvalue_v6(plan_seed, button_obb)
        self._spec.record("objects.swap_initiators", [f"bin_{a}" for a, _b in pairs])
        for k, (a, _b) in enumerate(pairs):
            setattr(self, f"swap_pair{k+1}_idx1", self.spawned_cubes[a])
            setattr(self, f"swap_pair{k+1}_idx2", None)
        self._refresh_swap_schedule()

    def _plan_swaps_xhard(self, initiator_seq, partner_u, button_obb):
        """V5 xhard 搭档 reset 规划（计划 2.15，L48/L49/L54）；只在 xhard 的 ``_load_cubes_xhard`` 里调用。

        在 6 个名义槽位（刚生成的方块位姿）上，用 ``cube_shape_specs(hs + sweep_margin_m)`` 做交换扫掠
        可行性过滤（按无序槽位对缓存）；``button_obstacle`` 为真时把按钮底座（``button_base_state``，
        按 ``build_button`` 的最终中心与 scale）作为静止 bystander，压按钮的候选视为不可行（L54）。
        对第 k 次发起者按距离排序其余槽位，前 ``nearest_k`` 个可行者里用 ``u[k]`` 均匀选；最近
        ``nearest_k`` 个都不可行则在任一可行里均匀选；没有任何可行搭档抛真 ``SceneGenerationError``（L3）。

        逐次把 ``{"initiator": "bin_a", "partner": "bin_b"}`` 以 ``value`` 注入 ``actions.swap_pairs.<k>``；
        回放时用冻结值，并复核「发起者与规划一致、搭档扫掠可行」（N17）。结果存
        ``self._xhard_swap_partners``（第 k 次交换的搭档方块号），``step`` 的 xhard 分支据此取搭档。
        """
        swap_cfg = self._sampling["decision"]["xhard"]["swap_plan"]
        if swap_cfg["initiator_rule"] != "target_then_randperm_k_mod_cube_count":
            raise ValueError(f"VideoRepick xhard 未实现发起者规则 {swap_cfg['initiator_rule']!r}")
        if swap_cfg["partner_rule"] != "reset_plan_nearest_feasible":
            raise ValueError(f"VideoRepick xhard 未实现搭档规则 {swap_cfg['partner_rule']!r}")
        nearest_k = swap_cfg["nearest_k"]
        if isinstance(nearest_k, bool) or not isinstance(nearest_k, int) or nearest_k < 1:
            raise ValueError(f"VideoRepick xhard swap_plan.nearest_k 必须是正整数，收到 {nearest_k!r}")
        margin = float(swap_cfg["sweep_margin_m"])
        if not margin >= 0.0:
            raise ValueError(f"VideoRepick xhard swap_plan.sweep_margin_m 不能为负，收到 {margin}")
        half = float(self.cube_half_size)

        slots = []
        for cube in self.spawned_cubes:
            c, axes, _h = cube_obb2d_exact(cube, half)
            slots.append((float(c[0]), float(c[1]), float(np.arctan2(axes[1, 0], axes[0, 0]))))
        statics = []
        if swap_cfg["button_obstacle"]:
            if button_obb is None:
                raise ValueError("VideoRepick xhard：规划要把按钮底座作静止障碍，但 avoid 里没有按钮 OBB")
            statics.append(button_base_state(
                "button_base", button_obb[0], scale=float(self._sampling["positions"]["button"]["scale"])
            ))
        feasibility = _XhardSlotSweepFeasibility(_xhard_slot_states(slots, half, margin), statics)

        def resolve(k, initiator, partner):
            chosen = self._spec.value(
                f"actions.swap_pairs.{k}",
                {"initiator": f"bin_{initiator}", "partner": f"bin_{partner}"},
                decision_key="xhard.swap_plan",
            )
            if not isinstance(chosen, dict) or chosen.get("initiator") != f"bin_{initiator}":
                raise _EpisodeSpecError(
                    f"xhard: actions.swap_pairs.{k} 的发起者 {chosen!r} 与规划的 bin_{initiator} 不一致"
                )
            return _cube_index_of(chosen["partner"])

        plan = _plan_swap_partners_xhard(
            [slot[:2] for slot in slots], initiator_seq, partner_u, feasibility.feasible, nearest_k, resolve=resolve,
        )
        self._xhard_swap_partners = [step["partner"] for step in plan]
        # 只读诊断（不进规格）：逐次路径长度、候选池大小、回退，以及被检查过的槽位对
        self._xhard_swap_plan_info = {
            "slots": slots,
            "plan": plan,
            "checked_slot_pairs": len(feasibility.cache),
            "infeasible_slot_pairs": sorted(key for key, ok in feasibility.cache.items() if not ok),
        }
        return plan

    def _plan_swaps_newvalue_v6(self, plan_seed, button_obb):
        """V6 四档交换序列 reset 规划（计划 2.5，M6(a) S5）；只在 ``_load_cubes_newvalue`` 里调用。

        槽位可行性与 V5 同一口径（``_XhardSlotSweepFeasibility``：半边 + ``sweep_margin_m`` 余量、按钮底座作静止障碍），
        一次计算当前档全部 C(n,2) 个槽位对得到可行图 M；M 有孤立槽位即抛真 ``SceneGenerationError``。在 M 上跑 S5
        （``swap_uniform.plan_balanced_swaps``，局部流由 ``plan_seed`` 播种），逐次 ``value`` 进
        ``actions.swap_pairs.<k>``；回放用冻结值并复核「槽位对可行、没有立即换回」（N17，违反抛 ``EpisodeSpecError``）。
        ``record`` ``objects.swap_plan`` = {counts, range, tries, undo}。返回 ``[(发起者, 搭档), ...]``。
        """
        swap_cfg = self._sampling["decision"][self.difficulty]["swap_plan"]
        if swap_cfg["initiator_rule"] != "s5_balanced_pair" or swap_cfg["partner_rule"] != "s5_balanced_greedy":
            raise ValueError(f"VideoRepick {self.difficulty} 未实现交换规则 {swap_cfg['initiator_rule']!r}/{swap_cfg['partner_rule']!r}")
        s5 = swap_uniform.parse_inner_swap_plan_cfg(swap_cfg["s5"])
        margin = float(swap_cfg["sweep_margin_m"])
        if not margin >= 0.0:
            raise ValueError(f"VideoRepick {self.difficulty} swap_plan.sweep_margin_m 不能为负，收到 {margin}")
        half = float(self.cube_half_size)
        slots = []
        for cube in self.spawned_cubes:
            c, axes, _h = cube_obb2d_exact(cube, half)
            slots.append((float(c[0]), float(c[1]), float(np.arctan2(axes[1, 0], axes[0, 0]))))
        statics = []
        if swap_cfg["button_obstacle"]:
            if button_obb is None:
                raise ValueError(f"VideoRepick {self.difficulty}：规划要把按钮底座作静止障碍，但 avoid 里没有按钮 OBB")
            statics.append(button_base_state(
                "button_base", button_obb[0], scale=float(self._sampling["positions"]["button"]["scale"])
            ))
        feasibility = _XhardSlotSweepFeasibility(_xhard_slot_states(slots, half, margin), statics)
        graph = swap_uniform.slot_pair_graph(len(slots), feasibility.feasible)
        self._spec.record("layout.swap_graph", [[int(a), int(b)] for a, b in swap_uniform.graph_edges(graph)])
        isolated = swap_uniform.isolated_slots(graph)
        if isolated or (s5["require_connected_graph"] and not swap_uniform.graph_connected(graph)):
            raise _RealSceneGenerationError(
                f"{self.difficulty}: 槽位 {isolated} 没有扫掠可行的搭档（可行槽位对 {swap_uniform.graph_edges(graph)}）"
            )
        local = torch.Generator()
        local.manual_seed(int(plan_seed))
        plan = swap_uniform.plan_balanced_swaps(graph, int(self.swap_times), local, score=s5["score"],
                                                budget=int(s5["range_retry_budget"]),
                                                accept_range=int(s5["accept_range"]), forbid_undo=True)
        if plan is None:
            raise _RealSceneGenerationError(f"{self.difficulty}: S5 无法规划 {self.swap_times} 次交换")
        pairs = []
        for k, (a, b) in enumerate(plan.pairs):
            chosen = self._spec.value(
                f"actions.swap_pairs.{k}",
                {"initiator": f"bin_{a}", "partner": f"bin_{b}"},
                decision_key=f"{self.difficulty}.swap_plan",
            )
            try:
                pairs.append((_cube_index_of(chosen["initiator"]), _cube_index_of(chosen["partner"])))
            except (TypeError, KeyError, ValueError, IndexError) as exc:
                raise _EpisodeSpecError(f"{self.difficulty}: actions.swap_pairs.{k} 形状非法：{chosen!r}") from exc
        problems, stats = swap_uniform.verify_swap_sequence(graph, pairs, forbid_undo=True)
        if problems:
            raise _EpisodeSpecError(f"{self.difficulty}: 交换序列违反 V6 S5 规则：" + "；".join(problems))
        summary = stats.summary()
        summary["tries"] = int(plan.tries)
        self._spec.record("objects.swap_plan", summary)
        self._newvalue_swap_partners = [int(b) for _a, b in pairs]
        # 只读诊断（不进规格）
        self._newvalue_swap_plan_info = {
            "slots": slots,
            "plan": [{"initiator": a, "partner": b, "slot_a": sa, "slot_b": sb}
                     for (a, b), (sa, sb) in zip(pairs, stats.slot_pairs)],
            "graph_edges": swap_uniform.graph_edges(graph),
            "summary": summary,
            "checked_slot_pairs": len(feasibility.cache),
            "infeasible_slot_pairs": sorted(key for key, ok in feasibility.cache.items() if not ok),
        }
        return pairs

    def _newvalue_planned_partner(self, sweep_index, initiator):
        """第 ``sweep_index`` 次交换 reset 时规划好的搭档 actor。"""
        partners = getattr(self, "_newvalue_swap_partners", None)
        if partners is None or sweep_index >= len(partners):
            raise SpecBindingError(f"{self.difficulty}: 第 {sweep_index} 次交换没有 reset 规划的搭档")
        partner = self.spawned_cubes[partners[sweep_index]]
        if partner is initiator:
            raise SpecBindingError(f"{self.difficulty}: 第 {sweep_index} 次交换规划的搭档与发起者是同一块")
        return partner

    def _sweep_checks_enabled(self):
        """D5（H2）：几何检查只在「甲通道」或新值档乙通道开启；原三档乙通道仍不检查。"""
        return self._episode_spec is not None or is_newvalue_difficulty(self.difficulty)



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            if self._sweep_checks_enabled():
                # 初始化复核：读实际碰撞盒查方块两两之间；构造期与正式 reset 各走一次
                # （V4 D5：甲通道或 xhard 乙通道开启，原三档乙通道仍不跑）
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
            # V4 xhard：静止/交换段改用只吞 AttributeError 的等待函数（见 _solve_hold_obj_xhard），
            # 否则 D5 抛出的 BinCollisionError 会被 solve_hold_obj 的裸 except 吞掉并死循环；原三档仍用原函数
            hold_fn = _solve_hold_obj_xhard if is_newvalue_difficulty(self.difficulty) else solve_hold_obj
            tasks = [
            {
                "func": (lambda: is_obj_pickup(self, obj=self.target_cube_1)),
                "name": f"pick up the cube",
                "subgoal_segment":f"pick up the cube at <>",
                "choice_label": "pick up the cube",
                "demonstration": True,
                "failure_func": lambda:None,
                "solve": lambda env, planner: [solve_pickup(env, planner, obj=self.target_cube_1)],
                'segment':self.target_cube_1,
            },{
                "func": (lambda: is_obj_dropped(self, obj=self.target_cube_1)),
                "name": "drop the cube on the table",
                "subgoal_segment":f"drop the cube on the table",
                "choice_label": "put it down",
                "demonstration": True,
                "failure_func": lambda: None,
                "solve": lambda env, planner: [solve_putdown_whenhold(env, planner,release_z=0.03)]
                }, 
            ]
            if self.swap_times>=1:
                tasks.append(   {
                                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=20),
                                "name": "static",
                                "subgoal_segment":"static",
                                "demonstration": True,
                                "failure_func": None,
                                "solve": lambda env, planner: [solve_reset(env,planner),hold_fn(env, planner, static_steps=20)],
                                },)
            if self.swap_times>=1:
                for count in range(self.swap_times):
                    if is_newvalue_difficulty(self.difficulty):
                        # V6 审查修复 N11（用户「n11 a」）：第 k 段的完成判定与等待解法都钉到该段的绝对结束步
                        # swap_schedule[k][3]（首段任务成为当前任务时 step() 会按 start_step 刷新时间表），见 _solve_hold_until_step_xhard
                        tasks.append(   {
                                "func": lambda k=count: bool(is_static(self)) and int(self.elapsed_steps) >= int(self.swap_schedule[k][3]),
                                "name": "static",
                                "subgoal_segment":"static",
                                "demonstration": True,
                                "failure_func": None,
                                "specialflag":"swap",
                                "solve": lambda env, planner, k=count: [_solve_hold_until_step_xhard(env, planner, self.swap_schedule[k][3])],
                                },)
                        continue
                    tasks.append(   {
                                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=self.swap_schedule[-1][3]-self.swap_schedule[-1][2]),
                                "name": "static",
                                "subgoal_segment":"static",
                                "demonstration": True,
                                "failure_func": None,
                                "specialflag":"swap",
                                "solve": lambda env, planner: [hold_fn(env, planner, static_steps=self.swap_schedule[-1][3]-self.swap_schedule[-1][2])],
                                },)
                
            tasks.append(             {
                                "func": lambda:reset_check(self),
                                "name": "NO RECORD",
                                "subgoal_segment":"NO RECORD",
                                "demonstration": True,
                                "failure_func": None,
                                "solve": lambda env, planner: [ solve_strong_reset(env,planner)],
                                },)
            ordinal_words = [
                "first",
                "second",
                "third",
                "fourth",
                "fifth",
                "sixth",
                "seventh",
                "eighth",
                "ninth",
                "tenth",
            ]
            # V6 审查修复 M1（用户「m1 a」）：新四档「提前按按钮即失败」的时间窗随轮次滚动——每轮抓取／放下各用自己的计时器
            # （从该任务成为当前任务起算），首轮保留原 [50,500] 步、后续轮次从 0 步起，窗口覆盖到最后一次放下；
            # 原三档仍用单一计时器 2/3（从首次抓取起 [50,500] 步，不重置）。
            rolling_window = is_newvalue_difficulty(self.difficulty)
            for i in range(self.num_repeats):
                ordinal = ordinal_words[i] if i < len(ordinal_words) else f"{i+1}th"
                pick_timer = f"xhard_pick_{i}" if rolling_window else 2
                put_timer = f"xhard_put_{i}" if rolling_window else 3
                window_min = 50 if (not rolling_window or i == 0) else 0
                tasks.append(  {
                        "func": (lambda: is_obj_pickup(self, obj=self.target_cube_1)),
                        "name": f"pick up the correct cube for the {ordinal} time" ,
                        "subgoal_segment":f"pick up the correct cube at <> for the {ordinal} time" ,
                        "choice_label": "pick up the cube",
                        "demonstration": False,
                        "failure_func": lambda pick_timer=pick_timer, window_min=window_min: [
                            is_any_obj_pickup(self,[cube for cube in self.spawned_cubes if cube != self.target_cube_1]),
                            timewindow(self, lambda: is_button_pressed(self, obj=self.button_left),min_steps=window_min,max_steps=500,timewindow_timer=pick_timer,),],
                        "solve": lambda env, planner: [solve_pickup(env, planner, obj=self.target_cube_1)],
                        'segment':self.target_cube_1,
                    },)
                
                tasks.append({
                        "func": lambda: is_obj_dropped(self,obj=self.target_cube_1),
                    "name": "put it down",
                    "subgoal_segment":f"put it down",
                    "choice_label": "put it down",
                        "demonstration": False,
                        "failure_func": lambda put_timer=put_timer, window_min=window_min:[
                            is_any_obj_pickup(self,[cube for cube in self.spawned_cubes if cube != self.target_cube_1]),
                            timewindow(self, lambda: is_button_pressed(self, obj=self.button_left),min_steps=window_min,max_steps=500,timewindow_timer=put_timer,),],
                        "solve": lambda env, planner: solve_putdown_whenhold(env, planner,release_z=0.01)
                    })

            tasks.append({
                    "func": lambda: is_button_pressed(self, obj=self.button_left),
                    "name": "press the button to finish",
                    "subgoal_segment":f"press the button at <> to finish",
                    "choice_label": "press the button to finish",
                    "demonstration": False,
                    "failure_func":lambda: is_any_obj_pickup(self,[cube for cube in self.spawned_cubes]),
                    "solve": lambda env, planner: solve_button(env, planner, obj=self.button_left),
                    "segment":self.cap_link 
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

    def _get_obs_extra(self, info: Dict):
        return dict()



    def evaluate(self,solve_complete_eval=False):
        previous_failure = getattr(self, "failureflag", None)
        self.successflag = torch.tensor([False])
        if previous_failure is not None and bool(previous_failure.item()):
            self.failureflag = previous_failure
        else:
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

    def _get_other_bins_for_pair(self, idx_a: int, idx_b: int):
        """Return bins that are not part of the provided pair indices."""
        if not hasattr(self, "spawned_bins"):
            return []

        total_bins = len(self.spawned_cubes)
        if idx_a >= total_bins or idx_b >= total_bins:
            return []

        # Prefer precomputed lists when available
        if hasattr(self, "otherbins") and idx_a < len(self.otherbins):
            other_candidates = [
                bin_actor
                for bin_actor in self.otherbins[idx_a]
                if bin_actor is not self.spawned_cubes[idx_b]
            ]
            return other_candidates

        return [
            bin_actor
            for i, bin_actor in enumerate(self.spawned_cubes)
            if i not in (idx_a, idx_b)
        ]

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

    def _select_swap_pair_from_positions(self, positions, generator=None):
        """Select one swap pair given current planned positions."""
        num_bins = len(positions)
        if num_bins < 2:
            return None

        candidate_map = self._compute_dynamic_swap_candidates(positions)
        valid_indices = [idx for idx, cands in candidate_map.items() if cands]
        if not valid_indices:
            return None

        if generator is None:
            generator = self.generator

        # 交换搭档是「事件发生时才知道」的量：按事件序号分开冻结（方案 8.2）
        event_index = getattr(self, "_native_swap_event_index", -1) + 1
        self._native_swap_event_index = event_index
        first_idx = valid_indices[
            self._spec.value(
                f"actions.swap_pairs.{event_index}.first_choice",
                int(torch.randint(0, len(valid_indices), (1,), generator=generator).item()),
            )
        ]
        candidates = candidate_map[first_idx]
        second_idx = candidates[
            self._spec.value(
                f"actions.swap_pairs.{event_index}.second_choice",
                int(torch.randint(0, len(candidates), (1,), generator=generator).item()),
            )
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
        actual_initiator = self.spawned_cubes.index(initiator)
        actual_partner = self.spawned_cubes.index(resolved_partner)
        reference = self._get_actor_position(initiator)
        candidates = [
            (index, self._get_actor_position(actor))
            for index, actor in enumerate(self.spawned_cubes)
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
        """把场上三个方块读成碰撞判据用的状态；只读真实碰撞盒。"""
        return [
            object_state_from_actor(actor, f"bin_{index}")
            for index, actor in enumerate(self.spawned_cubes)
            if actor is not None
        ]

    def _check_swap_sweep_from_actual(self, sweep_index, initiator, partner):
        """从实际位姿对整段交换路径做连续检查；命中即抛错中止该样本。"""
        states = {}
        for index, actor in enumerate(self.spawned_cubes):
            if actor is None:
                continue
            states[index] = object_state_from_actor(actor, f"bin_{index}")
        a = self.spawned_cubes.index(initiator)
        b = self.spawned_cubes.index(partner)
        bystanders = [state for index, state in sorted(states.items()) if index not in (a, b)]
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

    def _check_state_readonly(self, stage):
        """某一时刻的只读复核；返回拒绝证据，不抛错。"""
        return check_bin_state(self._object_states_for_collision(), stage=stage)

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
        if not self._sweep_checks_enabled() or not self._in_swap_window():
            return
        gap, rejection = self._check_state_readonly("substep_before")
        if rejection is not None:
            self._runtime_checks.append(
                {"kind": "substep_before", "control_step": int(self.elapsed_steps), "rejection": rejection.as_dict()}
            )
            raise BinCollisionError(rejection)

    def _after_simulation_step(self):
        super()._after_simulation_step()
        if not self._sweep_checks_enabled() or not self._in_swap_window():
            return
        gap, rejection = self._check_state_readonly("substep_after")
        if rejection is not None:
            self._runtime_checks.append(
                {"kind": "substep_after", "control_step": int(self.elapsed_steps), "rejection": rejection.as_dict()}
            )
            raise BinCollisionError(rejection)

    def _refresh_swap_schedule(self,start_step=400):
        # 通式：第 k 次 swap 占 [start_step+50k, start_step+50(k+1)]；1/2/3 次时与原三分支逐项相同，0 次时不赋值
        if self.swap_times < 1:
            return
        self.swap_schedule = [
            (getattr(self, f"swap_pair{k+1}_idx1"), getattr(self, f"swap_pair{k+1}_idx2"), start_step + 50 * k, start_step + 50 * (k + 1))
            for k in range(self.swap_times)
        ]

#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):


       
        if self.current_task_specialflag=="swap":
            if self.static_flag==False:
                self.static_flag=True
                self.start_step=int(self.elapsed_steps.item())
                self._refresh_swap_schedule(self.start_step)
                logger.debug("tag!")
             
        if self.static_flag==True:
            for i in range(len(self.swap_schedule)):
                start = self.swap_schedule[i][2]
                end = self.swap_schedule[i][3]
                if self.elapsed_steps in range (start,end):
                    # Select corresponding swap pair based on index
                    pair_idx1 = getattr(self, f'swap_pair{i+1}_idx1')
                    pair_idx2 = getattr(self, f'swap_pair{i+1}_idx2')

                    if pair_idx2 is None and pair_idx1 is not None:
                        if is_newvalue_difficulty(getattr(self, "difficulty", None)):
                            # 新值档搭档用 reset 时规划好的，不再按实际 XY 取最近邻；
                            # 下面 D5 的实际位姿扫掠检查照旧跑，作运行时守卫
                            closest_actor = self._newvalue_planned_partner(i, pair_idx1)
                        else:
                            reference_pos = self._get_actor_position(pair_idx1)
                            closest_actor = None
                            closest_dist = float("inf")
                            for candidate in self.spawned_cubes:
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
                            # 再从**实际**起态做整段连续几何检查。关闭态两项都不跑。
                            if self._episode_spec is not None:
                                self._verify_swap_binding(i, pair_idx1, closest_actor)
                            # V4 D5（H2）：扫掠检查改为「甲通道，或新值档乙通道」；原三档乙通道仍不跑。
                            # 搭档身份核验只有甲的规格里预写了搭档，乙通道改为只读记录（见下）。
                            if self._sweep_checks_enabled():
                                self._check_swap_sweep_from_actual(i, pair_idx1, closest_actor)
                            if is_newvalue_difficulty(self.difficulty):
                                self._spec.record(
                                    f"actions.swap_pairs.{i}",
                                    {"initiator": f"bin_{self.spawned_cubes.index(pair_idx1)}",
                                     "partner": f"bin_{self.spawned_cubes.index(closest_actor)}"},
                                )
                                self._spec.record(
                                    f"actions.swap_windows.{i}", [int(start), int(end)]
                                )
                            setattr(self, f'swap_pair{i+1}_idx2', closest_actor)
                            self._refresh_swap_schedule(self.start_step)


            for idx_a, idx_b, start_step, end_step in self.swap_schedule:
                if idx_a is None or idx_b is None:
                    continue
                if self.elapsed_steps >= int(start_step) and self.elapsed_steps <= int(end_step):
                        
                    swap_flat_two_lane(
                                    self,
                                    cube_a=idx_a,
                                    cube_b=idx_b,
                                    start_step=start_step,
                                    end_step=end_step,
                                    cur_step=self.elapsed_steps,
                                    lane_offset=0.07,
                                    smooth=True,
                                    keep_upright=True,
                                    other_cube=[b for b in self.spawned_cubes if b not in (idx_a, idx_b)],  # Keep all other bins in place to prevent collision during swap
                                )



        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
