"""官方包判定阈值与边界的钉值文件（R8 允许的钉值文件；只供 tests/robomme_ood/unit/robomme/ 与 tests/robomme_ood/unit/common/ 引用）。

本文件集中登记 T3 各测试用到的官方判定阈值、时间窗口、手摆算例的具体坐标／步数，以及驱动离线世界的摆放高度。
每个常量都注明对应的官方源码 ``文件::函数`` 锚点（官方 commit 1fadc0ec，src/robomme 冻结）；测试文件不再写这些数值，
边界一律写成 ``阈值 ± EPS``。数值改动必须同时核对锚点处的官方源码。
"""
from __future__ import annotations

import math

# 边界算例的步长：阈值两侧各取 EPS
EPS = 0.001

# ---------------------------------------------------------------- 判定阈值（subgoal_evaluate_func.py）

# subgoal_evaluate_func.py::is_obj_pickup —— 物体高度严格大于此值才算拿起
PICKUP_Z = 0.05
# subgoal_evaluate_func.py::is_obj_dropped（in_demonstration 分支）—— 物体高度 <= 此值且未被抓才算放下
DROPPED_Z = 0.035
# subgoal_evaluate_func.py::is_obj_dropped_onto —— 放到目标的水平距离阈值（<=）
DROP_ONTO_XY = 0.05
# subgoal_evaluate_func.py::is_bin_pickup —— 容器高度严格大于此值才算拿起
BIN_PICKUP_Z = 0.15
# subgoal_evaluate_func.py::is_bin_putdown —— 容器高度 <= 此值、未被抓、tcp 高于 PICKUP_Z 才算放下
BIN_PUTDOWN_Z = 0.07
# subgoal_evaluate_func.py::is_button_pressed —— 按钮深度严格大于此值才算按下
BUTTON_DEPTH = 0.005
# subgoal_evaluate_func.py::is_obj_pushed_onto(must_gripper_open)／check_block_away_gripper —— 两指都大于此值算张开
GRIPPER_OPEN = 0.02
# subgoal_evaluate_func.py::is_A_pickup_notB —— 抓取端高度严格大于此值
PEG_PICKUP_Z = 0.1
# subgoal_evaluate_func.py::is_A_insert_notB(threashold) —— 插入端到盒子的距离严格小于此值
INSERT_XY = 0.05
# subgoal_evaluate_func.py::is_A_insert_notB —— |tcp_y − box_y| 小于此值时改用抓取端判侧
DIRECTION_NEAR_ZERO = 1e-3
# subgoal_evaluate_func.py::is_obj_swing_onto 默认参数（PatternLock 触到按钮、_wrong_button_touch 用默认值）
TOUCH_XY = 0.01
TOUCH_Z = 0.1
# subgoal_evaluate_func.py::is_obj_stopped_onto —— cube_half_size × 3；panda 方块半边长 0.02（pick_cube_cfgs）→ 0.06
STOP_ONTO_XY = 0.06
CUBE_HALF = 0.02

# ---------------------------------------------------------------- 任务里的阈值

# SwingXtimes.py::_load_scene 子任务 is_obj_swing_onto(distance_threshold, z_threshold) 与 SwingXtimes.py::step 的进入阈值
SWING_ENTER_XY = 0.03
SWING_ENTER_Z = 0.12
# SwingXtimes.py::step —— 离开迟滞阈值（>= 进入阈值）
SWING_EXIT_XY = 0.04
# RouteStick.py::_load_scene 子任务 is_obj_swing_onto(distance_threshold=0.03, z_threshold=self.z_threshold)
ROUTE_TOUCH_XY = 0.03
ROUTE_TOUCH_Z = 0.15
# MoveCube.py::evaluate —— is_obj_pushed_onto(distance_threshold=cube_half_size*2*1.2)，panda 半边长 0.02 → 0.048
PUSH_ONTO_XY = 0.048
# VideoUnmask.py::step／ButtonUnmask.py::step／*UnmaskSwap.py::step —— lift_and_drop_objects_back_to_original(0, 64)，
# 窗口前半段（< 32 步）容器在 (10, 10, 10)，第 32 步放回原位
REVEAL_DROP_STEP = 32
REVEAL_END_STEP = 64
REVEAL_AWAY_Z = 10.0
# VideoRepick.py::_initialize_episode —— 拾放子任务的按钮失败包在 timewindow(min_steps=50, max_steps=500)
REPICK_BUTTON_WINDOW = (50, 500)
# VideoUnmask.py::_load_scene —— 演示段 static_check(static_steps=64)
VIDEO_UNMASK_STATIC_STEPS = 64
# BinFill.py::step —— dynamic 时第 idx 块的抬起窗口是 [0, idx*100]；hard 最多 12 块 → 最晚 1100 步结束，
# 在线算例从 2000 步开始，确保替身位置只由测试摆放
BINFILL_ONLINE_START = 2000
# BinFill.py::_initialize_episode —— 放进箱子的方块被 is_obj_dropped_onto_delete 移到 (10, 10, 0)
BINFILL_REMOVED_XYZ = (10.0, 10.0, 0.0)

# ---------------------------------------------------------------- StopCube 手算算例

# StopCube.py::_initialize_episode 与 StopCube.py::step：方块在两端点间往返，每段 move_interval 步、smoothstep 插值，
# 段中点正好经过目标中心。以下按具体 (move_interval, stop_time) 手算：
#   visits[k-1] = 第 k 次经过目标中心时 move_straight_line 的 cur_step；
#   window = correct_timestep 的闭区间 [mi·(st−1), mi·st]；deadline = mi·st（超过即判失败）；
#   off_target_step = 方块距目标还有一大段的时刻（第一段的 1/6 处）
STOPCUBE_CASES = (
    {"move_interval": 60, "stop_time": 3, "visits": (30, 90, 150, 210, 270), "window": (120, 180), "deadline": 180,
     "off_target_step": 10},
    {"move_interval": 80, "stop_time": 2, "visits": (40, 120, 200, 280, 360), "window": (80, 160), "deadline": 160,
     "off_target_step": 13},
)

# ---------------------------------------------------------------- 离线世界的摆放高度（相对上面阈值取值）

LIFT_Z = 0.10        # 抓起后的物体高度：> PICKUP_Z，< SWING_ENTER_Z（方块搬到目标上方即算摆到）
TABLE_Z = 0.02       # 放下后的物体高度：<= DROPPED_Z
TCP_UP_Z = 0.15      # 松手后 tcp 高度：> PICKUP_Z
CARRY_HIGH_Z = 0.2   # 搬运／绕行时离开所有目标的高度：> SWING_ENTER_Z
BIN_UP_Z = 0.2       # 拿起容器的高度：> BIN_PICKUP_Z
PEG_LIFT_Z = 0.15    # 拿起 peg 的高度：> PEG_PICKUP_Z
STICK_TOUCH_Z = 0.05  # stick 触到目标时 tcp 高度：< TOUCH_Z
STICK_HIGH_Z = 0.3   # stick 绕行高度：> ROUTE_TOUCH_Z
DETOUR_OFFSET = 0.08  # RouteStick 绕行点偏离「上一目标→本目标」连线的法向距离（叉积符号只看偏向哪一侧）
PEG_END_OFFSET = 0.08  # 插入算例里抓取端到盒子中心的 y 向距离（> INSERT_XY + EPS，总比插入端远）

# ---------------------------------------------------------------- 包装层固定值

# episode_config_resolver.py::make_env_for_episode —— 交给 gym.make 的四个固定参数
GYM_MAKE_FIXED_KWARGS = {"obs_mode": "rgb+depth+segmentation", "control_mode": "pd_joint_pos",
                         "render_mode": "rgb_array", "reward_mode": "dense"}
# reset_panda.py::get_reset_panda_param("action") —— Panda home 位关节角 + 夹爪张开（1.0）
HOME_ACTION = (0.0, 0.0, 0.0, -math.pi / 2, 0.0, math.pi / 2, math.pi / 4, 1.0)
