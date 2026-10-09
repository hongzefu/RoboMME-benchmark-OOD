"""Pin file for official package thresholds and boundaries (a pin file allowed by R8; referenced only by tests/robomme_ood/unit/robomme/ and tests/robomme_ood/unit/common/).

This file centrally registers the official thresholds, time windows, concrete coordinates/steps of hand-placed cases used by T3 tests, and the placement heights driving the offline world.
Each constant notes its official source ``file::function`` anchor (official commit 1fadc0ec, src/robomme frozen); test files no longer write these values,
and boundaries are always written as ``threshold +/- EPS``. Any value change must be checked against the official source at the anchor.
"""
from __future__ import annotations

import math

# step for boundary cases: take EPS on each side of a threshold
EPS = 0.001

# ---------------------------------------------------------------- Thresholds (subgoal_evaluate_func.py)

# subgoal_evaluate_func.py::is_obj_pickup -- object counts as picked up only if its height is strictly greater than this
PICKUP_Z = 0.05
# subgoal_evaluate_func.py::is_obj_dropped (in_demonstration branch) -- object counts as dropped only if its height <= this and it is not grasped
DROPPED_Z = 0.035
# subgoal_evaluate_func.py::is_obj_dropped_onto -- horizontal distance threshold for placing onto the target (<=)
DROP_ONTO_XY = 0.05
# subgoal_evaluate_func.py::is_bin_pickup -- container counts as picked up only if its height is strictly greater than this
BIN_PICKUP_Z = 0.15
# subgoal_evaluate_func.py::is_bin_putdown -- container counts as put down only if its height <= this, not grasped, and tcp above PICKUP_Z
BIN_PUTDOWN_Z = 0.07
# subgoal_evaluate_func.py::is_button_pressed -- button counts as pressed only if its depth is strictly greater than this
BUTTON_DEPTH = 0.005
# subgoal_evaluate_func.py::is_obj_pushed_onto(must_gripper_open)/check_block_away_gripper -- open if both fingers are greater than this
GRIPPER_OPEN = 0.02
# subgoal_evaluate_func.py::is_A_pickup_notB -- grasp-end height strictly greater than this
PEG_PICKUP_Z = 0.1
# subgoal_evaluate_func.py::is_A_insert_notB(threashold) -- distance from the insertion end to the box strictly less than this
INSERT_XY = 0.05
# subgoal_evaluate_func.py::is_A_insert_notB -- when |tcp_y - box_y| is less than this, the side is judged by the grasp end instead
DIRECTION_NEAR_ZERO = 1e-3
# subgoal_evaluate_func.py::is_obj_swing_onto default parameters (PatternLock touching a button and _wrong_button_touch use the defaults)
TOUCH_XY = 0.01
TOUCH_Z = 0.1
# subgoal_evaluate_func.py::is_obj_stopped_onto -- cube_half_size x 3; panda cube half size 0.02 (pick_cube_cfgs) -> 0.06
STOP_ONTO_XY = 0.06
CUBE_HALF = 0.02

# ---------------------------------------------------------------- Thresholds inside tasks

# SwingXtimes.py::_load_scene subtask is_obj_swing_onto(distance_threshold, z_threshold) and the entry threshold of SwingXtimes.py::step
SWING_ENTER_XY = 0.03
SWING_ENTER_Z = 0.12
# SwingXtimes.py::step -- exit hysteresis threshold (>= entry threshold)
SWING_EXIT_XY = 0.04
# RouteStick.py::_load_scene subtask is_obj_swing_onto(distance_threshold=0.03, z_threshold=self.z_threshold)
ROUTE_TOUCH_XY = 0.03
ROUTE_TOUCH_Z = 0.15
# MoveCube.py::evaluate -- is_obj_pushed_onto(distance_threshold=cube_half_size*2*1.2), panda half size 0.02 -> 0.048
PUSH_ONTO_XY = 0.048
# VideoUnmask.py::step/ButtonUnmask.py::step/*UnmaskSwap.py::step -- lift_and_drop_objects_back_to_original(0, 64),
# in the first half of the window (< 32 steps) the containers are at (10, 10, 10); at step 32 they are put back
REVEAL_DROP_STEP = 32
REVEAL_END_STEP = 64
REVEAL_AWAY_Z = 10.0
# VideoRepick.py::_initialize_episode -- the button failure of the pick-and-place subtask is wrapped in timewindow(min_steps=50, max_steps=500)
REPICK_BUTTON_WINDOW = (50, 500)
# VideoUnmask.py::_load_scene -- demo segment static_check(static_steps=64)
VIDEO_UNMASK_STATIC_STEPS = 64
# BinFill.py::step -- when dynamic, the lift window of block idx is [0, idx*100]; hard has at most 12 blocks -> ends at step 1100 at the latest,
# online cases start at step 2000 so that double positions are set only by the test
BINFILL_ONLINE_START = 2000
# BinFill.py::_initialize_episode -- blocks placed into the bin are moved to (10, 10, 0) by is_obj_dropped_onto_delete
BINFILL_REMOVED_XYZ = (10.0, 10.0, 0.0)

# ---------------------------------------------------------------- StopCube hand-computed cases

# StopCube.py::_initialize_episode and StopCube.py::step: the cube shuttles between two endpoints, move_interval steps per segment, smoothstep interpolation,
# passing the target center exactly at segment midpoints. Hand-computed below for a concrete (move_interval, stop_time):
#   visits[k-1] = cur_step of move_straight_line at the k-th pass through the target center;
#   window = closed interval [mi*(st-1), mi*st] of correct_timestep; deadline = mi*st (beyond it counts as failure);
#   off_target_step = a moment when the cube is still far from the target (1/6 into the first segment)
STOPCUBE_CASES = (
    {"move_interval": 60, "stop_time": 3, "visits": (30, 90, 150, 210, 270), "window": (120, 180), "deadline": 180,
     "off_target_step": 10},
    {"move_interval": 80, "stop_time": 2, "visits": (40, 120, 200, 280, 360), "window": (80, 160), "deadline": 160,
     "off_target_step": 13},
)

# ---------------------------------------------------------------- Offline world placement heights (chosen relative to the thresholds above)

LIFT_Z = 0.10        # object height after pickup: > PICKUP_Z, < SWING_ENTER_Z (a cube carried above a target counts as swung onto it)
TABLE_Z = 0.02       # object height after drop: <= DROPPED_Z
TCP_UP_Z = 0.15      # tcp height after release: > PICKUP_Z
CARRY_HIGH_Z = 0.2   # height clear of all targets while carrying/detouring: > SWING_ENTER_Z
BIN_UP_Z = 0.2       # height of a lifted container: > BIN_PICKUP_Z
PEG_LIFT_Z = 0.15    # height of a lifted peg: > PEG_PICKUP_Z
STICK_TOUCH_Z = 0.05  # tcp height when the stick touches a target: < TOUCH_Z
STICK_HIGH_Z = 0.3   # stick detour height: > ROUTE_TOUCH_Z
DETOUR_OFFSET = 0.08  # normal distance of a RouteStick detour point from the "previous target -> this target" line (cross-product sign only decides the side)
PEG_END_OFFSET = 0.08  # y distance from the grasp end to the box center in insertion cases (> INSERT_XY + EPS, always farther than the insertion end)

# ---------------------------------------------------------------- Wrapper-layer fixed values

# episode_config_resolver.py::make_env_for_episode -- the four fixed arguments passed to gym.make
GYM_MAKE_FIXED_KWARGS = {"obs_mode": "rgb+depth+segmentation", "control_mode": "pd_joint_pos",
                         "render_mode": "rgb_array", "reward_mode": "dense"}
# reset_panda.py::get_reset_panda_param("action") -- Panda home joint angles + gripper open (1.0)
HOME_ACTION = (0.0, 0.0, 0.0, -math.pi / 2, 0.0, math.pi / 2, math.pi / 4, 1.0)
