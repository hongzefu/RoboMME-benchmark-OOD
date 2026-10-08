"""动作空间 wrapper 与演示时序测试的钉值文件（R8 允许的钉值文件；只供 tests/robomme_hard/unit/wrappers/ 引用）。

每个常量注明官方源码 ``文件::符号`` 锚点（官方 commit 1fadc0ec，src/robomme 冻结；hard 复制件与之相同）。
测试不从被测模块读这些值，而是与这里的钉值比对——被测模块改了值，测试就会失败。
"""
from __future__ import annotations

# DemonstrationWrapper.py::get_demonstration_trajectory、EndeffectorDemonstrationWrapper.py::step、
# MultiStepDemonstrationWrapper.py::_get_planner、OraclePlannerDemonstrationWrapper.py::reset ——
# PatternLock／RouteStick 的 stick 规划器构造参数 joint_vel_limits
STICK_JOINT_VEL_LIMITS = 0.3

# DemonstrationWrapper.py::__init__ —— 演示阶段 screw 总尝试次数（_demo_screw_max_attempts）与 RRT* 总尝试次数（_demo_rrt_max_attempts）
DEMO_SCREW_ATTEMPTS = 1
DEMO_RRT_ATTEMPTS = 3

# OraclePlannerDemonstrationWrapper.py::__init__ —— _oracle_screw_max_attempts、_oracle_rrt_max_attempts
ORACLE_SCREW_ATTEMPTS = 3
ORACLE_RRT_ATTEMPTS = 3
