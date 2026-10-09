"""Pin file for the action-space wrapper and demo timing tests (a pin file allowed by R8; referenced only by tests/robomme_ood/unit/wrappers/).

Each constant notes its official source ``file::symbol`` anchor (official commit 1fadc0ec, src/robomme frozen; the hard copy is identical).
Tests do not read these values from the modules under test but compare against the pins here -- if a module under test changes a value, the tests fail.
"""
from __future__ import annotations

# DemonstrationWrapper.py::get_demonstration_trajectory, EndeffectorDemonstrationWrapper.py::step,
# MultiStepDemonstrationWrapper.py::_get_planner, OraclePlannerDemonstrationWrapper.py::reset --
# joint_vel_limits constructor argument of the PatternLock/RouteStick stick planner
STICK_JOINT_VEL_LIMITS = 0.3

# DemonstrationWrapper.py::__init__ -- total screw attempts (_demo_screw_max_attempts) and total RRT* attempts (_demo_rrt_max_attempts) in the demo phase
DEMO_SCREW_ATTEMPTS = 1
DEMO_RRT_ATTEMPTS = 3

# OraclePlannerDemonstrationWrapper.py::__init__ -- _oracle_screw_max_attempts, _oracle_rrt_max_attempts
ORACLE_SCREW_ATTEMPTS = 3
ORACLE_RRT_ATTEMPTS = 3
