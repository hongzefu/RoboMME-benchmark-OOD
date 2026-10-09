"""Shared demonstration segment for the PatternLock/RouteStick (stick-holding robot) truth tables: following the demonstration items of the task table, moves the TCP through the stops on the demonstration path in order,
resets, then returns to the starting pose (the effect of production ``solve_swingonto(record_swing_qpos=True)`` and ``solve_strong_reset``).

Coupling points with the production solvers (this driver does not run the motion planner, it writes env attributes directly to emulate the side effects of the two solvers):

* ``env.swing_qpos``: in production ``utils/subgoal_planner_func.py::solve_swingonto`` with ``record_swing_qpos=True``
  moves the TCP above the first stop (z=0.07), closes the gripper and writes ``env.swing_qpos = env.agent.robot.qpos`` (a reference, not a copy);
  this driver first ``touch``es the first stop in the first NO RECORD item, then sets the joint angles to a hand-picked pose different from the reset pose
  ``stick_reset_qpos() + 0.25`` and writes ``env.swing_qpos = qpos.clone()``. Both ``reset_check(target_qpos=self.swing_qpos)`` in the task table
  and ``solve_strong_reset(action=self.swing_qpos)`` read this attribute, so the driver relies on two things: "the attribute name ``swing_qpos`` stays the same" and
  "the reset criterion compares joint angles"; the concrete starting-pose values are not production values.
* ``env.after_demo``: in production ``solve_strong_reset`` sets ``env.unwrapped.after_demo = True`` on every step of its ``timestep``-iteration ``env.step`` loop
  (also setting/restoring ``reset_in_proecess``); the task's ``evaluate`` only records touched stops into ``achieved_list`` when
  ``after_demo`` is true. This driver sets ``after_demo = True`` once only in the strong-reset item,
  does not step 30 times and does not touch ``reset_in_proecess``; so the timing "touches in the demonstration segment do not count, touches after the strong reset do" is hand-written by the driver per
  production semantics, and if production changes when it sets the flag or the attribute name, this driver will not follow.
* The resulting truth table only verifies that the task's ``evaluate`` verdict logic is correct under these side effects, not the two solvers themselves
  (solver behavior in real simulation is not covered at the L2 unit layer).
"""
from __future__ import annotations

import torch

from robomme_ood.robomme_env.utils import reset_panda

TOUCH_Z = 0.05  # below the stop height threshold of both tasks
TRAVEL_Z = 0.3  # above all height thresholds: touches no stop while moving


def stick_reset_qpos():
    return torch.as_tensor(reset_panda.get_reset_panda_param("qpos", gripper="stick"), dtype=torch.float32)


def touch(w, button):
    x, y = w.xyz(button)[:2]
    w.tcp_to((x, y, TOUCH_Z))


def hover(w, xy):
    w.tcp_to((float(xy[0]), float(xy[1]), TRAVEL_Z))


def run_demo(w, on_move=None):
    """Walk through the demonstration segment and stop at the first execution item; ``on_move(w, target)`` can replace how each move in the demonstration is done."""
    env = w.env
    w.agent.robot.set_qpos(stick_reset_qpos())
    swing_pose = stick_reset_qpos() + 0.25  # starting pose (any joint angles different from the reset pose)
    seen_first = False
    with w.demo_phase():
        for _ in range(500):
            if not env.task_list[w.stage]["demonstration"]:
                return
            # same as DemonstrationWrapper: call evaluate(solve_complete_eval=True) once before and once after each demonstration task
            env.evaluate(solve_complete_eval=True)
            if not env.task_list[w.stage]["demonstration"]:
                return
            task = env.task_list[w.stage]
            if task["name"] == "NO RECORD" and not seen_first:
                touch(w, env.selected_buttons[0])
                w.agent.robot.set_qpos(swing_pose)
                env.swing_qpos = w.agent.robot.qpos.clone()
                seen_first = True
            elif task["name"] == "NO RECORD" and w.agent.robot.qpos.equal(swing_pose.reshape(1, -1)):
                w.agent.robot.set_qpos(stick_reset_qpos())  # strong reset
                env.after_demo = True
            elif task["name"] == "NO RECORD":
                w.agent.robot.set_qpos(env.swing_qpos)  # back to the starting pose: TCP above the first stop
                touch(w, env.selected_buttons[0])
            else:
                i = sum(1 for t in env.task_list[: w.stage] if t["name"] != "NO RECORD")
                target = env.selected_buttons[i + 1]
                (on_move or (lambda w, t: touch(w, t)))(w, target)
            assert w.step()["fail"] is False, task["name"]
            env.evaluate(solve_complete_eval=True)
    raise AssertionError("demonstration segment not finished")
