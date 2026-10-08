"""PatternLock／RouteStick（握杆机器人）真值表的公共演示段：按任务表演示项把 TCP 依次移到演示路径上的落点，
复位、再回到起点姿态（生产 ``solve_swingonto(record_swing_qpos=True)`` 与 ``solve_strong_reset`` 的效果）。

与生产求解器的耦合点（本驱动不跑运动规划器，而是直接写环境属性来模拟两个求解器的副作用）：

* ``env.swing_qpos``：生产由 ``utils/subgoal_planner_func.py::solve_swingonto`` 在 ``record_swing_qpos=True`` 时
  把 TCP 移到第一个落点上方（z=0.07）并合爪后写 ``env.swing_qpos = env.agent.robot.qpos``（引用，非拷贝）；
  本驱动在第一个 NO RECORD 项里先 ``touch`` 第一个落点，再把关节角置为一组人为选的、不同于复位的姿态
  ``stick_reset_qpos() + 0.25`` 并写 ``env.swing_qpos = qpos.clone()``。任务表里 ``reset_check(target_qpos=self.swing_qpos)``
  与 ``solve_strong_reset(action=self.swing_qpos)`` 都读这个属性，所以驱动依赖「属性名 ``swing_qpos`` 不变、
  复位判据按关节角比对」这两点；起点姿态的具体数值不是生产值。
* ``env.after_demo``：生产由 ``solve_strong_reset`` 在其 ``timestep`` 次 ``env.step`` 循环里每步置
  ``env.unwrapped.after_demo = True``（同时置／复原 ``reset_in_proecess``）；任务 ``evaluate`` 只在
  ``after_demo`` 为真时把触碰到的落点记入 ``achieved_list``。本驱动只在强复位那一项里置一次 ``after_demo = True``，
  不 step 30 次、不碰 ``reset_in_proecess``；因此「演示段触碰不计入、强复位之后的触碰才计入」这一时序是驱动按
  生产语义手写的，生产若改了置位时机或属性名，本驱动不会跟着变。
* 由此得到的真值表只验证任务 ``evaluate`` 的判定逻辑在上述副作用下是否正确，不验证两个求解器本身
  （求解器在真仿真里的行为不在 L2 单元层覆盖）。
"""
from __future__ import annotations

import torch

from robomme_hard.robomme_env.utils import reset_panda

TOUCH_Z = 0.05  # 低于两任务的落点高度阈值
TRAVEL_Z = 0.3  # 高于全部高度阈值：移动途中不碰任何落点


def stick_reset_qpos():
    return torch.as_tensor(reset_panda.get_reset_panda_param("qpos", gripper="stick"), dtype=torch.float32)


def touch(w, button):
    x, y = w.xyz(button)[:2]
    w.tcp_to((x, y, TOUCH_Z))


def hover(w, xy):
    w.tcp_to((float(xy[0]), float(xy[1]), TRAVEL_Z))


def run_demo(w, on_move=None):
    """走完演示段，停在执行段第一项；``on_move(w, target)`` 可替换演示里每一步的移动方式。"""
    env = w.env
    w.agent.robot.set_qpos(stick_reset_qpos())
    swing_pose = stick_reset_qpos() + 0.25  # 起点姿态（任意一组不同于复位的关节角）
    seen_first = False
    with w.demo_phase():
        for _ in range(500):
            if not env.task_list[w.stage]["demonstration"]:
                return
            # 与 DemonstrationWrapper 相同：每个演示任务执行前后各调一次 evaluate(solve_complete_eval=True)
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
                w.agent.robot.set_qpos(stick_reset_qpos())  # 强复位
                env.after_demo = True
            elif task["name"] == "NO RECORD":
                w.agent.robot.set_qpos(env.swing_qpos)  # 回到起点姿态：TCP 在第一个落点上
                touch(w, env.selected_buttons[0])
            else:
                i = sum(1 for t in env.task_list[: w.stage] if t["name"] != "NO RECORD")
                target = env.selected_buttons[i + 1]
                (on_move or (lambda w, t: touch(w, t)))(w, target)
            assert w.step()["fail"] is False, task["name"]
            env.evaluate(solve_complete_eval=True)
    raise AssertionError("演示段未走完")
