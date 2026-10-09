"""DemonstrationWrapper.reset 的演示与初始动作时序（C09.02），官方与 hard 复制件各跑一遍。

调用真实的 ``reset`` → 真实 ``get_demonstration_trajectory`` → 真实 ``planner_denseStep._collect_dense_steps``；
只把运动规划器（mplib）换成 CPU spy（``planner_fail_safe`` 里的两个 FailAware 规划器类），任务的 ``solve`` 是测试手写的
替身。期望一律是手写事件序列：

- 事件次序：env.reset → 建规划器 → 每个演示任务「evaluate(True) → solve 的底层步 → evaluate(True)」→ 初始动作步；
  非演示任务的 solve 不被调用；
- 帧序列：演示各段按任务顺序拼接、初始动作步在最后；``NO RECORD`` 帧被滤掉，同一句子目标出现在不同下标时全部保留；
- 演示→在线：演示期间 ``demonstration_record_traj`` 为 True，reset 返回后为 False；演示步不计入无示教步数；
- 终态额外底层步：演示中某步终局时多走一个底层步（同一动作）但不进演示帧；
- 异常恢复钩子：screw 失败（异常或 -1）转 RRT*、RRT* 失败重试到上限（钉值见 wrappers_pins）后返回 -1；solve 抛 ScrewPlanFailure 或返回 -1
  不丢已收集的帧、后续任务照常执行；其他异常向外抛且 env.step 拦截被还原；solve 不可调用 → ValueError。
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from robomme.robomme_env.utils import planner_fail_safe as pfs
from robomme.robomme_env.utils.planner_fail_safe import ScrewPlanFailure

from tests.robomme_ood.unit.robomme import official_thresholds as T
from tests.robomme_ood.unit.wrappers import wrappers_pins as P
from tests.robomme_ood.unit.wrappers.wrappers_fakes import (
    SWING7,
    ScriptedEnv,
    as_made,
    frame_value,
    planner_spy_classes,
)

PKGS = ("robomme", "robomme_ood")


@pytest.fixture(params=PKGS)
def demo_cls(request):
    mod = importlib.import_module(f"{request.param}.env_record_wrapper.DemonstrationWrapper")
    return mod.DemonstrationWrapper


@pytest.fixture
def log():
    return []


def _install_planners(monkeypatch, log, scripts=None):
    arm, stick = planner_spy_classes(log, scripts)
    # hard 包的 planner_fail_safe 是官方模块的 shim（同一模块对象），替换一处两包都生效
    hard_pfs = importlib.import_module("robomme_ood.robomme_env.utils.planner_fail_safe")
    assert hard_pfs is pfs
    monkeypatch.setattr(pfs, "FailAwarePandaArmMotionPlanningSolver", arm)
    monkeypatch.setattr(pfs, "FailAwarePandaStickMotionPlanningSolver", stick)


def _make(demo_cls, log, tasks, env_id="PickXtimes", outcomes=(), max_steps=100):
    inner = ScriptedEnv(env_id=env_id, log=log, outcomes=outcomes, task_list=tasks)
    w = demo_cls(as_made(inner), max_steps_without_demonstration=max_steps, gui_render=False)
    return w, inner


def _subgoal(env, text, demo=True):
    """模拟 sequential_task_check：直接改任务环境上的当前子目标文案与演示标志。"""
    env.unwrapped.current_task_name = text
    env.unwrapped.current_task_demonstration = demo


def _rgb_values(obs):
    return [int(np.asarray(f).reshape(-1)[0]) for f in obs["front_rgb_list"]]


# --------------------------------------------------------------------------- 主时序


def test_reset_timeline_and_frames(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log, {"screw": [2], "close": [1]})
    seen_record_flag = []

    def solve_t0(env, planner):
        seen_record_flag.append(env.unwrapped.demonstration_record_traj)
        _subgoal(env, "pick A")
        planner.move_to_pose_with_screw("P0")          # 2 个底层步
        _subgoal(env, "NO RECORD")
        planner.env.step(np.zeros(8))                   # 1 个底层步（应被滤掉）
        return 0

    def solve_t1(env, planner):  # 非演示任务：reset 不得调用
        log.append(("solve.t1",))

    def solve_t2(env, planner):
        seen_record_flag.append(env.unwrapped.demonstration_record_traj)
        _subgoal(env, "pick A")                         # 与 t0 同一句文案，不同下标
        planner.close_gripper()                         # 1 个底层步
        _subgoal(env, "put down", demo=False)           # 演示结束，切到在线子任务
        return None

    tasks = [{"name": "t0", "demonstration": True, "solve": solve_t0},
             {"name": "t1", "demonstration": False, "solve": solve_t1},
             {"name": "t2", "demonstration": True, "solve": solve_t2}]
    w, inner = _make(demo_cls, log, tasks)
    obs, info = w.reset()

    arm_kwargs = dict(debug=False, vis=False, base_pose=inner.agent.robot.pose,
                      visualize_target_grasp_pose=False, print_env_info=False)
    assert log == [
        ("env.reset",),
        ("planner.new", "arm", arm_kwargs),
        ("env.evaluate", True),
        ("planner.screw", "P0"), ("env.step", 1, "pick A"), ("env.step", 2, "pick A"),
        ("env.step", 3, "NO RECORD"),
        ("env.evaluate", True),
        ("env.evaluate", True),
        ("planner.close", None), ("env.step", 4, "pick A"),
        ("env.evaluate", True),
        ("env.step", 5, "put down"),                    # 初始动作步
    ]
    # 帧：第 3 步 NO RECORD 被滤掉；同文案 "pick A" 三帧全部保留
    assert _rgb_values(obs) == [frame_value(1), frame_value(2), frame_value(4), frame_value(5)]
    subgoals = w.demonstration_data[4]["simple_subgoal_online"]
    assert subgoals == ["pick A", "pick A", "pick A", "put down"]
    assert all(len(v) == 4 for v in obs.values())
    assert w.demonstration_data[1].shape == (4,)
    # 初始动作 = home 位
    np.testing.assert_allclose(inner.actions[-1], T.HOME_ACTION)
    # 演示 → 在线
    assert seen_record_flag == [True, True]
    assert inner.demonstration_record_traj is False
    assert w.steps_without_demonstration == 1          # 只有初始动作步计数
    assert (w.task_list_length, w.non_demonstration_task_length) == (3, 1)
    assert info["simple_subgoal_online"] == "put down" and info["status"] == "ongoing"


def test_planner_env_is_wrapper_and_step_restored(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log)
    holder = {}

    def solve(env, planner):
        holder["planner"] = planner
        holder["patched_step"] = planner.env.step
        planner.env.step(np.zeros(8))

    w, _ = _make(demo_cls, log, [{"name": "t", "demonstration": True, "solve": solve}])
    w.reset()
    assert holder["planner"].env is w                   # 规划器驱动的是 DemonstrationWrapper 本身
    assert holder["patched_step"].__name__ == "_step"  # solve 期间 env.step 被收集器拦截
    assert w.step.__func__ is demo_cls.step            # 结束后还原为原方法


def test_stick_env_uses_stick_planner_and_swing_init(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log)
    w, inner = _make(demo_cls, log, [], env_id="RouteStick")
    w.reset()
    new = [e for e in log if e[0] == "planner.new"]
    assert new == [("planner.new", "stick", dict(debug=False, vis=False, base_pose=inner.agent.robot.pose,
                                                 visualize_target_grasp_pose=False, print_env_info=False,
                                                 joint_vel_limits=P.STICK_JOINT_VEL_LIMITS))]
    np.testing.assert_allclose(inner.actions[-1], SWING7)
    assert inner.actions[-1].shape == (7,)


# --------------------------------------------------------------------------- 终态额外底层步


def test_terminal_inside_demo_takes_unrecorded_extra_step(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log, {"screw": [2]})

    def solve(env, planner):
        _subgoal(env, "demo")
        planner.move_to_pose_with_screw("P")

    # 第 2 个底层步判成功 → DemonstrationWrapper 额外多走一步（第 3 步），该步不进演示帧
    w, inner = _make(demo_cls, log, [{"name": "t", "demonstration": True, "solve": solve}],
                     outcomes=[(False, False), (True, False)])
    obs, info = w.reset()
    steps = [e[1] for e in log if e[0] == "env.step"]
    assert steps == [1, 2, 3, 4]                         # 2 规划步 + 1 额外步 + 1 初始动作步
    np.testing.assert_array_equal(inner.actions[2], inner.actions[1])  # 额外步重复同一动作
    assert _rgb_values(obs) == [frame_value(1), frame_value(2), frame_value(4)]
    assert w.demonstration_data[4]["status"] == ["ongoing", "success", "ongoing"]
    # 锁定现状：演示中的成功终局会把 episode_success 置 True，初始动作步非终局不改它
    assert w.episode_success is True


# --------------------------------------------------------------------------- screw → RRT* 恢复钩子


def test_screw_failure_falls_back_to_rrt_with_same_pose(demo_cls, log, monkeypatch):
    # screw 全部失败；RRT* 前几次交替抛异常／返回 -1，最后一次成功走 2 步
    rrt = [RuntimeError("r") if i % 2 == 0 else -1 for i in range(P.DEMO_RRT_ATTEMPTS - 1)] + [2]
    _install_planners(monkeypatch, log, {"screw": [ScrewPlanFailure("s")] * P.DEMO_SCREW_ATTEMPTS, "rrt": rrt})
    results = []

    def solve(env, planner):
        _subgoal(env, "demo")
        results.append(planner.move_to_pose_with_screw("POSE"))

    w, _ = _make(demo_cls, log, [{"name": "t", "demonstration": True, "solve": solve}])
    obs, _ = w.reset()
    calls = [e for e in log if e[0] in ("planner.screw", "planner.rrt")]
    # 演示阶段 screw 与 RRT* 各试到钉值上限；每次收到的都是同一个目标
    assert calls == [("planner.screw", "POSE")] * P.DEMO_SCREW_ATTEMPTS + [("planner.rrt", "POSE")] * P.DEMO_RRT_ATTEMPTS
    assert results == [0] and w._current_demo_task_screw_failed is False
    assert len(obs["front_rgb_list"]) == 3              # RRT* 成功的 2 步 + 初始动作步


def test_all_planning_fails_keeps_frames_and_continues(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log, {"screw": [-1] * P.DEMO_SCREW_ATTEMPTS, "rrt": [-1] * P.DEMO_RRT_ATTEMPTS})
    order, flags = [], []

    def solve_a(env, planner):
        _subgoal(env, "a")
        planner.env.step(np.zeros(8))                   # 规划前先走了 1 步
        res = planner.move_to_pose_with_screw("P")
        order.append(("a", res))
        return res                                      # 把 -1 原样交回

    def solve_b(env, planner):
        flags.append(env._current_demo_task_screw_failed)  # 新任务开始时失败标志已清零
        _subgoal(env, "b")
        planner.env.step(np.zeros(8))
        order.append(("b", 0))

    tasks = [{"name": "a", "demonstration": True, "solve": solve_a},
             {"name": "b", "demonstration": True, "solve": solve_b}]
    w, _ = _make(demo_cls, log, tasks)
    obs, _ = w.reset()
    assert order == [("a", -1), ("b", 0)]
    assert flags == [False]
    assert [e[0] for e in log].count("planner.screw") == P.DEMO_SCREW_ATTEMPTS
    assert [e[0] for e in log].count("planner.rrt") == P.DEMO_RRT_ATTEMPTS
    # solve 返回 -1 不丢 a 已收集的帧：a 1 帧 + b 1 帧 + 初始动作步
    assert w.demonstration_data[4]["simple_subgoal_online"][:2] == ["a", "b"]
    assert len(obs["front_rgb_list"]) == 3


def test_solve_raising_screw_failure_keeps_frames(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log)
    evals_before = []

    def solve_a(env, planner):
        _subgoal(env, "a")
        planner.env.step(np.zeros(8))
        raise ScrewPlanFailure("boom")

    def solve_b(env, planner):
        evals_before.append([e for e in log if e[0] == "env.evaluate"].copy())
        _subgoal(env, "b")
        planner.env.step(np.zeros(8))

    tasks = [{"name": "a", "demonstration": True, "solve": solve_a},
             {"name": "b", "demonstration": True, "solve": solve_b}]
    w, _ = _make(demo_cls, log, tasks)
    obs, _ = w.reset()
    assert w.demonstration_data[4]["simple_subgoal_online"][:2] == ["a", "b"]
    # a 抛异常后其收尾 evaluate 仍执行：b 开始前已有 a 前、a 后、b 前三次
    assert len(evals_before[0]) == 3


def test_other_exception_propagates_and_restores_step(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log)

    def solve(env, planner):
        planner.env.step(np.zeros(8))
        raise RuntimeError("solver crashed")

    w, _ = _make(demo_cls, log, [{"name": "t", "demonstration": True, "solve": solve}])
    with pytest.raises(RuntimeError, match="solver crashed"):
        w.reset()
    assert w.step.__func__ is demo_cls.step


def test_non_callable_solve_rejected(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log)
    w, _ = _make(demo_cls, log, [{"name": "broken", "demonstration": True, "solve": None}])
    with pytest.raises(ValueError, match="broken"):
        w.reset()


# --------------------------------------------------------------------------- 连续两局（A → B）


def test_second_reset_rebuilds_planner_and_drops_old_frames(demo_cls, log, monkeypatch):
    _install_planners(monkeypatch, log)
    planners = []

    def solve(env, planner):
        planners.append(planner)
        _subgoal(env, f"ep{len(planners)}")
        planner.env.step(np.zeros(8))
        _subgoal(env, "online", demo=False)

    w, inner = _make(demo_cls, log, [{"name": "t", "demonstration": True, "solve": solve}])
    w.reset()
    w.step(np.zeros(8))
    obs, _ = w.reset()
    assert len(planners) == 2 and planners[0] is not planners[1]
    assert w.demonstration_data[4]["simple_subgoal_online"] == ["ep2", "online"]
    # 第二局的帧来自第 4、5 次底层 step（第一局 2 帧 + 1 在线步之后）
    assert _rgb_values(obs) == [frame_value(4), frame_value(5)]
    assert w.steps_without_demonstration == 1
