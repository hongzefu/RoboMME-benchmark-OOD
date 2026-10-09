"""官方 vqa_options：16 任务的多选项（C07 选项与子目标绑定）。

放在 tasks/ 下是为了用离线世界里真实建出来的任务实例（选项读的是任务实例上的物体列表）。期望：
- 每个任务的选项动作文本按下表（独立期望，来源：官方 doc/env_format.md 的 multi_choice 说明与各任务目标语言）；
  label 从 a 起连续；带 available 的选项引用任务实例上的那份物体列表（同一对象）；
- 每个在线子目标的 choice_label 都能在选项动作里找到（录制时据此把子目标映射成选项 label）；
- 选中需要目标的选项而没给目标 → ValueError；未知任务 → 空列表；求解函数分派到正确的规划函数与物体。
"""
from __future__ import annotations

import pytest

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env.utils import vqa_options
from robomme.robomme_env.utils.vqa_options import get_vqa_options

from _official_world import OfficialWorld

ACTIONS = {
    "PickXtimes": ["pick up the cube", "place the cube onto the target", "press the button to stop"],
    "StopCube": ["move to the top of the button to prepare", "remain static", "press button to stop the cube"],
    "SwingXtimes": ["pick up the cube", "move to the top of the target", "put the cube on the table", "press the button"],
    "BinFill": ["pick up the cube", "put it into the bin", "press the button"],
    "VideoUnmaskSwap": ["pick up the container", "put down the container"],
    "VideoUnmask": ["pick up the container", "put down the container"],
    "ButtonUnmaskSwap": ["press the first button", "press the second button", "pick up the container",
                         "put down the container"],
    "ButtonUnmask": ["press the button", "pick up the container", "put down the container"],
    "VideoRepick": ["pick up the cube", "put it down", "press the button to finish"],
    "VideoPlaceButton": ["pick up the cube", "drop onto", "press the button"],
    "VideoPlaceOrder": ["pick up the cube", "drop onto", "press the button"],
    "PickHighlight": ["press button", "pick up the highlighted cube", "place the cube onto the table"],
    "InsertPeg": ["pick up the peg by grasping one end", "insert the peg from the right side",
                  "insert the peg from the left side"],
    "MoveCube": ["pick up the peg", "hook the cube to the target with the peg",
                 "close gripper and push the cube to the target", "pick up the cube", "place the cube onto the target"],
    "PatternLock": [f"move {d}" for d in ("forward", "backward", "left", "right", "forward-left", "forward-right",
                                          "backward-left", "backward-right")],
    "RouteStick": [f"move to the nearest {s} target by circling around the stick {d}"
                   for d in ("clockwise", "counterclockwise") for s in ("left", "right")],
}
# 带 available 的选项及其应引用的实例属性
AVAILABLE = {
    "PickXtimes": {"a": "all_cubes"}, "SwingXtimes": {"a": "all_cubes"}, "BinFill": {"a": "all_cubes"},
    "VideoUnmaskSwap": {"a": "spawned_bins"}, "VideoUnmask": {"a": "spawned_bins"},
    "ButtonUnmaskSwap": {"c": "spawned_bins"}, "ButtonUnmask": {"b": "spawned_bins"},
    "VideoRepick": {"a": "spawned_cubes"}, "VideoPlaceButton": {"a": "all_cubes", "b": "targets"},
    "VideoPlaceOrder": {"a": "all_cubes", "b": "targets"}, "PickHighlight": {"b": "all_cubes"},
}
TASKS = BenchmarkEnvBuilder.get_task_list()


def test_table_covers_all_tasks():
    assert set(ACTIONS) == set(TASKS)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("diff", ("easy", "hard"))
def test_options_and_choice_labels(task, diff):
    with OfficialWorld(task) as world:
        env = world.make(diff, seed=0).env
        opts = get_vqa_options(env, None, {"obj": None}, task)
    assert [o["action"] for o in opts] == ACTIONS[task]
    assert [o["label"] for o in opts] == [chr(ord("a") + i) for i in range(len(opts))]
    avail = {o["label"]: o["available"] for o in opts if "available" in o}
    expected = AVAILABLE.get(task, {})
    if task == "SwingXtimes":
        assert set(avail) == {"a", "b"} and avail["b"] == [env.target_right, env.target_left]
        avail.pop("b")
    if task == "InsertPeg":
        assert set(avail) == {"a"} and avail["a"] == env.peg_heads + env.peg_tails
        avail.pop("a")
    assert set(avail) == set(expected)
    for label, attr in expected.items():
        assert avail[label] is getattr(env, attr), (task, label)
    online_labels = {t.get("choice_label") for t in env.task_list if not t.get("demonstration")} - {None}
    assert online_labels <= set(ACTIONS[task]), online_labels - set(ACTIONS[task])


def test_unknown_task_has_no_options():
    with OfficialWorld("PickXtimes") as world:
        env = world.make("easy").env
        assert get_vqa_options(env, None, {"obj": None}, "NotATask") == []


def test_target_option_without_target_raises():
    with OfficialWorld("PickXtimes") as world:
        env = world.make("easy").env
        opts = get_vqa_options(env, None, {"obj": None}, "PickXtimes")
        with pytest.raises(ValueError, match="No available target"):
            opts[0]["solve"]()


def test_solve_dispatch(monkeypatch):
    calls = []
    for name in ("solve_pickup", "solve_button", "solve_putonto_whenhold", "solve_putonto_whenhold_binspecial"):
        monkeypatch.setattr(vqa_options, name, lambda *a, _n=name, **k: calls.append((_n, a, k)))
    planner = object()
    with OfficialWorld("PickXtimes") as world:
        env = world.make("easy").env
        chosen = env.all_cubes[0]
        opts = get_vqa_options(env, planner, {"obj": chosen}, "PickXtimes")
        for o in opts:
            o["solve"]()
    assert calls[0] == ("solve_pickup", (env, planner), {"obj": chosen})
    assert calls[1] == ("solve_putonto_whenhold", (env, planner), {"target": env.target})
    assert calls[2] == ("solve_button", (env, planner), {"obj": env.button})
    calls.clear()
    with OfficialWorld("BinFill") as world:
        env = world.make("easy").env
        get_vqa_options(env, planner, {"obj": None}, "BinFill")[1]["solve"]()
    assert calls == [("solve_putonto_whenhold_binspecial", (env, planner), {"target": env.board_with_hole})]
