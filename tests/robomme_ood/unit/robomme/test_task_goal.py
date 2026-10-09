"""官方 task_goal.get_language_goal：16 任务目标语言的分支（C07 语言与对象／次数／方向绑定）。

调用方式与 DemonstrationWrapper 相同：第一个参数是包装层（``.env.unwrapped`` 取任务实例，其余属性透传）。
期望是目标语言本身的手写片段；两条与旧测试不同、已按官方现状修正：未知任务返回 []；
SwingXtimes 写作 back-and-forth。语言与真实任务实例的绑定（同一局的次数、颜色）在 tasks/ 下的真值表里交叉核对。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from robomme.robomme_env.utils.task_goal import get_language_goal


class _Wrapper:
    def __init__(self, unwrapped=None, **attrs):
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(**(unwrapped or {})))
        self.__dict__.update(attrs)


def goal(env_id, unwrapped=None, **attrs):
    return get_language_goal(_Wrapper(unwrapped, **attrs), env_id)


def test_unknown_env_returns_empty_list():
    assert goal("NotATask") == []


@pytest.mark.parametrize("counts, fragment", [
    ({}, "put the cubes into the bin, then press the button to stop"),
    ({"red_cubes_target_number": 1}, "put one red cube into the bin"),
    ({"blue_cubes_target_number": 3}, "put three blue cubes into the bin"),
    ({"red_cubes_target_number": 1, "green_cubes_target_number": 2}, "put one red cube and two green cubes into"),
    ({"red_cubes_target_number": 2, "blue_cubes_target_number": 3, "green_cubes_target_number": 1},
     "put two red cubes, three blue cubes and one green cube into"),
    ({"red_cubes_target_number": 21}, "put 21 red cubes into"),
])
def test_binfill(counts, fragment):
    g = goal("BinFill", counts)
    assert len(g) == 2 and fragment in g[0]
    assert g[0].endswith("then press the button to stop") and g[1].endswith("and press the button to stop")


def test_pickxtimes():
    one = goal("PickXtimes", {"num_repeats": 1, "target_color_name": "red"})
    assert one == ["pick up the red cube and place it on the target, then press the button to stop"]
    many = goal("PickXtimes", {"num_repeats": 3, "target_color_name": "blue"})
    assert len(many) == 2 and "repeating this action three times" in many[0]
    assert "pick-and-place action three times" in many[1] and all("blue cube" in g for g in many)


def test_swingxtimes():
    one = goal("SwingXtimes", {"num_repeats": 1, "target_color_name": "green"})
    assert len(one) == 2 and "put it down on the left-side target" in one[0] and "repeating" not in " ".join(one)
    many = goal("SwingXtimes", {"num_repeats": 5, "target_color_name": "green"})
    assert "repeating this back-and-forth motion five times" in many[0]
    assert "right-to-left swing motion five times" in many[1]


@pytest.mark.parametrize("env_id, lead", [("VideoUnmask", "watch the video carefully, then pick up"),
                                          ("ButtonUnmask", "first press the button, then pick up")])
@pytest.mark.parametrize("picks", [1, 2])
def test_unmask_by_difficulty_pick_count(env_id, lead, picks):
    unwrapped = {"color_names": ["red", "blue", "green"], "configs": {"hard": {"pick": picks}}}
    g = goal(env_id, unwrapped, difficulty="hard")
    assert len(g) == 1 and g[0].startswith(lead) and "hiding the red cube" in g[0]
    assert ("another container hiding the blue cube" in g[0]) is (picks == 2)


@pytest.mark.parametrize("env_id, lead", [("VideoUnmaskSwap", "watch the video carefully"),
                                          ("ButtonUnmaskSwap", "first press both buttons on the table")])
@pytest.mark.parametrize("picks", [1, 2])
def test_unmask_swap_by_pick_times(env_id, lead, picks):
    g = goal(env_id, {"color_names": ["green", "red", "blue"]}, pick_times=picks)
    assert len(g) == 1 and g[0].startswith(lead) and "hiding the green cube" in g[0]
    assert ("another container hiding the red cube" in g[0]) is (picks == 2)


@pytest.mark.parametrize("lang, last", [("before", "last placed before the button"),
                                        ("after", "first placed after the button")])
def test_videoplacebutton(lang, last):
    g = goal("VideoPlaceButton", target_color_name="red", target_target_language=lang)
    assert len(g) == 4 and all(f" {lang} the button was pressed" in x for x in g[:3])
    assert last in g[3] and all("red cube" in x for x in g)


@pytest.mark.parametrize("k, word", [(1, "first"), (3, "third"), (4, "fourth")])
def test_videoplaceorder(k, word):
    g = goal("VideoPlaceOrder", target_color_name="blue", which_in_subset=k)
    assert g == [f"watch the video carefully, then place the blue cube on the {word} target it was previously placed on",
                 f"watch the video carefully and place the blue cube on the {word} target where it was placed"]


@pytest.mark.parametrize("n, frags", [
    (1, ["picked up again", "picked up again"]),
    (2, ["for two times", "picked up twice", "place down the same cube twice"]),
    (3, ["for three times", "picked up three times", "place down the same cube three times"]),
])
def test_videorepick(n, frags):
    g = goal("VideoRepick", num_repeats=n)
    assert len(g) == len(frags) and all(f in x for f, x in zip(frags, g))


@pytest.mark.parametrize("k, word", [(2, "second"), (5, "fifth")])
def test_stopcube(k, word):
    g = goal("StopCube", {"stop_time": k})
    assert g == [f"press the button to stop the cube just as it reaches the target for the {word} time",
                 f"press the button to stop the cube exactly at the target on its {word} visit"]


@pytest.mark.parametrize("env_id, fragment", [
    ("PickHighlight", "pick up all highlighted cubes"),
    ("InsertPeg", "insert it into the same side of the box"),
    ("MoveCube", "move the cube to the target in the same manner"),
    ("PatternLock", "retrace the same pattern"),
    ("RouteStick", "navigate around the sticks on the table, following the same path"),
])
def test_fixed_goals_have_two_phrasings(env_id, fragment):
    g = goal(env_id)
    assert len(g) == 2 and len(set(g)) == 2 and any(fragment in x for x in g)
