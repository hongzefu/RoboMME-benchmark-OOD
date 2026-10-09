"""Goal text of the 16 tasks' new-value tiers (``utils/task_goal.get_language_goal``): each delivered cell builds the scene offline from packaged formal episode 1,
and the goal text must be bound to this episode's actual objects, counts, ordinals and directions.

Expectations are derived independently: English cardinal/ordinal words use this file's small table (independent of production's word table), colors and counts use the episode's actual state.
"""
from __future__ import annotations

import pytest

from robomme_ood.robomme_env.utils import task_goal

from . import cells as C
from . import offline_scene as O

CARDINAL = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
            10: "ten", 11: "eleven", 12: "twelve"}
ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
           9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth"}


class _Wrapper:
    """In the evaluation chain the caller passes a wrapper: ``self.env.unwrapped`` gets the env, other attributes forward to the env."""

    def __init__(self, env):
        self.env = type("E", (), {"unwrapped": env})()
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


def _goals(task, tier):
    _, env = C.replayed(task, tier, 0)
    goals = task_goal.get_language_goal(_Wrapper(env), task)
    assert goals and all(isinstance(g, str) and g.strip() for g in goals)
    return env, goals


@pytest.mark.parametrize("task,tier", O.cells_of("PickXtimes", "SwingXtimes"))
def test_repeat_count_and_colour(task, tier):
    env, goals = _goals(task, tier)
    for g in goals:
        assert f"the {env.target_color_name} cube" in g
        assert f" {CARDINAL[env.num_repeats]} times" in g


@pytest.mark.parametrize("task,tier", O.cells_of("VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"))
def test_unmask_lists_hidden_colours_in_pick_order(task, tier):
    env, goals = _goals(task, tier)
    n = getattr(env, "xhard_pick_count", None) or env.pick_times
    (g,) = goals
    pos = [g.index(f"hiding the {c} cube") for c in env.color_names[:n]]
    assert pos == sorted(pos), "color order matches pick order"
    assert g.count("hiding the") == n
    assert g.startswith("first press") == task.startswith("Button")


@pytest.mark.parametrize("tier", O.tiers_of("BinFill"))
def test_binfill_counts_per_colour(tier):
    env, goals = _goals("BinFill", tier)
    for colour in ("red", "green", "blue"):
        n = getattr(env, f"{colour}_cubes_target_number")
        for g in goals:
            if n:
                noun = "cube" if n == 1 else "cubes"
                assert f"{CARDINAL[n]} {colour} {noun}" in g
            else:
                assert f" {colour} " not in g


@pytest.mark.parametrize("tier", O.tiers_of("VideoPlaceButton"))
@pytest.mark.parametrize("k", range(C.REPLAY_ROWS))
def test_vpb_single_sentence_bound_to_before_or_after(tier, k):
    _, env = C.replayed("VideoPlaceButton", tier, k)
    (g,) = task_goal.get_language_goal(_Wrapper(env), "VideoPlaceButton")
    assert f"the {env.target_color_name} cube" in g
    phrase = {"before": "last placed before", "after": "first placed after"}[env.target_target_language]
    other = {"before": "first placed after", "after": "last placed before"}[env.target_target_language]
    assert phrase in g and other not in g


@pytest.mark.parametrize("tier", O.tiers_of("VideoPlaceOrder"))
def test_vpo_ordinal(tier):
    env, goals = _goals("VideoPlaceOrder", tier)
    for g in goals:
        assert f"the {ORDINAL[env.which_in_subset]} target" in g and f"the {env.target_color_name} cube" in g


@pytest.mark.parametrize("tier", O.tiers_of("StopCube"))
def test_stopcube_ordinal(tier):
    env, goals = _goals("StopCube", tier)
    assert all(ORDINAL[env.stop_time] in g for g in goals)


@pytest.mark.parametrize("tier", O.tiers_of("VideoRepick"))
def test_videorepick_count(tier):
    env, goals = _goals("VideoRepick", tier)
    word = CARDINAL[env.num_repeats]
    assert word in goals[0]
    assert all(word in g or (word == "two" and "twice" in g) for g in goals)


@pytest.mark.parametrize("tier", O.tiers_of("PickHighlight"))
def test_pickhighlight_newvalue_wording_ends_with_button(tier):
    _, goals = _goals("PickHighlight", tier)
    assert len(goals) == 2 and all("finally press the button" in g for g in goals)
    assert not any("highlighteted" in g for g in goals)


@pytest.mark.parametrize("task,tier", O.cells_of("InsertPeg", "MoveCube", "PatternLock", "RouteStick"))
def test_fixed_wording_tasks(task, tier):
    _, goals = _goals(task, tier)
    assert len(goals) == 2 and goals[0].startswith("watch the video carefully")
