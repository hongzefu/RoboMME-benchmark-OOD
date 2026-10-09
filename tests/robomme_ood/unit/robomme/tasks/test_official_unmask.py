"""VideoUnmask and ButtonUnmask native three-tier truth tables (C05).

Shared verdicts of both tasks: "pick up container" = container height > T.BIN_PICKUP_Z (strictly greater), "put down" = height <= T.BIN_PUTDOWN_Z, not grasped, tcp above T.PICKUP_Z.
- VideoUnmask: the demo segment is T.VIDEO_UNMASK_STATIC_STEPS steps of watching still (containers lifted away and put back meanwhile), then pick up bin_0 hiding the target-color cube;
  hard requires two choices (bin_0 -> put down -> bin_1).
- ButtonUnmask: press the button first (pressed only when depth > T.BUTTON_DEPTH), then choose the same way.
The color in the goal language must be the color of the cube hidden under bin_0/bin_1 (the cube's initial xy coincides with the container).
"""
from __future__ import annotations

import numpy as np
import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

DIFFS = ("easy", "medium", "hard")
BIN_UP = T.BIN_UP_Z
REVEAL_END = T.REVEAL_END_STEP + 1  # past the reveal animation window


@pytest.fixture(params=["VideoUnmask", "ButtonUnmask"])
def task(request):
    return request.param


@pytest.fixture
def world(task):
    with OfficialWorld(task) as w:
        yield w


def _to_online(ep, task):
    env = ep.env
    if task == "VideoUnmask":
        # demo segment: the robot stays still for T.VIDEO_UNMASK_STATIC_STEPS steps (real static_check) before the subtask pointer enters the online segment
        guard = 0
        while ep.task_index < ep.first_online_index():
            ep.step()
            guard += 1
            assert guard < 200
    else:
        # wait out the reveal animation (containers lifted out of the scene and put back) before pressing the button, see test_button_unmask_press_during_reveal_fails
        ep.step(REVEAL_END)
        ep.press(env.button_left)
        ep.step()
        ep.unpress(env.button_left)
    assert ep.task_index == ep.first_online_index() + (1 if task == "ButtonUnmask" else 0)


def _pick_count(env):
    return env.configs[env.difficulty]["pick"]


def test_goal_colour_is_the_cube_under_bin0(world, task):
    for diff in DIFFS:
        ep = world.make(diff, seed=2)
        env = ep.env
        cube0 = getattr(env, f"target_cube_{env.color_names[0]}")
        np.testing.assert_allclose(cube0.xyz[:2], env.bin_0.xyz[:2], atol=1e-6)
        goals = goal_text(env)
        assert all(f"hiding the {env.color_names[0]} cube" in g for g in goals)
        if _pick_count(env) > 1:
            assert all(f"hiding the {env.color_names[1]} cube" in g for g in goals)


@pytest.mark.parametrize("diff", DIFFS)
def test_correct_selection_succeeds(world, task, diff):
    ep = world.make(diff, seed=2)
    env = ep.env
    _to_online(ep, task)
    ep.grasp(env.bin_0, z=BIN_UP)
    ep.step()
    if _pick_count(env) == 1:
        assert ep.success and not ep.fail
        return
    assert not ep.success
    ep.release(env.bin_0)
    ep.step()
    ep.grasp(env.bin_1, z=BIN_UP)
    ep.step()
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", DIFFS)
def test_wrong_container_fails(world, task, diff):
    ep = world.make(diff, seed=2)
    env = ep.env
    _to_online(ep, task)
    wrong = env.spawned_bins[-1]
    assert wrong is not env.bin_0
    ep.grasp(wrong, z=BIN_UP)
    ep.step()
    assert ep.fail and not ep.success


def test_hard_reversed_order_fails(world, task):
    ep = world.make("hard", seed=2)
    env = ep.env
    _to_online(ep, task)
    ep.grasp(env.bin_1, z=BIN_UP)
    ep.step()
    assert ep.fail and not ep.success


def test_hard_second_pick_without_putdown_fails(world, task):
    ep = world.make("hard", seed=2)
    env = ep.env
    _to_online(ep, task)
    ep.grasp(env.bin_0, z=BIN_UP)
    ep.step()
    env.bin_1.move_to(z=BIN_UP)  # pick up the second while bin_0 is still held
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("z, picked", [(T.BIN_PICKUP_Z + T.EPS, True), (T.BIN_PICKUP_Z, False)])
def test_pickup_height_threshold_is_strict(world, task, z, picked):
    ep = world.make("easy", seed=2)
    env = ep.env
    _to_online(ep, task)
    env.bin_0.move_to(z=z)
    ep.step()
    assert ep.success is picked


def test_video_unmask_motion_restarts_static_window():
    """static_check of the demo segment: as soon as the robot moves, the still timer restarts (the demo->online switch timing)."""
    with OfficialWorld("VideoUnmask") as world:
        ep = world.make("easy", seed=2)
        ep.step(T.VIDEO_UNMASK_STATIC_STEPS // 2)
        ep.move_robot()
        ep.step()
        ep.hold_robot()
        moved_at = int(ep.env.elapsed_steps)
        while ep.task_index == 0:
            ep.step()
        assert int(ep.env.elapsed_steps) - moved_at >= T.VIDEO_UNMASK_STATIC_STEPS


def test_video_unmask_bins_return_to_origin_after_reveal():
    """During the reveal window the containers are lifted away and put back at the window midpoint; at the start of the online segment all containers are at their initial positions."""
    with OfficialWorld("VideoUnmask") as world:
        ep = world.make("hard", seed=2)
        origin = [b.xyz.copy() for b in ep.env.spawned_bins]
        ep.step(10)
        assert all(b.xyz[2] == T.REVEAL_AWAY_Z for b in ep.env.spawned_bins)  # during the reveal: lifted out of the scene
        while ep.task_index == 0:
            ep.step()
        for b, o in zip(ep.env.spawned_bins, origin):
            np.testing.assert_allclose(b.xyz, o, atol=1e-6)


# the button depth "strictly greater" contract is asserted only in test_sequential_check.py::test_button_depth_strict (closest to the production function).


@pytest.mark.parametrize("press_at, fails", [
    (1, True), (T.REVEAL_DROP_STEP - 1, True), (T.REVEAL_DROP_STEP, False), (T.REVEAL_END_STEP, False),
])
def test_button_unmask_press_during_reveal_fails(press_at, fails):
    """Current official behavior (registered as conditional in the contract delta): in the first half of the reveal window (elapsed < T.REVEAL_DROP_STEP) containers are temporarily moved high out of the scene;
    if the button is already pressed then, the "picked up another container" failure condition holds on the next step -> fail (then terminated)."""
    with OfficialWorld("ButtonUnmask") as world:
        ep = world.make("easy", seed=2)
        ep.step(press_at - 1)
        ep.press(ep.env.button_left)
        ep.step(2)
        assert any(f for _, f, _ in ep.history) is fails


def test_button_unmask_failure_not_latched_by_env():
    """ButtonUnmask.evaluate resets failureflag every step (no latch); the failure terminal relies on terminated truncation, not env memory."""
    with OfficialWorld("ButtonUnmask") as world:
        ep = world.make("easy", seed=2)
        _to_online(ep, "ButtonUnmask")
        wrong = ep.env.spawned_bins[-1]
        ep.grasp(wrong, z=BIN_UP)
        ep.step()
        assert ep.fail
        ep.release(wrong)
        ep.step()
        assert not ep.fail


def test_button_unmask_pick_before_button_is_not_success():
    """Skipping the button and picking a container first: neither advances nor fails (the button subtask has no failure condition); success only after pressing the button later -- current official behavior."""
    with OfficialWorld("ButtonUnmask") as world:
        ep = world.make("easy", seed=2)
        env = ep.env
        ep.step(REVEAL_END)
        ep.grasp(env.bin_0, z=BIN_UP)
        ep.step(3)
        assert not ep.success and not ep.fail and ep.task_index == 0
        ep.press(env.button_left)
        ep.step()
        ep.step()
        assert ep.success
