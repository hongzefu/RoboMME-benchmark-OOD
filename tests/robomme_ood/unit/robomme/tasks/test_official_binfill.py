"""BinFill native three-tier truth table (C05): put the specified number of cubes of each color into the bin per the goal language, then press the button.

Expectations: for every color the number put in equals the target count exactly (stated in the goal language) -> button success; pressing one short, putting one extra, wrong color,
pressing early -> failure. Cubes put into the bin are moved out of the scene (10, 10) and are not counted afterwards.
"""
from __future__ import annotations

import numpy as np
import pytest

from _official_world import OfficialWorld, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

TASK = "BinFill"
DIFFS = ("easy", "medium", "hard")
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
# past the dynamic lift animation window (at most idx*100 steps), so double positions are set only by the test
ONLINE_START = T.BINFILL_ONLINE_START


@pytest.fixture
def world():
    with OfficialWorld(TASK) as w:
        yield w


def _targets(env):
    return {"red": env.red_cubes_target_number, "blue": env.blue_cubes_target_number,
            "green": env.green_cubes_target_number}


def _cubes(env, color):
    return {"red": env.red_cubes, "blue": env.blue_cubes, "green": env.green_cubes}[color]


def _put_into_bin(ep, env, cube):
    ep.grasp(cube)
    ep.step()
    ep.place_on(cube, env.board_with_hole)
    ep.step()


def _fill_exact(ep, env):
    used = {c: 0 for c in ("red", "blue", "green")}
    for color, count in env.binfill_language_sequence:
        for _ in range(count):
            _put_into_bin(ep, env, _cubes(env, color)[used[color]])
            used[color] += 1
    return used


def _start(world, diff, seed=1):
    ep = world.make(diff, seed=seed)
    ep.skip_demo(elapsed=ONLINE_START)
    return ep


@pytest.mark.parametrize("diff", DIFFS)
def test_exact_counts_then_button_succeeds(world, diff):
    ep = _start(world, diff)
    env = ep.env
    targets = _targets(env)
    goals = goal_text(env)
    for color, n in targets.items():
        if n > 0:
            assert all(f"{WORDS[n]} {color} cube" in g for g in goals), (color, n, goals)
    _fill_exact(ep, env)
    assert [env.red_cubes_in_bin, env.blue_cubes_in_bin, env.green_cubes_in_bin] == [
        targets["red"], targets["blue"], targets["green"]]
    assert not ep.success and not ep.fail
    ep.press(env.button)
    ep.step()
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", DIFFS)
def test_cube_put_into_bin_is_removed_from_scene(world, diff):
    ep = _start(world, diff)
    env = ep.env
    color, _ = env.binfill_language_sequence[0]
    cube = _cubes(env, color)[0]
    _put_into_bin(ep, env, cube)
    np.testing.assert_allclose(cube.xyz, T.BINFILL_REMOVED_XYZ, atol=1e-6)
    ep.step(3)  # cubes already moved out are not counted again
    assert sum([env.red_cubes_in_bin, env.blue_cubes_in_bin, env.green_cubes_in_bin]) == 1


@pytest.mark.parametrize("diff", DIFFS)
def test_button_one_short_fails(world, diff):
    ep = _start(world, diff)
    env = ep.env
    seq = env.binfill_language_sequence
    total = sum(n for _, n in seq)
    done = 0
    used = {c: 0 for c in ("red", "blue", "green")}
    for color, count in seq:
        for _ in range(count):
            if done == total - 1:
                break
            _put_into_bin(ep, env, _cubes(env, color)[used[color]])
            used[color] += 1
            done += 1
    ep.press(env.button)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_extra_cube_after_exact_fails(world, diff):
    ep = _start(world, diff)
    env = ep.env
    used = _fill_exact(ep, env)
    spare = next((c for c in env.all_cubes if c.xyz[0] < 5 and all(
        c is not x for col in used for x in _cubes(env, col)[:used[col]])), None)
    assert spare is not None, "spawn count >= target count, at least one extra cube can be placed"
    _put_into_bin(ep, env, spare)
    assert ep.fail and not ep.success


def test_wrong_color_fails_at_button(world):
    """Only medium/hard have multiple colors; put in a cube of a color outside the target colors; count mismatch -> button failure."""
    for seed in range(40):
        ep = _start(world, "hard", seed=seed)
        env = ep.env
        targets = _targets(env)
        wrong = [c for c in ("red", "blue", "green") if targets[c] == 0 and _cubes(env, c)]
        if wrong:
            break
    else:
        pytest.fail("no layout found with cubes of a non-target color")
    seq = env.binfill_language_sequence
    # place the sequence's total count, but swap the first cube for a wrong color
    total = sum(n for _, n in seq)
    wrong_cube = _cubes(env, wrong[0])[0]
    _put_into_bin(ep, env, wrong_cube)
    used = {c: 0 for c in ("red", "blue", "green")}
    placed = 1
    for color, count in seq:
        for _ in range(count):
            if placed >= total:
                break
            _put_into_bin(ep, env, _cubes(env, color)[used[color]])
            used[color] += 1
            placed += 1
    assert not ep.fail  # no failure while placing (the pick subtask accepts any cube)
    ep.press(env.button)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_early_button_fails(world, diff):
    ep = _start(world, diff)
    ep.press(ep.env.button)
    ep.step()
    assert ep.fail and not ep.success
    ep.unpress(ep.env.button)
    ep.step(2)
    assert ep.fail  # failure latches


def test_drop_outside_bin_does_not_count(world):
    ep = _start(world, "easy")
    env = ep.env
    color, _ = env.binfill_language_sequence[0]
    cube = _cubes(env, color)[0]
    ep.grasp(cube)
    ep.step()
    bx, by, _ = env.board_with_hole.xyz
    ep.release(cube, bx + T.DROP_ONTO_XY + T.EPS, by)  # just outside the placement threshold
    ep.step()
    assert sum([env.red_cubes_in_bin, env.blue_cubes_in_bin, env.green_cubes_in_bin]) == 0


def test_drop_with_closed_gripper_does_not_count(world):
    """check_block_away_gripper: not counted into the bin while the gripper is not open (both fingers <= T.GRIPPER_OPEN)."""
    ep = _start(world, "easy")
    env = ep.env
    color, _ = env.binfill_language_sequence[0]
    cube = _cubes(env, color)[0]
    ep.grasp(cube)
    ep.step()
    ep.close_gripper()
    ep.place_on(cube, env.board_with_hole)
    ep.step()
    assert sum([env.red_cubes_in_bin, env.blue_cubes_in_bin, env.green_cubes_in_bin]) == 0
    ep.open_gripper()
    ep.step()
    assert sum([env.red_cubes_in_bin, env.blue_cubes_in_bin, env.green_cubes_in_bin]) == 1
