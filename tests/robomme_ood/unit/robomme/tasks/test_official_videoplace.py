"""VideoPlaceButton and VideoPlaceOrder native three-tier truth tables (C05, C07 language binding).

The demo segment is really executed by the generic driver below according to each demo subtask's meaning (pick up -> place on that subtask's segment target -> button -> put back on the table
-> still -> swap -> reset); the driver also records "targets placed in order during the demo". Expectations:
- VideoPlaceButton: goal language says before -> the correct target is the last one placed before the button press; after -> the first one placed after the button press;
- VideoPlaceOrder: goal language says "the k-th" -> the correct target is the target of the k-th placement in the demo (time order, not spatial order; the button takes no index);
- placing on another target or grabbing the wrong cube -> failure; in hard, after target plates are swapped, verdicts are by identity.
"""
from __future__ import annotations

import numpy as np
import pytest

from _official_world import OfficialWorld, find_seed, goal_text
from tests.robomme_ood.unit.robomme import official_thresholds as T

DIFFS = ("easy", "medium", "hard")
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def drive_demo(ep):
    """Really advance each demo subtask; return (targets placed in time order during the demo, after which placement the button was inserted)."""
    env = ep.env
    cube = env.target_cube
    placed, button_after = [], None
    guard = 0
    while ep.task_index < ep.first_online_index():
        entry = env.task_list[ep.task_index]
        name = entry["name"]
        before = ep.task_index
        if name == "pick up the cube":
            ep.grasp(cube)
            ep.step()
        elif name == "drop the cube onto target":
            ep.place_on(cube, entry["segment"])
            ep.step()
            placed.append(entry["segment"])
        elif name == "drop the cube onto table":
            ep.place_on(cube, env.goal_site)
            ep.step()
        elif name == "press the button":
            ep.press(env.button)
            ep.step()
            ep.unpress(env.button)
            button_after = len(placed)
        else:  # static/NO RECORD: robot still, joints at the reset pose
            ep.step()
        guard += 1
        assert guard < 1000
        if name != "static" and name != "NO RECORD":
            assert ep.task_index == before + 1, f"demo subtask {name} did not advance"
    ep.step()  # past the final positioning of the swap at the start of the next step
    return placed, button_after


# --------------------------------------------------------------------------- VideoPlaceButton


@pytest.fixture
def vpb():
    with OfficialWorld("VideoPlaceButton") as w:
        yield w


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("lang", ["before", "after"])
def test_vpb_before_after_target(vpb, diff, lang):
    seed = find_seed("VideoPlaceButton", diff, lambda e: e.target_target_language == lang)
    ep = vpb.make(diff, seed=seed)
    env = ep.env
    assert all(lang in g for g in goal_text(env)[:2])
    placed, button_after = drive_demo(ep)
    expected = placed[button_after - 1] if lang == "before" else placed[button_after]
    assert env.target_target is expected
    ep.grasp(env.target_cube)
    ep.step()
    ep.place_on(env.target_cube, expected)
    ep.step()
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("lang", ["before", "after"])
def test_vpb_reversed_target_fails(vpb, diff, lang):
    seed = find_seed("VideoPlaceButton", diff, lambda e: e.target_target_language == lang)
    ep = vpb.make(diff, seed=seed)
    env = ep.env
    placed, button_after = drive_demo(ep)
    wrong = placed[button_after] if lang == "before" else placed[button_after - 1]
    ep.grasp(env.target_cube)
    ep.step()
    ep.place_on(env.target_cube, wrong)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", ("medium", "hard"))
def test_vpb_wrong_cube_fails(vpb, diff):
    ep = vpb.make(diff, seed=0)
    env = ep.env
    drive_demo(ep)
    ep.grasp(env.non_target_cubes[0])
    ep.step()
    assert ep.fail and not ep.success


def test_vpb_native_configs_have_no_extra_demo_placements(vpb):
    for diff in DIFFS:
        env = vpb.make(diff, seed=0).env
        assert (env.pre_flag, env.post_flag) == (0, 0)
        assert (env.swap_target_a is not None) is (diff == "hard")


def test_vpb_hard_swap_follows_identity():
    """hard: swapped target plates are judged by identity -- placing at its original position (now another plate) fails, placing at its current position succeeds."""
    seed = find_seed("VideoPlaceButton", "hard",
                     lambda e: e.target_target is e.swap_target_a or e.target_target is e.swap_target_b)
    for put_on_original, ok in ((True, False), (False, True)):
        with OfficialWorld("VideoPlaceButton") as world:
            ep = world.make("hard", seed=seed)
            env = ep.env
            origin = env.target_target.xyz[:2].copy()
            drive_demo(ep)
            assert np.linalg.norm(env.target_target.xyz[:2] - origin) > T.DROP_ONTO_XY  # the original position is no longer within its placement radius
            ep.grasp(env.target_cube)
            ep.step()
            if put_on_original:
                ep.carry(env.target_cube, *origin)
                ep.release(env.target_cube, *origin)
            else:
                ep.place_on(env.target_cube, env.target_target)
            ep.step()
            assert ep.success is ok and ep.fail is (not ok)


# --------------------------------------------------------------------------- VideoPlaceOrder


@pytest.fixture
def vpo():
    with OfficialWorld("VideoPlaceOrder") as w:
        yield w


def _vpo_case(world, diff, predicate):
    seed = find_seed("VideoPlaceOrder", diff, predicate)
    ep = world.make(diff, seed=seed)
    return ep, ep.env


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("which", ["first", "last"])
def test_vpo_kth_temporal_target_succeeds(vpo, diff, which):
    pred = (lambda e: e.which_in_subset == 1) if which == "first" else (
        lambda e: e.which_in_subset == len(e.which_targets_to_pick) and e.which_in_subset > 1)
    ep, env = _vpo_case(vpo, diff, pred)
    k = env.which_in_subset
    assert all(f"the {ORDINALS[k]} target" in g for g in goal_text(env))
    placed, _ = drive_demo(ep)
    assert env.target_target is placed[k - 1]
    ep.grasp(env.target_cube)
    ep.step()
    ep.place_on(env.target_cube, placed[k - 1])
    ep.step()
    assert ep.success and not ep.fail


def test_vpo_button_does_not_take_an_ordinal(vpo):
    """When the button is inserted before the k-th placement, the index still counts placements only."""
    ep, env = _vpo_case(vpo, "medium", lambda e: 0 < e.button_task_index // 2 < e.which_in_subset)
    placed, button_after = drive_demo(ep)
    assert 0 < button_after < env.which_in_subset
    assert env.target_target is placed[env.which_in_subset - 1]


def test_vpo_spatial_order_is_not_temporal_order(vpo):
    """When the k-th target by y coordinate != the target of the k-th placement in the demo, placing on the "spatial k-th" -> failure."""
    for seed in range(64):
        ep = vpo.make("medium", seed=seed)
        env = ep.env
        k = env.which_in_subset
        spatial = sorted(env.which_targets_to_pick, key=lambda t: float(t.xyz[1]))
        if spatial[k - 1] is not env.which_targets_to_pick[k - 1]:
            break
    else:
        pytest.fail("no layout found where spatial order differs from time order")
    placed, _ = drive_demo(ep)
    ep.grasp(env.target_cube)
    ep.step()
    ep.place_on(env.target_cube, spatial[k - 1])
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_vpo_other_target_fails(vpo, diff):
    ep = vpo.make(diff, seed=0)
    env = ep.env
    drive_demo(ep)
    other = next(t for t in env.targets if t is not env.target_target)
    ep.grasp(env.target_cube)
    ep.step()
    ep.place_on(env.target_cube, other)
    ep.step()
    assert ep.fail and not ep.success


def test_vpo_demo_drop_tasks_bind_their_own_target(vpo):
    """Each "place on target" subtask in the demo binds its own target (closure captures by value): placing on the last target first does not advance the first placement subtask."""
    ep, env = _vpo_case(vpo, "easy", lambda e: len(e.which_targets_to_pick) >= 2 and e.button_task_index != 0)
    ep.grasp(env.target_cube)
    ep.step()
    idx = ep.task_index
    ep.place_on(env.target_cube, env.which_targets_to_pick[-1])
    ep.step()
    assert ep.task_index == idx
    ep.place_on(env.target_cube, env.which_targets_to_pick[0])
    ep.step()
    assert ep.task_index == idx + 1
