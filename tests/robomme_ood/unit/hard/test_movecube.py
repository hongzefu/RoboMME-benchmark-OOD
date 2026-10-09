"""MoveCube new-value tiers (V9 delivers only xhard4): cube/goal/peg positions and move mode of both the demonstration and execution layouts, packaged spec replay and self-export.

Expectations: region parameters come from the ``region`` recorded in the spec (consistency with the header decision's region asserted separately); cube-to-goal distance and cube-to-
arm-base distance computed by hand from actual actor poses.
"""
from __future__ import annotations

import numpy as np
import pytest

from . import cells as C
from . import offline_scene as O

TASK = "MoveCube"


def _decision(tier):
    header, _ = O.delivered_rows(TASK, tier, 0)
    return header["sampling_config"][TASK]["decision"]


def _xy(actor):
    return actor.pose.p[0, :2].numpy().astype(np.float64)


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK))
def test_packaged_spec_replays_with_zero_mismatch(task, tier, k):
    C.check_packaged_replay(task, tier, k)


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK))
def test_offline_export_equals_package_and_replays(task, tier, k):
    C.check_self_export(task, tier, k)


@pytest.mark.parametrize("tier", O.tiers_of(TASK))
def test_tampered_spec_is_detected(tier):
    C.check_tamper_detected(TASK, tier)


@pytest.mark.parametrize("tier", O.tiers_of(TASK))
@pytest.mark.parametrize("k", range(C.REPLAY_ROWS))
def test_two_layouts_regions_and_way(tier, k):
    row, env = C.replayed(TASK, tier, k)
    dec = _decision(tier)
    layout = row["spec"]["layout"]
    # the execution-segment cube is parked off-scene during the demonstration; checked against the execution-segment initial poses recorded by production (cube_init_pose_2/goal_site_2_pose_p)
    pairs = {"demo": (_xy(env.cube), _xy(env.goal_site)),
             "execution": (env.cube_init_pose_2.p[0, :2].numpy().astype(np.float64),
                           np.asarray(env.goal_site_2_pose_p, dtype=np.float64).reshape(-1)[:2])}
    for seg, (cube, goal) in pairs.items():
        region = layout[seg]["region"]
        declared = dec[f"{seg}_layout"][tier]["region"]
        # region recorded in the spec ⊇ region declared in the header (every declared key appears unchanged)
        assert all(region[key] == value for key, value in declared.items())
        # actual actor poses are exactly the cube and goal positions frozen in the spec
        assert np.allclose(cube, layout[seg]["cube_pose"][:2], atol=1e-6)
        assert np.allclose(goal, layout[seg]["goal_xy"], atol=1e-6)
        # by hand: cube-to-goal distance not below the lower bound; cube-to-arm-base distance within the base distance range
        assert np.linalg.norm(cube - goal) >= region["min_cube_goal_m"] - 1e-6
        base = np.asarray(region["robot_base_xy"], dtype=np.float64)
        lo, hi = region["base_dist"]
        assert lo - 1e-6 <= np.linalg.norm(cube - base) <= hi + 1e-6
    # move mode: one of three, taken from production's mode table
    assert env.way in env.ways and len(set(env.ways)) == 3
