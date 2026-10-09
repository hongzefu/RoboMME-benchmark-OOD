"""InsertPeg new-value tiers (V9 delivers only xhard4): peg count, peg positions and gaps, grasp end and insertion end binding from offline ``_load_scene`` + two ``_initialize_episode``,
packaged spec replay and self-export; hand-computed examples of the peg outline gap ``footprint_gap``."""
from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from . import cells as C
from . import offline_scene as O

TASK = "InsertPeg"
IP = importlib.import_module("robomme_ood.robomme_env.InsertPeg")


def _decision(tier):
    header, _ = O.delivered_rows(TASK, tier, 0)
    return header["sampling_config"][TASK]["decision"][tier]


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
def test_pegs_count_gaps_and_target_binding(tier, k):
    row, env = C.replayed(TASK, tier, k)
    dec = _decision(tier)
    assert len(env.pegs) == dec["peg_count"] == len(env.peg_heads) == len(env.peg_tails)
    init = row["spec"]["initializations"][str(1)]  # evaluation uses the second initialization
    # measured minimum gap (production record) must be strictly greater than this tier's lower bound (the rule is a strict inequality)
    assert init["min_pair_gap_m"] > dec["peg_min_pair_gap_m"]
    assert init["min_box_gap_m"] > dec["peg_box_min_gap_m"]
    # peg root x does not exceed this tier's upper bound; peg position is the frozen value (root moved to the spec position by set_pose)
    for i, peg in enumerate(env.pegs):
        (x, y), _yaw = init["pegs"][str(i)]
        assert x <= dec["peg_x_max_m"] + 1e-9
    # the grasped target is always the first peg; grasp end and insertion end are the two ends of the same peg
    assert env.peg is env.pegs[0]
    assert {env.grasp_target, env.insert_target} == {env.peg_head, env.peg_tail}
    assert env.grasp_target is not env.insert_target
    assert env.insert_way in ("left", "right")
    assert env.grasp_target_distance in ("near", "far")


# ── footprint_gap hand-computed examples (independent geometry: gap between two parallel axis-aligned rectangles) ────────────────


def test_footprint_gap_parallel_rectangles():
    # two pegs along x, length 0.1, half width 0.01, roots y=0.05 apart: gap between outlines = 0.05 − 2×0.01 = 0.03
    a = IP.peg_footprint(np.array([0.0, 0.0]), 0.0, 0.1, 0.01)
    b = IP.peg_footprint(np.array([0.0, 0.05]), 0.0, 0.1, 0.01)
    assert math.isclose(IP.footprint_gap(a, b), 0.03, abs_tol=1e-9)
    assert math.isclose(IP.footprint_gap(b, a), 0.03, abs_tol=1e-9)


def test_footprint_gap_overlap_and_far_apart():
    a = IP.peg_footprint(np.array([0.0, 0.0]), 0.0, 0.1, 0.01)
    same = IP.peg_footprint(np.array([0.0, 0.0]), 0.0, 0.1, 0.01)
    assert IP.footprint_gap(a, same) <= 0.0
    far = IP.peg_footprint(np.array([1.0, 0.0]), 0.0, 0.1, 0.01)
    assert IP.footprint_gap(a, far) > 0.5
