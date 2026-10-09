"""Unified distractor-container sampler for the Unmask family (``utils/unmask_distractor_sampler.py``): config validation, point-to-OBB criterion,
re-checking sampled results by the same rules, replay-injection re-check rejecting a corrupted frozen layout, park points and the "lift away -- put back" timeline.

Expectations come from properties and small hand-written tables: sampled results must pass the independent rule re-check (``verify_distractor_layout`` is only the other half under test;
annulus and spacing are also computed by hand); the park-point grid is checked against hand-written coordinates.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import sapien
import torch

from robomme_ood.robomme_env.utils import unmask_distractor_sampler as S
from robomme_ood.robomme_env.utils.episode_spec import SpecRecorder
from robomme_ood.robomme_env.utils.SceneGenerationError import SceneGenerationError

from . import offline_scene as O

H = 0.02  # cube half extent (same value as the task; used only as a geometric scale)


def _cfg(**over):
    """Use the real distractor config of in-package VideoUnmask xhard2 as the baseline."""
    header, _ = O.delivered_rows("VideoUnmask", "xhard2", 0)
    cfg = copy.deepcopy(header["sampling_config"]["VideoUnmask"]["decision"]["xhard2"]["distractor"])
    cfg.update(over)
    return cfg


def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── parse_distractor_cfg ──────────────────────────────────────────────────────


def test_parse_accepts_packaged_config():
    c = S.parse_distractor_cfg(_cfg())
    assert c.count == _cfg()["count"] and c.ring == tuple(_cfg()["ring_max_abs_xy"])


@pytest.mark.parametrize("over", [
    {"count": -1},
    {"ring_max_abs_xy": [0.3, 0.2]},
    {"cube_count_range": [3, 99]},
    {"color_pool": ["red"]},
    {"color_rule": "random"},
    {"min_gap_factor": -0.1},
    {"max_trials": 0},
])
def test_parse_rejects_bad_values(over):
    with pytest.raises(ValueError):
        S.parse_distractor_cfg(_cfg(**over))


def test_parse_rejects_missing_or_extra_keys():
    cfg = _cfg()
    del cfg["max_trials"]
    with pytest.raises(ValueError):
        S.parse_distractor_cfg(cfg)
    with pytest.raises(ValueError):
        S.parse_distractor_cfg(_cfg(extra=1))


# -- geometry ------------------------------------------------------------------------


def test_point_hits_obbs_strict_reach():
    # sizes are powers of 2 so distances are exactly representable: point to box edge is exactly 0.125
    box = (np.zeros(2), np.eye(2), np.array([0.25, 0.25]))
    assert S.point_hits_obbs(np.array([0.375, 0.0]), [box], 0.125 + 1e-9) is True
    assert S.point_hits_obbs(np.array([0.375, 0.0]), [box], 0.125) is False  # exactly equal to reach: strict < does not hold
    assert S.point_hits_obbs(np.array([0.5, 0.5]), [box], 0.35) is False  # to the corner √2·0.25≈0.3536
    assert S.point_hits_obbs(np.array([0.5, 0.5]), [box], 0.36) is True


def test_bin_obb2d_is_axis_square_rotated_by_yaw():
    c, A, h = S.bin_obb2d(0.1, 0.2, 30.0, H)
    assert np.allclose(c, [0.1, 0.2]) and np.allclose(A.T @ A, np.eye(2), atol=1e-9)
    assert np.allclose(h, [S.bin_outer_half(H)] * 2)
    angle = np.degrees(np.arctan2(A[1, 0], A[0, 0])) % 90.0
    assert angle == pytest.approx(30.0, abs=1e-6) or angle == pytest.approx(60.0, abs=1e-6)


# -- sampling and re-check ----------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_sampled_layout_obeys_ring_spacing_and_counts(seed):
    cfg = _cfg()
    lay = S.sample_distractor_layout(cfg, obstacles=[], generator=_gen(seed), cube_half_size=H)
    assert lay.count == cfg["count"]
    lo, hi = cfg["ring_max_abs_xy"]
    for x, y, yaw in lay.bins:
        m = max(abs(x), abs(y))
        assert lo - 1e-9 <= m <= hi + 1e-9, "inside the annulus (measured by the max absolute coordinate)"
        assert 0.0 <= yaw <= 90.0
    # hand-computed spacing: pairwise container-center distance is at least two outer half extents (necessary for no overlap in top view)
    half = S.bin_outer_half(H)
    for i in range(lay.count):
        for j in range(i + 1, lay.count):
            assert np.hypot(lay.bins[i][0] - lay.bins[j][0], lay.bins[i][1] - lay.bins[j][1]) >= 2 * half - 1e-9
    clo, chi = cfg["cube_count_range"]
    assert clo <= lay.cube_count <= chi and len(set(lay.cube_bins)) == lay.cube_count
    assert sorted(lay.color_order) == [0, 1, 2]
    assert S.verify_distractor_layout(lay, cfg, obstacles=[], cube_half_size=H) == []


def test_verify_flags_tampered_layout():
    cfg = _cfg()
    lay = S.sample_distractor_layout(cfg, obstacles=[], generator=_gen(1), cube_half_size=H)
    bad = copy.deepcopy(lay)
    bad.bins[1] = bad.bins[0]  # two containers stacked on each other
    bad.cube_bins = [0, 0]
    problems = S.verify_distractor_layout(bad, cfg, obstacles=[], cube_half_size=H)
    assert any("insufficient spacing" in p for p in problems) and any("cube_bins" in p for p in problems)


def test_sampling_exhaustion_raises_with_placed_count():
    blocker = (np.zeros(2), np.eye(2), np.array([1.0, 1.0]))  # the whole table is an obstacle
    with pytest.raises(S.DistractorPlacementError) as err:
        S.sample_distractor_layout(_cfg(max_trials=8), obstacles=[blocker], generator=_gen(0), cube_half_size=H)
    assert err.value.placed == 0


def test_commit_replay_rejects_frozen_layout_violating_rules():
    cfg = _cfg()
    lay = S.sample_distractor_layout(cfg, obstacles=[], generator=_gen(2), cube_half_size=H)
    exp = SpecRecorder(None, "VideoUnmask", difficulty="xhard2")
    S.commit_distractor_layout(lay, cfg=cfg, recorder=exp, obstacles=[], cube_half_size=H)
    spec = exp.to_dict()
    ok = SpecRecorder(copy.deepcopy(spec), "VideoUnmask", difficulty="xhard2")
    assert S.commit_distractor_layout(lay, cfg=cfg, recorder=ok, obstacles=[], cube_half_size=H).same_geometry(lay)
    assert ok.mismatches == []
    spec["objects"]["distractors"]["bins"]["1"] = list(spec["objects"]["distractors"]["bins"]["0"])
    bad = SpecRecorder(spec, "VideoUnmask", difficulty="xhard2")
    with pytest.raises(SceneGenerationError):
        S.commit_distractor_layout(lay, cfg=cfg, recorder=bad, obstacles=[], cube_half_size=H)


# -- park points and lift away -- put back ---------------------------------------------------------


def test_park_points_are_disjoint_grid_per_group():
    pts = {(g, i): tuple(S.xhard_park_point(g, i)) for g in S.XHARD_PARK_GROUPS for i in (0, 15, 16, 63)}
    assert len(set(pts.values())) == len(pts)
    x0, y0, z0 = S.XHARD_PARK_ORIGIN
    pitch = S.XHARD_PARK_PITCH_M
    assert pts[(S.XHARD_PARK_GROUPS[0], 0)] == (x0, y0, z0)
    assert pts[(S.XHARD_PARK_GROUPS[0], 16)] == (x0, y0 + pitch, z0)  # second row
    with pytest.raises(ValueError):
        S.xhard_park_point("nope", 0)
    with pytest.raises(ValueError):
        S.xhard_park_point(S.XHARD_PARK_GROUPS[0], S.XHARD_PARK_GROUP_CAPACITY)


def test_lift_and_park_timeline():
    """Window [10, 20): record the original position on first entry, put back at step = 10 + (20-10)//2 = 15, park at the park point on other steps; no motion outside the window."""
    actor = O.FakeActor("bin", sapien.Pose(p=[0.1, 0.2, 0.05]), "dynamic", [])
    env = type("E", (), {})()
    park = S.xhard_park_point("bin", 3)
    pos = {}
    for t in range(8, 23):
        S.lift_and_park_back_to_original(env, actor, 10, 20, t, park)
        pos[t] = actor.pose.p[0].numpy().round(6).tolist()
    origin = [0.1, 0.2, 0.05]
    assert pos[8] == pytest.approx(origin) and pos[9] == pytest.approx(origin)
    for t in range(10, 15):
        assert pos[t] == pytest.approx(park.tolist())
    for t in range(15, 23):
        assert pos[t] == pytest.approx(origin, abs=1e-6)
