"""Unmask 系干扰容器统一采样器（``utils/unmask_distractor_sampler.py``）：配置校验、点到 OBB 判据、
采样结果按同一规则复核、回注模式复核拒绝改坏的冻结布局、停放点与「抬走—放回」时间线。

期望来自性质与手写小表：采样结果必须过独立的规则复核（``verify_distractor_layout`` 只作被测的另一半，
另外手算环带与间距）；停放点网格按手写坐标核对。
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import sapien
import torch

from robomme_hard.robomme_env.utils import unmask_distractor_sampler as S
from robomme_hard.robomme_env.utils.episode_spec import SpecRecorder
from robomme_hard.robomme_env.utils.SceneGenerationError import SceneGenerationError

from . import offline_scene as O

H = 0.02  # 方块半边长（与任务同值，只作几何尺度）


def _cfg(**over):
    """取包内 VideoUnmask xhard2 的真实干扰配置做基准。"""
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


# ── 几何 ────────────────────────────────────────────────────────────────────────


def test_point_hits_obbs_strict_reach():
    # 尺寸取 2 的幂，距离可逐位表示：点到盒边恰 0.125
    box = (np.zeros(2), np.eye(2), np.array([0.25, 0.25]))
    assert S.point_hits_obbs(np.array([0.375, 0.0]), [box], 0.125 + 1e-9) is True
    assert S.point_hits_obbs(np.array([0.375, 0.0]), [box], 0.125) is False  # 恰等于 reach：严格 < 不成立
    assert S.point_hits_obbs(np.array([0.5, 0.5]), [box], 0.35) is False  # 到角点 √2·0.25≈0.3536
    assert S.point_hits_obbs(np.array([0.5, 0.5]), [box], 0.36) is True


def test_bin_obb2d_is_axis_square_rotated_by_yaw():
    c, A, h = S.bin_obb2d(0.1, 0.2, 30.0, H)
    assert np.allclose(c, [0.1, 0.2]) and np.allclose(A.T @ A, np.eye(2), atol=1e-9)
    assert np.allclose(h, [S.bin_outer_half(H)] * 2)
    angle = np.degrees(np.arctan2(A[1, 0], A[0, 0])) % 90.0
    assert angle == pytest.approx(30.0, abs=1e-6) or angle == pytest.approx(60.0, abs=1e-6)


# ── 采样与复核 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(4))
def test_sampled_layout_obeys_ring_spacing_and_counts(seed):
    cfg = _cfg()
    lay = S.sample_distractor_layout(cfg, obstacles=[], generator=_gen(seed), cube_half_size=H)
    assert lay.count == cfg["count"]
    lo, hi = cfg["ring_max_abs_xy"]
    for x, y, yaw in lay.bins:
        m = max(abs(x), abs(y))
        assert lo - 1e-9 <= m <= hi + 1e-9, "落在环带内（以最大坐标绝对值计）"
        assert 0.0 <= yaw <= 90.0
    # 手算间距：容器中心两两距离不小于两个外廓半边（俯视不相交的必要条件）
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
    bad.bins[1] = bad.bins[0]  # 两个容器叠在一起
    bad.cube_bins = [0, 0]
    problems = S.verify_distractor_layout(bad, cfg, obstacles=[], cube_half_size=H)
    assert any("间距不足" in p for p in problems) and any("cube_bins" in p for p in problems)


def test_sampling_exhaustion_raises_with_placed_count():
    blocker = (np.zeros(2), np.eye(2), np.array([1.0, 1.0]))  # 整个桌面都是障碍
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


# ── 停放点与抬走—放回 ─────────────────────────────────────────────────────────


def test_park_points_are_disjoint_grid_per_group():
    pts = {(g, i): tuple(S.xhard_park_point(g, i)) for g in S.XHARD_PARK_GROUPS for i in (0, 15, 16, 63)}
    assert len(set(pts.values())) == len(pts)
    x0, y0, z0 = S.XHARD_PARK_ORIGIN
    pitch = S.XHARD_PARK_PITCH_M
    assert pts[(S.XHARD_PARK_GROUPS[0], 0)] == (x0, y0, z0)
    assert pts[(S.XHARD_PARK_GROUPS[0], 16)] == (x0, y0 + pitch, z0)  # 第二行
    with pytest.raises(ValueError):
        S.xhard_park_point("nope", 0)
    with pytest.raises(ValueError):
        S.xhard_park_point(S.XHARD_PARK_GROUPS[0], S.XHARD_PARK_GROUP_CAPACITY)


def test_lift_and_park_timeline():
    """窗口 [10, 20)：首次进入记原位，落回步 = 10 + (20−10)//2 = 15 放回原位，其余步停在停放点；窗口外不动。"""
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
