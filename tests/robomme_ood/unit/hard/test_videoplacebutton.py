"""VideoPlaceButton 新值档（xhard1、xhard2）：目标盘数、演示方块、按钮前后额外放置、答案绑定，包内规格回放与自导出。"""
from __future__ import annotations

import pytest

from . import cells as C
from . import offline_scene as O

TASK = "VideoPlaceButton"


def _decision(tier):
    header, _ = O.delivered_rows(TASK, tier, 0)
    return header["sampling_config"][TASK]["decision"]


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
def test_targets_demo_and_extra_places(tier, k):
    row, env = C.replayed(TASK, tier, k)
    dec = _decision(tier)
    sub = dec[tier]
    assert len(env.targets) == dec["targets"][tier]
    assert len(env.demo_cubes) == sub["demo_object_count"]
    assert len(env.demo_extra_place_before) == sub["extra_place_before"]
    assert len(env.demo_extra_place_after) == sub["extra_place_after"]
    for cube, target in env.demo_extra_place_before + env.demo_extra_place_after:
        assert cube in env.demo_cubes and target in env.targets
    # 演示放置序列：每步 (演示块序号, 目标序号) 合法，总数即 target_placement_count
    seq = row["spec"]["actions"]["place_sequence"]
    assert len(seq) == env.target_placement_count
    assert all(0 <= c < len(env.demo_cubes) and 0 <= t < len(env.targets) for c, t in seq)
    # 答案目标在目标盘里，「错误目标」恰是其余全部
    assert env.target_target in env.targets
    assert set(env.targets_not_true) == set(env.targets) - {env.target_target}
    assert env.target_target_language in ("before", "after")
    assert env.target_cube in env.demo_cubes
    assert set(env.non_target_cubes) == set(env.all_cubes) - {env.target_cube}
    if sub["demo_return_policy"] == "return_to_origin":
        assert len(env.xhard_home_sites) == len(env.demo_cubes)
