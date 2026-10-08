"""VideoPlaceOrder 新值档（xhard1、xhard2）：演示方块数、每块访问次数、「第 k 次放置」答案绑定，包内规格回放与自导出。"""
from __future__ import annotations

import pytest

from . import cells as C
from . import offline_scene as O

TASK = "VideoPlaceOrder"


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
def test_demo_visits_and_kth_answer(tier, k):
    row, env = C.replayed(TASK, tier, k)
    dec = _decision(tier)
    assert len(env.targets) == dec["targets"][tier]
    sub = dec[tier]
    assert len(env.demo_cubes) == sub["demo_object_count"]
    visits = env.demo_visit_targets
    assert sorted(len(v) for v in visits) == sorted(sub["visit_counts"])
    for seq in visits:
        assert all(t in env.targets for t in seq)
        assert all(a is not b for a, b in zip(seq, seq[1:])), "同一方块连续两次放同一目标"
    # 答案：被问方块的访问序列里第 which_in_subset 次放置的目标（时间序，不是空间序）
    # （两块的访问序列可能相同，所以按规格记录的被问方块序号取，不按内容反查）
    ans = row["spec"]["objects"]["answer_demo_index"]
    assert env.target_cube is env.demo_cubes[ans]
    assert env.which_targets_to_pick == visits[ans]
    assert 1 <= env.which_in_subset <= len(env.which_targets_to_pick)
    assert env.target_target is env.which_targets_to_pick[env.which_in_subset - 1]
    assert set(env.targets_not_true) == set(env.targets) - {env.target_target}
    # 演示总放置次数 = 各块访问次数之和；放回原位策略下每个演示块各有一个落点
    assert env.target_placement_count == sum(len(v) for v in visits)
    if sub["demo_return_policy"] == "return_to_origin":
        assert len(env.xhard_home_sites) == len(env.demo_cubes)
