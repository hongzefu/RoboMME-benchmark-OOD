"""ButtonUnmaskSwap 新值档（xhard1、xhard2）：内环交换次数／时间窗、藏物、干扰容器与外环交换均衡、两个按钮，
包内规格回放与自导出。

Swap 任务的离线建场较重（含交换路径可行性预判），每格只把第 1 行放进日常门禁，第 2、3 行标 slow。
"""
from __future__ import annotations

import pytest

from . import cells as C
from . import offline_scene as O
from . import swap_common as S

TASK = "ButtonUnmaskSwap"


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK, slow_from=1))
def test_packaged_spec_replays_with_zero_mismatch(task, tier, k):
    C.check_packaged_replay(task, tier, k)


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK, slow_from=1))
def test_offline_export_equals_package_and_replays(task, tier, k):
    C.check_self_export(task, tier, k)


@pytest.mark.parametrize("tier", O.tiers_of(TASK))
def test_tampered_spec_is_detected(tier):
    C.check_tamper_detected(TASK, tier)


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK, slow_from=1))
def test_swaps_hidden_cubes_distractors_and_two_buttons(task, tier, k):
    S.check_swap_layout(task, tier, k)
    _, env = C.replayed(task, tier, k)
    assert {env.button_left, env.button_right} == set(env.button_list)
    assert env.button_left is not env.button_right
