"""ButtonUnmask 新值档（xhard1～xhard4）：与 VideoUnmask 同一套布局断言（容器、藏物、干扰物、抓取次数），
外加按钮；包内规格回放与自导出。"""
from __future__ import annotations

import pytest

from . import cells as C
from . import offline_scene as O
from . import unmask_common as U

TASK = "ButtonUnmask"


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
def test_containers_hidden_cubes_distractors_and_button(tier, k):
    U.check_unmask_layout(TASK, tier, k)
    _, env = C.replayed(TASK, tier, k)
    assert env.button is not None and env.task_list[0]["name"] == "press the button"
