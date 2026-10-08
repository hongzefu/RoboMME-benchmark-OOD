"""StopCube 新值档（xhard1～xhard5）：离线 ``_load_scene`` + 两次 ``_initialize_episode`` 的停止序号、运动节拍、
往返路线，包内规格回放与自导出。"""
from __future__ import annotations

import numpy as np
import pytest

from . import cells as C
from . import offline_scene as O

TASK = "StopCube"


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
def test_stop_index_rhythm_and_route(tier, k):
    _, env = C.replayed(TASK, tier, k)
    dec = _decision(tier)
    rng = dec["stop_time_range"]
    assert rng["low"] <= env.stop_time < rng["high_exclusive"]
    assert env.move_interval in dec["move_interval_choices"]
    # 停止窗口恰是第 stop_time 段（第 n 次经过目标发生在第 n 段）：[(n−1)·T, n·T]
    t, n = env.move_interval, env.stop_time
    assert tuple(env.stop_time_range) == (t * (n - 1), t * n)
    assert (n - 1) * t < env.steps_press < n * t
    # 往返段数够覆盖第 n 次经过
    assert env.motion_segments >= n
    # 路线：起终点关于目标中心对称，方块从起点出发
    tgt = env.target.pose.p[0, :2].numpy().astype(np.float64)
    assert np.allclose((np.asarray(env.start_pos_xy) + np.asarray(env.end_pos_xy)) / 2, tgt, atol=1e-6)
    assert np.allclose(env.cube.pose.p[0, :2].numpy(), env.start_pos_xy, atol=1e-6)
    # 任务表最后一项是「按钮停方块」，此前全是准备与静止检查点
    assert env.task_list[-1]["name"] == "press the button to stop the cube on the target"
