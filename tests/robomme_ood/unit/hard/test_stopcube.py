"""StopCube new-value tiers (xhard1-xhard5): stop index, motion cadence and round-trip route from offline ``_load_scene`` + two ``_initialize_episode``,
packaged spec replay and self-export."""
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
    # the stop window is exactly segment stop_time (the n-th pass over the target happens in segment n): [(n−1)·T, n·T]
    t, n = env.move_interval, env.stop_time
    assert tuple(env.stop_time_range) == (t * (n - 1), t * n)
    assert (n - 1) * t < env.steps_press < n * t
    # the number of round-trip segments is enough to cover the n-th pass
    assert env.motion_segments >= n
    # route: start and end points symmetric about the target center, the cube departs from the start point
    tgt = env.target.pose.p[0, :2].numpy().astype(np.float64)
    assert np.allclose((np.asarray(env.start_pos_xy) + np.asarray(env.end_pos_xy)) / 2, tgt, atol=1e-6)
    assert np.allclose(env.cube.pose.p[0, :2].numpy(), env.start_pos_xy, atol=1e-6)
    # the last item of the task table is "button stops the cube"; all before it are preparation and static checkpoints
    assert env.task_list[-1]["name"] == "press the button to stop the cube on the target"
