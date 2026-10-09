"""C14 ``Policy`` 接口：基类必须由参赛者实现；示例 ``DummyPolicy`` 的输出形状与首步记账。"""
from __future__ import annotations

import numpy as np
import pytest

from challenge_interface.policy import DummyPolicy, Policy
from challenge_interface.scripts.phase1_eval import EXPECTED_ACTION_SHAPES


def _obs(n_frames: int, first: bool) -> dict:
    return {
        "task_goal": ["目标"],
        "is_first_step": first,
        "front_rgb_list": [np.zeros((2, 2, 3), dtype=np.uint8)] * n_frames,
    }


def test_base_policy_methods_are_abstract():
    p = Policy()
    with pytest.raises(NotImplementedError):
        p.infer({})
    with pytest.raises(NotImplementedError):
        p.reset()


def test_dummy_policy_is_a_policy_and_outputs_joint_angle_chunk():
    p = DummyPolicy()
    assert isinstance(p, Policy)
    out = p.infer(_obs(3, first=True))
    assert set(out) == {"actions"}
    actions = out["actions"]
    # 块长读 DummyPolicy 的实际属性，每步动作形状读评估端的生产常量（关节角空间）。
    assert actions.shape == (p.chunk_size, *EXPECTED_ACTION_SHAPES["joint_angle"])
    assert p.chunk_size > 0
    # 夹爪维不加噪声，必须恰好是 1.0。
    assert np.all(actions[:, -1] == 1.0)
    assert np.all(np.isfinite(actions))


def test_dummy_policy_first_step_records_exec_start_idx_and_reset_clears_it():
    p = DummyPolicy()
    p.infer(_obs(4, first=True))
    # 首步时前 3 帧是条件视频，第 4 帧（下标 3）是当前执行帧。
    assert p.exec_start_idx == 3
    # 非首步不改动 exec_start_idx。
    p.infer(_obs(9, first=False))
    assert p.exec_start_idx == 3
    p.reset()
    assert p.exec_start_idx == 0


def test_dummy_policy_requires_is_first_step_key():
    with pytest.raises(KeyError):
        DummyPolicy().infer({"front_rgb_list": []})
