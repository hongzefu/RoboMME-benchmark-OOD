"""官方 rpy_util（两个包共用：robomme_hard 经 shim 借用同一模块）——ee 位姿的四元数与 RPY 连续性。

手算算例：绕 z 转 90° 的 wxyz = (cos45°, 0, 0, sin45°)；四元数 q 与 −q 表示同一旋转，按上一帧点积取号；
yaw 从 3.1 跨到 −3.1（实际只转了 0.083 rad）时连续化成 3.1832；非有限或零范数四元数回退成单位四元数。
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from robomme.robomme_env.utils import rpy_util as ru

S = math.sqrt(0.5)


@pytest.mark.parametrize("rpy, quat", [
    ((0.0, 0.0, 0.0), (1, 0, 0, 0)),
    ((0.0, 0.0, math.pi / 2), (S, 0, 0, S)),
    ((math.pi / 2, 0.0, 0.0), (S, S, 0, 0)),
    ((0.0, math.pi / 2, 0.0), (S, 0, S, 0)),
])
def test_rpy_to_quat_handcomputed_and_roundtrip(rpy, quat):
    q = ru.rpy_xyz_to_quat_wxyz_torch(torch.tensor(rpy, dtype=torch.float64))
    np.testing.assert_allclose(q.numpy(), quat, atol=1e-9)
    back = ru.quat_wxyz_to_rpy_xyz_torch(q)
    np.testing.assert_allclose(back.numpy(), rpy, atol=1e-9)


def test_batched_shapes_preserved():
    rpy = torch.zeros(3, 3, dtype=torch.float32)
    q = ru.rpy_xyz_to_quat_wxyz_torch(rpy)
    assert q.shape == (3, 4) and q.dtype == torch.float32


def test_sign_alignment():
    q = torch.tensor([[-S, 0, 0, -S]])
    prev = torch.tensor([[S, 0, 0, S]])
    np.testing.assert_allclose(ru.align_quat_sign_with_prev_torch(q, prev).numpy(), [[S, 0, 0, S]], atol=1e-7)
    assert ru.align_quat_sign_with_prev_torch(q, None) is q
    assert ru.align_quat_sign_with_prev_torch(q, torch.zeros(4)) is q  # 形状不符不对齐


def test_unwrap_across_pi():
    out = ru.unwrap_rpy_with_prev_torch(torch.tensor([0.0, 0.0, -3.1], dtype=torch.float64),
                                        torch.tensor([0.0, 0.0, 3.1], dtype=torch.float64))
    np.testing.assert_allclose(out.numpy(), [0, 0, 2 * math.pi - 3.1], atol=1e-12)
    same = torch.tensor([0.1, 0.2, 0.3])
    assert ru.unwrap_rpy_with_prev_torch(same, None) is same


@pytest.mark.parametrize("bad", [(0.0, 0, 0, 0), (math.nan, 0, 0, 1), (math.inf, 0, 0, 0)])
def test_normalize_falls_back_to_identity(bad):
    q = torch.tensor(bad, dtype=torch.float32)
    np.testing.assert_allclose(ru.normalize_quat_wxyz_torch(q).numpy(), [1, 0, 0, 0])


def test_normalize_scales_to_unit():
    np.testing.assert_allclose(ru.normalize_quat_wxyz_torch(torch.tensor([0.0, 0, 0, 2.0])).numpy(), [0, 0, 0, 1])


def test_pose_dict_pipeline_updates_cache_and_stays_continuous():
    p = torch.tensor([0.1, 0.2, 0.3])
    yaws = [3.0, 3.1, -3.1, -3.0]  # 主值在 ±π 处跳变
    prev_q = prev_rpy = None
    out = []
    for y in yaws:
        q = ru.rpy_xyz_to_quat_wxyz_torch(torch.tensor([0.0, 0.0, y], dtype=torch.float64))
        pose, prev_q, prev_rpy = ru.build_endeffector_pose_dict(p, q, prev_q, prev_rpy)
        assert pose["pose"] is p and set(pose) == {"pose", "quat", "rpy"}
        assert torch.equal(prev_rpy, pose["rpy"]) and prev_rpy is not pose["rpy"]  # 缓存是副本
        out.append(float(pose["rpy"][2]))
    np.testing.assert_allclose(np.diff(out), [0.1, 2 * math.pi - 6.2, 0.1], atol=1e-9)  # 每帧真实转角
