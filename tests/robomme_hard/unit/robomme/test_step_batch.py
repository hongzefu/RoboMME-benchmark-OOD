"""官方 planner_denseStep：逐步收集与批的构造／拼接（C09；决定 obs 的 dict-of-lists 形状）。

统一批契约：(obs: dict[str, list], reward: float32[N], terminated: bool[N], truncated: bool[N], info: dict[str, list])。
期望全部由手写小例子得出。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from robomme.robomme_env.utils import planner_denseStep as pds


def _step(i, extra_obs=None, extra_info=None, term=False):
    obs = {"x": np.array([i, i]), **(extra_obs or {})}
    info = {"k": i, **(extra_info or {})}
    return obs, torch.tensor([float(i)]), torch.tensor([term]), torch.tensor([False]), info


def test_empty_batch_contract():
    obs, r, t, tr, info = pds.to_step_batch([])
    assert obs == {} and info == {}
    assert r.shape == (0,) and r.dtype == torch.float32
    assert t.dtype == torch.bool and tr.dtype == torch.bool and t.numel() == 0


def test_to_step_batch_columnar_with_none_fill():
    steps = [_step(0), _step(1, extra_obs={"y": 5}, extra_info={"only1": "a"}), _step(2, term=True)]
    obs, r, t, tr, info = pds.to_step_batch(steps)
    assert set(obs) == {"x", "y"} and obs["y"] == [None, 5, None]
    assert info["only1"] == [None, "a", None] and info["k"] == [0, 1, 2]
    assert r.tolist() == [0.0, 1.0, 2.0] and t.tolist() == [False, False, True]
    assert all(len(v) == 3 for v in obs.values())


def test_singleton_lists_collapse_except_task_goal():
    steps = [({"a": [[7]]}, 0.0, False, False, {"task_goal": ["only goal"], "b": [3]})]
    obs, _r, _t, _tr, info = pds.to_step_batch(steps)
    assert obs["a"] == [7]
    assert info["b"] == [3]
    assert info["task_goal"] == [["only goal"]]  # task_goal 保持列表，不被拆成字符串


def test_scalars_from_tensors_and_python_values():
    steps = [({}, torch.tensor([[1.5]]), torch.tensor([True]), False, {}), ({}, 2, 0, torch.tensor([]), {})]
    _o, r, t, tr, _i = pds.to_step_batch(steps)
    assert r.tolist() == [1.5, 2.0] and t.tolist() == [True, False] and tr.tolist() == [False, False]


def test_concat_skips_empty_and_unions_keys():
    a = pds.to_step_batch([_step(0)])
    b = pds.to_step_batch([_step(1, extra_obs={"y": 1}), _step(2, extra_obs={"y": 2})])
    out = pds.concat_step_batches([pds.empty_step_batch(), a, None, b])
    obs, r, t, tr, info = out
    assert r.tolist() == [0.0, 1.0, 2.0]
    assert obs["y"] == [None, 1, 2] and [int(v[0]) for v in obs["x"]] == [0, 1, 2]
    assert pds.concat_step_batches([pds.empty_step_batch()])[1].numel() == 0


class _Env:
    def __init__(self):
        self.calls = 0
        self.buf = np.zeros(2)

    def step(self, action):
        self.calls += 1
        self.buf[:] = self.calls  # 复用同一块内存
        return {"x": self.buf}, torch.tensor([0.0]), torch.tensor([False]), torch.tensor([False]), {"n": self.calls}


def test_collect_dense_steps_intercepts_and_restores():
    env = _Env()
    planner = type("P", (), {})()
    planner.env = env
    original = env.step

    def fn():
        for _ in range(3):
            planner.env.step(None)
        return 0

    steps = pds._collect_dense_steps(planner, fn)
    assert len(steps) == 3 and planner.env.step == original
    assert [int(s[0]["x"][0]) for s in steps] == [1, 2, 3]  # 快照：不被后续步覆盖


def test_collect_dense_steps_minus_one_and_exception_restore():
    env = _Env()
    planner = type("P", (), {})()
    planner.env = env
    original = env.step
    assert pds._collect_dense_steps(planner, lambda: -1) == -1
    assert pds._run_with_dense_collection(planner, lambda: -1) == -1

    def boom():
        planner.env.step(None)
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        pds._collect_dense_steps(planner, boom)
    assert planner.env.step == original


def test_batch_output_is_split_into_steps_when_collected():
    """env.step 若返回整批（如 DemonstrationWrapper._step_batch 的形状），收集时拆回逐步。"""
    batch = pds.to_step_batch([_step(0), _step(1)])
    env = type("E", (), {"step": lambda self, a: batch})()
    planner = type("P", (), {})()
    planner.env = env
    steps = pds._collect_dense_steps(planner, lambda: planner.env.step(None))
    assert len(steps) == 2 and [s[4]["k"] for s in steps] == [0, 1]
