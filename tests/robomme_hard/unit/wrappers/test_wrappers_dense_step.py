"""planner_denseStep 的四个对外规划入口（C09.02）：真实规划边界换成 CPU spy 后的调用次序与参数。

``move_to_pose_with_RRTStar／move_to_pose_with_screw／close_gripper／open_gripper`` 各自只调用规划器上对应的那一个方法、
原样传入目标位姿，把期间的底层 step 收成统一批；规划返回 -1 → 入口返回 -1；结束后 env.step 拦截被还原。
（批的构造与拼接细节已在 tests/robomme_hard/unit/robomme/test_step_batch.py，这里只测四个入口的分派与参数。）
hard 包的 planner_denseStep 是官方模块的 shim，用同一组用例核对两包导入到的是同一实现。
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from tests.robomme_hard.unit.wrappers.wrappers_fakes import planner_spy_classes

ENTRIES = {
    "move_to_pose_with_RRTStar": ("planner.rrt", True),
    "move_to_pose_with_screw": ("planner.screw", True),
    "close_gripper": ("planner.close", False),
    "open_gripper": ("planner.open", False),
}
SCRIPT_KEY = {"planner.rrt": "rrt", "planner.screw": "screw", "planner.close": "close", "planner.open": "open"}


class _CountingEnv:
    def __init__(self, log):
        self.log = log
        self.n = 0

    def step(self, action):
        self.n += 1
        self.log.append(("env.step", self.n, float(np.asarray(action).reshape(-1)[0])))
        obs = {"k": np.array([self.n, self.n])}
        return obs, torch.tensor([float(self.n)]), torch.tensor([False]), torch.tensor([False]), {"n": self.n}


@pytest.fixture(params=("robomme", "robomme_hard"))
def pds(request):
    return importlib.import_module(f"{request.param}.robomme_env.utils.planner_denseStep")


@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_entry_dispatches_to_matching_method(pds, entry):
    log = []
    method, takes_pose = ENTRIES[entry]
    arm, _ = planner_spy_classes(log, {SCRIPT_KEY[method]: [3]})
    env = _CountingEnv(log)
    planner = arm(env)
    original = planner.env.step
    pose = object()
    fn = getattr(pds, entry)
    batch = fn(planner, pose) if takes_pose else fn(planner)
    calls = [e for e in log if e[0].startswith("planner.") and e[0] != "planner.new"]
    assert calls == [(method, pose if takes_pose else None)]   # 只调对应方法，位姿是同一对象
    obs, reward, terminated, truncated, info = batch
    assert reward.tolist() == [1.0, 2.0, 3.0] and info["n"] == [1, 2, 3]
    assert [int(v[0]) for v in obs["k"]] == [1, 2, 3]
    assert planner.env.step == original


@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_entry_returns_minus_one_on_failure(pds, entry):
    log = []
    method, takes_pose = ENTRIES[entry]
    arm, _ = planner_spy_classes(log, {SCRIPT_KEY[method]: [-1]})
    env = _CountingEnv(log)
    planner = arm(env)
    original = planner.env.step
    fn = getattr(pds, entry)
    assert (fn(planner, "P") if takes_pose else fn(planner)) == -1
    assert env.n == 0 and planner.env.step == original


def test_hard_planner_dense_step_is_official_module():
    off = importlib.import_module("robomme.robomme_env.utils.planner_denseStep")
    hard = importlib.import_module("robomme_hard.robomme_env.utils.planner_denseStep")
    assert hard is off
