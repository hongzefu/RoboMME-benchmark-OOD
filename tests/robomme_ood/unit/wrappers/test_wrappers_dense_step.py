"""The four public planning entry points of planner_denseStep (C09.02): call order and arguments after the real planning boundary is replaced by a CPU spy.

``move_to_pose_with_RRTStar/move_to_pose_with_screw/close_gripper/open_gripper`` each call only the one corresponding planner method,
pass the target pose through unchanged, and collect the low-level steps meanwhile into a uniform batch; planning returns -1 -> the entry returns -1; afterwards the env.step interception is restored.
(Batch construction and concatenation details are in tests/robomme_ood/unit/robomme/test_step_batch.py; here only the dispatch and arguments of the four entries are tested.)
The hard package's planner_denseStep is a shim of the official module; the same tests check that both packages import the same implementation.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from tests.robomme_ood.unit.wrappers.wrappers_fakes import planner_spy_classes

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


@pytest.fixture(params=("robomme", "robomme_ood"))
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
    assert calls == [(method, pose if takes_pose else None)]   # only the corresponding method is called; the pose is the same object
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
    hard = importlib.import_module("robomme_ood.robomme_env.utils.planner_denseStep")
    assert hard is off
