"""L4 simulation smoke: one real ``make_env_for_episode`` + ``reset`` per task × tier, no step.

Scale: ``xhard0 16 tasks × 1 episode + xhard1-5 43 cells × 1 episode = 59`` resets; together with
the 1 in ``test_official_one_reset.py``, each run of ``tests/robomme_ood/sim`` does ``59 + 1 = 60`` resets and 0 trajectory generations.
These 60 resets are covered by the user's standing authorization (plan ``1003-code-test-maintenance-todo.md`` Q8) and are not requested each time.

Run only like this (pick an idle GPU with ``nvidia-smi`` first)::

    CUDA_VISIBLE_DEVICES=<idle GPU> uv run --no-sync python -m pytest tests/robomme_ood/sim --allow-sim-reset -q

Without ``--allow-sim-reset`` the resource guard rejects with UsageError (see ``tests/robomme_ood/_support/resource_policy.py``).

Cell enumeration is the same as the pre-cleanup scan ``docs/validation/test-redesign-20261003/records/reset_sweep.py``:
iterate ``BenchmarkEnvBuilder.get_task_list()``; xhard0 takes the first episode of the hard subset of the official ``test`` set
(``hard_specs.XHARD0_EPISODES[0]``, i.e. episode 0 of ``dataset="hard-verify"``; this repo's builder no longer accepts ``dataset="test"``);
new-value tiers take the first episode of each tier in ``dataset="ood"`` episode order.
Collection only reads metadata/specs on CPU, does not build scenes or initialize the GPU; runs sequentially in one process, ``env.close()`` after each cell.

Per-cell assertions (each checked against the scan's measured record ``reset-sweep.jsonl``, all 59 cells ok):
- the wrapper chain is exactly ``FailAwareWrapper → DemonstrationWrapper → TimeLimitWrapper → OrderEnforcing → <task class>``,
  layer 0 ``FailAwareWrapper`` is imported via ``robomme_ood.env_record_wrapper.FailAwareWrapper``, but that is a shim:
  the module alias points to official ``robomme.env_record_wrapper.FailAwareWrapper``, so the class object is the official class; layer 1
  ``DemonstrationWrapper`` is the class of ``robomme_ood``'s own copy; the task class is ``robomme_ood``'s class;
- obs has exactly five keys with fixed shapes and dtypes, the five lists have equal length; ``gripper_state_list`` is float32 for arm tasks,
  and float64 all-zeros for the two ``panda_stick`` tasks (PatternLock, RouteStick) (production code
  ``DemonstrationWrapper`` explicitly builds ``np.zeros(2, float64)`` for stick envs, matching the measured record);
- info has exactly seven keys, ``status == "ongoing"``, ``task_goal`` is 1-4 non-empty strings;
- frame count: if this episode's task list contains a ``demonstration=True`` subtask (derived from ``unwrapped.task_list`` after reset) frames > 1,
  otherwise exactly 1; the derived presence is cross-checked against the measured list ``DEMO_TASKS_MEASURED`` (9 tasks);
- xhard1-5: ``spec_binding`` has ``mode == "replay"``, ``spec_sha256`` equals the packaged spec row (and equals the hash recomputed from that row's ``spec``),
  ``injected_mismatch == 0``, ``unused == 0``, proving the spec replay is really consumed by the env;
  xhard0 has no spec, ``mode`` must not be ``replay``;
- ``unwrapped`` ``seed`` and ``difficulty`` match the spec row (official test metadata for xhard0).
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

from robomme_ood.env_record_wrapper import BenchmarkEnvBuilder, hard_specs
from robomme_ood.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
from robomme_ood.env_record_wrapper.FailAwareWrapper import FailAwareWrapper

pytestmark = pytest.mark.sim

#: Measured list of demonstration tasks, used only for cross-checking (the primary criterion is whether ``unwrapped.task_list`` has ``demonstration=True`` after reset).
#: Source: measured record of the same-scope pre-cleanup scan ``docs/validation/test-redesign-20261003/records/reset-sweep.jsonl`` (all 59 cells ok):
#: these 9 tasks have > 1 frame in all tiers; the other 7 tasks have exactly 1 frame.
DEMO_TASKS_MEASURED = frozenset({
    "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "VideoUnmask", "VideoUnmaskSwap",
    "InsertPeg", "MoveCube", "PatternLock", "RouteStick",
})
assert len(DEMO_TASKS_MEASURED) == 9

OBS_SPEC = {
    "front_rgb_list": ((256, 256, 3), np.uint8),
    "wrist_rgb_list": ((256, 256, 3), np.uint8),
    "joint_state_list": ((7,), np.float32),
    "eef_state_list": ((6,), np.float64),
    "gripper_state_list": ((2,), np.float32),  # stick envs: see _expected_obs_spec
}
INFO_KEYS = {
    "elapsed_steps", "success", "fail", "simple_subgoal_online", "grounded_subgoal_online", "task_goal", "status",
}
CHAIN_NAMES = ("FailAwareWrapper", "DemonstrationWrapper", "TimeLimitWrapper", "OrderEnforcing")


def _enumerate_cells() -> list[tuple[str, str, int, str]]:
    """Enumerate (task, dataset, episode, tier) with the same scope as reset_sweep.py; reads only metadata and specs, builds no scenes."""
    cells: list[tuple[str, str, int, str]] = []
    for task in BenchmarkEnvBuilder.get_task_list():
        cells.append((task, "hard-verify", 0, hard_specs.XHARD0))  # hard-verify episode 0 = original episode XHARD0_EPISODES[0]
        builder = BenchmarkEnvBuilder(env_id=task, dataset="ood", action_space="joint_angle")
        seen: set[str] = set()
        for episode in range(builder.get_episode_num()):
            tier = builder.resolve_episode(episode)[1]
            if tier == hard_specs.XHARD0 or tier in seen:
                continue  # defensive: ood has no xhard0; xhard0 is covered by the hard-verify cells above
            seen.add(tier)
            cells.append((task, "ood", episode, tier))
    return cells


CELLS = _enumerate_cells()
_XHARD0 = [c for c in CELLS if c[3] == hard_specs.XHARD0]
_NEW = [c for c in CELLS if c[3] != hard_specs.XHARD0]
# Enumeration drift fails loudly at collection: 16 xhard0 + exactly the 43 cells of the delivery cell table
assert len(_XHARD0) == len(hard_specs.ALL_TASKS) == 16, f"xhard0 cell count {len(_XHARD0)} != 16"
assert {(t, tier) for t, _, _, tier in _NEW} == set(hard_specs.EXPECTED_CELLS) and len(_NEW) == 43, \
    f"new-value tier cell count {len(_NEW)} does not match delivery cell table EXPECTED_CELLS (43 cells)"
assert len(CELLS) == 59


@functools.lru_cache(maxsize=None)
def _spec_rows(tier: str) -> dict[tuple[str, int], dict]:
    """Packaged spec rows (formal episodes), indexed by (task, candidate); independent of the builder's read path."""
    _header, rows = hard_specs.load_specs(hard_specs.packaged_specs_path(tier), check_fingerprint=False)
    return {(row["task"], int(row["candidate"])): row for row in rows if hard_specs.delivered(row)}


def _chain(env) -> list:
    layers, e = [], env
    while hasattr(e, "env"):
        layers.append(e)
        e = e.env
    return layers + [e]


def _expected_obs_spec(unwrapped) -> dict:
    spec = dict(OBS_SPEC)
    if unwrapped.robot_uids == "panda_stick":
        spec["gripper_state_list"] = ((2,), np.float64)
    return spec


@pytest.mark.parametrize(
    ("task", "dataset", "episode", "tier"),
    CELLS,
    ids=[f"{task}-{tier}-ep{episode}" for task, _ds, episode, tier in CELLS],
)
def test_reset_cell(task: str, dataset: str, episode: int, tier: str) -> None:
    builder = BenchmarkEnvBuilder(
        env_id=task, dataset=dataset, action_space="joint_angle",
        max_steps=1300 if tier == hard_specs.XHARD0 else 1600,  # xhard0 uses the official default 1300, new-value tiers use EXEC_CAP 1600
    )
    if tier == hard_specs.XHARD0:
        expected_seed, resolved_tier = builder.resolve_episode(episode)
        assert resolved_tier == hard_specs.XHARD0
        assert builder.resolve_identity(episode)["source_episode"] == hard_specs.XHARD0_EPISODES[0]
        expected_difficulty = "hard"  # xhard0 takes the official native hard branch: gym.make receives difficulty="hard"
        row = None
    else:
        identity = builder.resolve_identity(episode)
        assert identity["tier"] == tier
        row = _spec_rows(tier)[(task, int(identity["candidate"]))]
        assert int(row["seed"]) == identity["seed"]
        assert row["spec_sha256"] == identity["spec_sha256"] == hard_specs.spec_sha256(row["spec"])
        expected_seed, expected_difficulty = int(row["seed"]), tier

    env = builder.make_env_for_episode(episode)
    try:
        obs, info = env.reset()
        unwrapped = env.unwrapped

        # —— wrapper chain ——
        layers = _chain(env)
        assert [type(x).__name__ for x in layers] == [*CHAIN_NAMES, task]
        assert type(layers[0]) is FailAwareWrapper  # official class (robomme_ood side is just a module alias)
        assert type(layers[1]) is DemonstrationWrapper  # robomme_ood's own copy
        assert layers[-1] is unwrapped
        assert type(unwrapped).__module__.startswith("robomme_ood."), type(unwrapped).__module__

        # —— obs ——
        spec = _expected_obs_spec(unwrapped)
        assert set(obs) == set(spec)
        lengths = {key: len(obs[key]) for key in spec}
        assert len(set(lengths.values())) == 1, f"five lists have unequal lengths: {lengths}"
        n_frames = lengths["front_rgb_list"]
        assert n_frames >= 1
        for key, (shape, dtype) in spec.items():
            for i, frame in enumerate(obs[key]):
                arr = np.asarray(frame)
                assert arr.shape == shape and arr.dtype == dtype, f"{key}[{i}] is {arr.shape} {arr.dtype}"
        if unwrapped.robot_uids == "panda_stick":
            assert all(not np.any(np.asarray(g)) for g in obs["gripper_state_list"])

        # —— info ——
        assert set(info) == INFO_KEYS
        assert info["status"] == "ongoing"
        goals = info["task_goal"]
        assert isinstance(goals, list) and 1 <= len(goals) <= 4, goals
        assert all(isinstance(g, str) and g.strip() for g in goals), goals

        # —— demonstration frames ——
        has_demo = any(t.get("demonstration", False) for t in unwrapped.task_list)
        if has_demo:
            assert n_frames > 1, f"{task} has demonstration subtasks but only {n_frames} frames"
        else:
            assert n_frames == 1, f"{task} has no demonstration subtasks but has {n_frames} frames"
        assert has_demo == (task in DEMO_TASKS_MEASURED), f"{task} demonstration presence does not match the measured list"

        # —— spec replay ——
        binding = hard_specs.spec_binding(env)
        if row is None:
            assert binding.get("mode") != "replay", binding
        else:
            assert binding["available"] is True
            assert binding["mode"] == "replay", binding
            assert binding["spec_sha256"] == row["spec_sha256"]
            assert binding["injected_mismatch"] == 0, binding
            assert binding["unused"] == 0, binding

        # —— identity ——
        assert int(unwrapped.seed) == int(expected_seed)
        assert unwrapped.difficulty == expected_difficulty
    finally:
        env.close()
