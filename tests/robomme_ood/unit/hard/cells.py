"""Per-cell (task × V9 delivered tier) shared assertions: packaged spec replay, self-export, tampering negative cases.

* Packaged replay: the ``spec`` of the first 3 formal episodes of the cell (same order as builder: ``delivered`` in ascending candidate order) is used as
  ``native_episode_spec``; with the evaluation chain's parameters (``seed``, ``difficulty``, header-embedded ``sampling_config``) the real
  ``_load_scene`` and two ``_initialize_episode`` calls (the real count of evaluation chain make + reset) run offline; the original draw at every injection point must equal
  the frozen value bit for bit (mismatches recorded by ``value()`` are empty); the production summary ``spec_binding`` has ``injected_mismatch == 0``,
  ``unused == 0``, ``mode == "replay"``, and ``spec_sha256`` equal to the row's ``spec_sha256``.
* Self-export: export with the same seed and no spec; every value site of the exported document equals the packaged spec (record points allow ``RECORDED_FLOAT_TOL``,
  the float32 tail difference between GPU generation and CPU offline, taken from a production constant); then replay the exported document with zero mismatches at injection points.
* Tampering negative case: change the frozen value of one injection point and replay; it must be recorded as a mismatch (or rejected by production re-checks during replay).
"""
from __future__ import annotations

import copy
import functools

from robomme_ood.env_record_wrapper.hard_specs import RECORDED_FLOAT_TOL, spec_binding, spec_sha256
from robomme_ood.robomme_env.utils.episode_spec import EpisodeSpecError
from robomme_ood.robomme_env.utils.SceneGenerationError import SceneGenerationError

from . import offline_scene as O
from .world import World, cpu_world

SECTIONS = ("layout", "objects", "actions", "initializations")
REPLAY_ROWS = 3


def leaves(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else str(k))
    else:
        yield prefix, node


def spec_leaves(spec: dict) -> dict:
    out = {}
    for sec in SECTIONS:
        if spec.get(sec) is not None:
            out.update(dict(leaves(spec[sec], sec)))
    return out


def close(a, b, tol: float) -> bool:
    """Tree-wise value comparison: numeric difference ≤ tol, structure and non-numerics strictly equal."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    return a == b


def value_mismatches(env) -> list[dict]:
    """Mismatches at injection points (``value()``); tail differences of record points (``record()``) are looked at separately."""
    rec = env._spec
    spec_paths = {t["path"] for t in rec.trace if t["source"] == "spec"}
    return [m for m in rec.mismatches if m["path"] in spec_paths]


def unused_paths(env) -> list[str]:
    rec = env._spec
    consumed = set(rec.consumed_paths())
    return [p for p in rec.leaf_paths() if not any(p == c or p.startswith(c + ".") for c in consumed)]


@functools.lru_cache(maxsize=None)
def replayed(task: str, tier: str, k: int):
    """Offline replay of the k-th formal episode (cached; callers read only)."""
    header, rows = O.delivered_rows(task, tier, REPLAY_ROWS)
    row = rows[k]
    with cpu_world():
        world = World.build(task, tier, k)
    return row, world.env


@functools.lru_cache(maxsize=None)
def exported(task: str, tier: str, k: int):
    header, rows = O.delivered_rows(task, tier, REPLAY_ROWS)
    row = rows[k]
    with cpu_world():
        env = O.make_offline(task, seed=row["seed"], difficulty=tier, sampling_config=header["sampling_config"][task])
        World.from_env(env)
    return row, env, env._spec.to_dict()


def check_packaged_replay(task: str, tier: str, k: int) -> None:
    row, env = replayed(task, tier, k)
    assert env.seed == row["seed"] and env.difficulty == tier
    assert value_mismatches(env) == []
    binding = spec_binding(env)
    assert binding["mode"] == "replay"
    assert binding["injected_mismatch"] == 0
    assert binding["recorded_max_abs"] <= RECORDED_FLOAT_TOL
    assert binding["spec_sha256"] == row["spec_sha256"] == spec_sha256(row["spec"])
    assert binding["unused"] == 0, f"in the spec but not consumed by this episode: {unused_paths(env)}"


def check_self_export(task: str, tier: str, k: int) -> None:
    row, env, doc = exported(task, tier, k)
    assert env._spec.mode == "export" and env._spec.mismatches == []
    got, want = spec_leaves(doc), spec_leaves(row["spec"])
    assert set(got) == set(want), f"value site sets differ: extra {sorted(set(got) - set(want))} missing {sorted(set(want) - set(got))}"
    diff = [p for p in got if not close(got[p], want[p], RECORDED_FLOAT_TOL)]
    assert diff == [], f"offline export does not match the packaged spec: {diff[:5]}"
    # replay of the exported document: CPU→CPU, zero mismatches at injection points, record points also equal bit for bit
    with cpu_world():
        env2 = World.build(task, tier, k, spec=copy.deepcopy(doc)).env
    assert env2._spec.mismatches == []
    b = spec_binding(env2)
    assert b["injected_mismatch"] == 0 and b["unused"] == 0 and b["recorded_drift"] == 0


def first_value_path(env) -> str:
    """Path of the first injection point (taken from the production trace, not hard-coded in tests)."""
    for t in env._spec.trace:
        if t["source"] == "spec":
            return t["path"]
    raise AssertionError("this episode has no injection point")


def _perturb(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1e-3
    if isinstance(value, list) and value:
        return [_perturb(value[0]), *value[1:]]
    if isinstance(value, dict) and value:
        k = next(iter(value))
        return {**value, k: _perturb(value[k])}
    if isinstance(value, str):
        return value + "_x"
    raise AssertionError(f"cannot tamper with {value!r}")


def _set(tree: dict, path: str, value) -> None:
    node = tree
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value


def _get(tree: dict, path: str):
    node = tree
    for p in path.split("."):
        node = node[p]
    return node


def check_tamper_detected(task: str, tier: str) -> None:
    """Negative case: after tampering with the first injection point and replaying, production code must record a mismatch or reject outright."""
    row, env = replayed(task, tier, 0)
    path = first_value_path(env)
    bad = copy.deepcopy(row["spec"])
    _set(bad, path, _perturb(_get(bad, path)))
    try:
        with cpu_world():
            env2 = World.build(task, tier, 0, spec=bad).env
    except (SceneGenerationError, EpisodeSpecError, ValueError, AssertionError):
        return
    paths = [m["path"] for m in value_mismatches(env2)]
    assert path in paths, f"tampering with {path} not detected: {paths}"
    assert spec_binding(env2)["injected_mismatch"] >= 1


def replay_cases(*tasks: str, slow_from: int | None = None):
    """(task, tier, k) parameters: tier is each task's actually delivered V9 tier; rows from ``slow_from`` on are marked slow
    (only for the heavy Swap tasks, to control daily gate time; no assertions removed)."""
    import pytest

    out = []
    for task in tasks:
        for tier in O.tiers_of(task):
            for k in range(REPLAY_ROWS):
                marks = [pytest.mark.slow] if slow_from is not None and k >= slow_from else []
                out.append(pytest.param(task, tier, k, id=f"{task}-{tier}-r{k}", marks=marks))
    return out
