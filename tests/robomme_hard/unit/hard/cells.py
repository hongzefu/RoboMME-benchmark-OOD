"""按格（任务 × V9 交付档）的公共断言：包内规格回放、自导出、篡改负例。

* 包内回放：该格前 3 个正式局（builder 同序：``delivered`` 且按 candidate 升序）的 ``spec`` 作
  ``native_episode_spec``，按评估链参数（``seed``、``difficulty``、header 内嵌 ``sampling_config``）离线跑真实
  ``_load_scene`` 与两次 ``_initialize_episode``（评估链 make + reset 的真实次数）——每个回注点的原抽样必须与
  冻结值逐位相等（``value()`` 记录的 mismatch 为空）；生产摘要 ``spec_binding`` 的 ``injected_mismatch == 0``、
  ``unused == 0``、``mode == "replay"``、``spec_sha256`` 等于行上的 ``spec_sha256``。
* 自导出：同一 seed 不传规格导出，导出文档的每个取值点与包内规格相同（记录点允许 ``RECORDED_FLOAT_TOL``，
  这是 GPU 生成与 CPU 离线的 float32 尾差，取自生产常量），再把导出文档回放，回注点零不等。
* 篡改负例：把一个回注点的冻结值改掉再回放，必须被记成 mismatch（或在回放时被生产复核拒绝）。
"""
from __future__ import annotations

import copy
import functools

from robomme_hard.env_record_wrapper.hard_specs import RECORDED_FLOAT_TOL, spec_binding, spec_sha256
from robomme_hard.robomme_env.utils.episode_spec import EpisodeSpecError
from robomme_hard.robomme_env.utils.SceneGenerationError import SceneGenerationError

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
    """树形逐值比较：数值差 ≤ tol，结构与非数值严格相等。"""
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
    """回注点（``value()``）上的不等；记录点（``record()``）的尾差单独看。"""
    rec = env._spec
    spec_paths = {t["path"] for t in rec.trace if t["source"] == "spec"}
    return [m for m in rec.mismatches if m["path"] in spec_paths]


def unused_paths(env) -> list[str]:
    rec = env._spec
    consumed = set(rec.consumed_paths())
    return [p for p in rec.leaf_paths() if not any(p == c or p.startswith(c + ".") for c in consumed)]


@functools.lru_cache(maxsize=None)
def replayed(task: str, tier: str, k: int):
    """第 k 个正式局的离线回放（缓存；调用方只读）。"""
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
    assert binding["unused"] == 0, f"规格里有、本局没消费：{unused_paths(env)}"


def check_self_export(task: str, tier: str, k: int) -> None:
    row, env, doc = exported(task, tier, k)
    assert env._spec.mode == "export" and env._spec.mismatches == []
    got, want = spec_leaves(doc), spec_leaves(row["spec"])
    assert set(got) == set(want), f"取值点集合不同：多 {sorted(set(got) - set(want))} 少 {sorted(set(want) - set(got))}"
    diff = [p for p in got if not close(got[p], want[p], RECORDED_FLOAT_TOL)]
    assert diff == [], f"离线导出与包内规格不符：{diff[:5]}"
    # 导出文档回放：CPU→CPU，回注点零不等，记录点也逐位相等
    with cpu_world():
        env2 = World.build(task, tier, k, spec=copy.deepcopy(doc)).env
    assert env2._spec.mismatches == []
    b = spec_binding(env2)
    assert b["injected_mismatch"] == 0 and b["unused"] == 0 and b["recorded_drift"] == 0


def first_value_path(env) -> str:
    """第一个回注点路径（取自生产 trace，不在测试里写死）。"""
    for t in env._spec.trace:
        if t["source"] == "spec":
            return t["path"]
    raise AssertionError("本局没有回注点")


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
    raise AssertionError(f"无法篡改 {value!r}")


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
    """负例：第一个回注点被篡改后回放，生产代码必须记 mismatch 或直接拒绝。"""
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
    assert path in paths, f"篡改 {path} 未被发现：{paths}"
    assert spec_binding(env2)["injected_mismatch"] >= 1


def replay_cases(*tasks: str, slow_from: int | None = None):
    """(task, tier, k) 参数：tier 取各任务 V9 实际交付档；``slow_from`` 起的行标 slow
    （只用于重型 Swap 任务，控制日常门禁耗时，不删断言）。"""
    import pytest

    out = []
    for task in tasks:
        for tier in O.tiers_of(task):
            for k in range(REPLAY_ROWS):
                marks = [pytest.mark.slow] if slow_from is not None and k >= slow_from else []
                out.append(pytest.param(task, tier, k, id=f"{task}-{tier}-r{k}", marks=marks))
    return out
