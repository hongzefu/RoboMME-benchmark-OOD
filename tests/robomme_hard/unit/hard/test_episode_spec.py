"""每局规格的导出／回注（``utils/episode_spec.SpecRecorder``）与回注绑定摘要（``hard_specs.spec_binding``，非分层路径）。

手写小规格做输入：导出模式返回抽样值并记录；回注模式一定返回冻结值、不等记 mismatch（新值模式带
decision_key 归因）；规格类别与难度不符、任务不符即拒；``record`` 的观测点按 1e-5 容差（生产常量
``RECORDED_FLOAT_TOL``）分成 recorded_drift 与 injected_mismatch，等于容差算尾差；``unused`` 数未被访问的取值点。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from robomme_hard.env_record_wrapper.hard_specs import RECORDED_FLOAT_TOL, spec_binding, spec_sha256
from robomme_hard.robomme_env.utils.episode_spec import (
    SPEC_KIND,
    SPEC_KIND_NEWVALUE,
    EpisodeSpecError,
    SpecRecorder,
    spec_kind_for,
)

TIER = "xhard2"


def _spec(kind=SPEC_KIND_NEWVALUE, task="T", **sections):
    doc = {"spec_kind": kind, "task": task, "identity": {"seed": 1}}
    doc.update(sections)
    return doc


def _env(rec):
    return SimpleNamespace(unwrapped=SimpleNamespace(_spec=rec))


def test_kind_follows_difficulty():
    assert spec_kind_for(TIER) == SPEC_KIND_NEWVALUE
    assert spec_kind_for("hard") == SPEC_KIND and spec_kind_for(None) == SPEC_KIND


def test_export_returns_draw_and_records_document():
    rec = SpecRecorder(None, "T", {"seed": 7}, difficulty=TIER)
    assert rec.mode == "export" and rec.value("objects.n", 5) == 5
    rec.record("layout.obs", [1.0, 2.0])
    doc = rec.to_dict()
    assert doc["objects"] == {"n": 5} and doc["layout"] == {"obs": [1.0, 2.0]}
    assert doc["spec_kind"] == SPEC_KIND_NEWVALUE and doc["identity"] == {"seed": 7}
    assert doc["provenance"] == {"mode": "export", "value_points": 1, "mismatches": 0, "unattributed_mismatches": 0}
    b = spec_binding(_env(rec))
    assert b["mode"] == "export" and b["spec_sha256"] is None and b["injected_mismatch"] == 0


def test_replay_always_returns_frozen_and_attributes_mismatch():
    rec = SpecRecorder(_spec(objects={"n": 9, "m": 1}), "T", difficulty=TIER)
    assert rec.value("objects.n", 5, decision_key="n.xhard2") == 9
    assert rec.value("objects.m", 2) == 1
    assert rec.mismatches == [
        {"path": "objects.n", "drawn": 5, "frozen": 9, "decision_key": "n.xhard2"},
        {"path": "objects.m", "drawn": 2, "frozen": 1, "decision_key": None},
    ]
    assert [m["path"] for m in rec.unattributed_mismatches()] == ["objects.m"]
    assert spec_binding(_env(rec))["injected_mismatch"] == 2


def test_native_mode_mismatch_has_no_attribution_field():
    rec = SpecRecorder(_spec(kind=SPEC_KIND, objects={"n": 9}), "T", difficulty="hard")
    rec.value("objects.n", 5, decision_key="ignored")
    assert rec.mismatches == [{"path": "objects.n", "drawn": 5, "frozen": 9}]
    assert rec.unattributed_mismatches() == rec.mismatches


@pytest.mark.parametrize("spec,difficulty,task", [
    (_spec(kind=SPEC_KIND), TIER, "T"),  # 原值规格喂新值档
    (_spec(kind=SPEC_KIND_NEWVALUE), "hard", "T"),  # 新值规格喂原三档
    (_spec(), TIER, "Other"),  # 任务不符
    ("not a dict", TIER, "T"),
])
def test_replay_rejects_wrong_kind_or_task(spec, difficulty, task):
    with pytest.raises(EpisodeSpecError):
        SpecRecorder(spec, task, difficulty=difficulty)


def test_replay_missing_value_point_is_rejected():
    rec = SpecRecorder(_spec(objects={}), "T", difficulty=TIER)
    with pytest.raises(EpisodeSpecError):
        rec.value("objects.n", 1)


def test_replay_does_not_alias_input():
    spec = _spec(objects={"lst": [1, 2]})
    rec = SpecRecorder(spec, "T", difficulty=TIER)
    spec["objects"]["lst"].append(3)
    assert rec.value("objects.lst", [1, 2]) == [1, 2] and rec.mismatches == []


def test_recorded_drift_tolerance_is_inclusive():
    """观测点：差恰等于容差 → recorded_drift；略大于容差 → injected_mismatch；回注点任何不等都算 injected。"""
    tol = RECORDED_FLOAT_TOL
    rec = SpecRecorder(_spec(layout={"a": 0.0, "b": 0.0, "c": [1.0, 2.0]}), "T", difficulty=TIER)
    rec.record("layout.a", tol)
    rec.record("layout.b", tol * 1.5)
    rec.record("layout.c", [1.0, 2.0])
    b = spec_binding(_env(rec))
    assert (b["recorded_drift"], b["injected_mismatch"]) == (1, 1)
    assert b["recorded_max_abs"] == pytest.approx(tol)
    rec2 = SpecRecorder(_spec(layout={"a": 0.0}), "T", difficulty=TIER)
    rec2.value("layout.a", tol / 10)
    assert spec_binding(_env(rec2))["injected_mismatch"] == 1, "回注点没有容差"


def test_record_structure_mismatch_is_injected_not_drift():
    rec = SpecRecorder(_spec(layout={"a": [0.0, 0.0]}), "T", difficulty=TIER)
    rec.record("layout.a", [0.0])
    b = spec_binding(_env(rec))
    assert (b["recorded_drift"], b["injected_mismatch"]) == (0, 1)


def test_record_absent_from_spec_is_written_not_mismatch():
    rec = SpecRecorder(_spec(layout={}), "T", difficulty=TIER)
    rec.record("layout.new", 3)
    assert rec.mismatches == [] and rec.to_dict()["layout"] == {"new": 3}


def test_unused_counts_unconsumed_leaves_and_prefix_consumption():
    spec = _spec(layout={"a": 1, "b": {"x": 1, "y": 2}}, objects={"n": 3}, actions={"z": 0})
    rec = SpecRecorder(spec, "T", difficulty=TIER)
    rec.value("layout.a", 1)
    rec.value("layout.b", {"x": 1, "y": 2})  # 整棵子树作为一个取值点消费
    assert sorted(rec.leaf_paths()) == ["actions.z", "layout.a", "layout.b.x", "layout.b.y", "objects.n"]
    b = spec_binding(_env(rec))
    assert b["unused"] == 2 and b["mode"] == "replay"
    assert b["spec_sha256"] == spec_sha256(spec)
    assert b["layered"] is False and b["layout_drift"] == 0


def test_binding_unavailable_without_recorder():
    assert spec_binding(SimpleNamespace(unwrapped=SimpleNamespace())) == {"available": False}
