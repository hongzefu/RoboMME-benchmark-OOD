"""Per-episode spec export/injection (``utils/episode_spec.SpecRecorder``) and the injection binding summary (``hard_specs.spec_binding``, non-tiered path).

Hand-written small specs as input: export mode returns and records the sampled value; injection mode always returns the frozen value and records a mismatch on inequality (new-value mode attributes it with
decision_key); spec kind not matching the difficulty or a task mismatch is rejected; ``record`` observation points are split by the 1e-5 tolerance (production constant
``RECORDED_FLOAT_TOL``) into recorded_drift and injected_mismatch, equal to the tolerance counts as tail difference; ``unused`` counts value sites never accessed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from robomme_ood.env_record_wrapper.hard_specs import RECORDED_FLOAT_TOL, spec_binding, spec_sha256
from robomme_ood.robomme_env.utils.episode_spec import (
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
    (_spec(kind=SPEC_KIND), TIER, "T"),  # original-value spec fed to a new-value tier
    (_spec(kind=SPEC_KIND_NEWVALUE), "hard", "T"),  # new-value spec fed to an original tier
    (_spec(), TIER, "Other"),  # task mismatch
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
    """Observation points: difference exactly equal to the tolerance → recorded_drift; slightly above → injected_mismatch; any inequality at an injection point counts as injected."""
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
    assert spec_binding(_env(rec2))["injected_mismatch"] == 1, "injection points have no tolerance"


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
    rec.value("layout.b", {"x": 1, "y": 2})  # the whole subtree is consumed as one value site
    assert sorted(rec.leaf_paths()) == ["actions.z", "layout.a", "layout.b.x", "layout.b.y", "objects.n"]
    b = spec_binding(_env(rec))
    assert b["unused"] == 2 and b["mode"] == "replay"
    assert b["spec_sha256"] == spec_sha256(spec)
    assert b["layered"] is False and b["layout_drift"] == 0


def test_binding_unavailable_without_recorder():
    assert spec_binding(SimpleNamespace(unwrapped=SimpleNamespace())) == {"available": False}
