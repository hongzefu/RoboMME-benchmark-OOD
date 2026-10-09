"""``sampling_config`` guard and difficulty tools (hard package ``utils/sampling_config.py``, ``utils/difficulty.py``).

Inputs are small hand-written tables: splitting (old and new formats, invalid shapes, deep copy without aliasing), boundaries of filling missing new-value tiers, per-key guard on the original-value part and
structural guard on the new-value part; difficulty normalization and tier numbers. Also, for the real ``_resolve_sampling_config`` of all 16 tasks: the in-package header config
is accepted; changing one original value or adding an undeclared key in a new-value tier is rejected.
"""
from __future__ import annotations

import copy

import pytest

from robomme_ood.robomme_env.utils import difficulty as D
from robomme_ood.robomme_env.utils import sampling_config as SC

from . import offline_scene as O

NATIVE = {"parameters": {"a": 1}, "positions": {"p": [0.0, 1.0]}}
DECISION = {"n": {"easy": 1, "hard": 3, "xhard4": 9, "xhard1": 6}, "xhard4": {"k": 2}, "xhard1": {"k": 1}}


# ── split_sampling_config ─────────────────────────────────────────────────────


def test_split_none_returns_deep_copies_of_defaults():
    dec, nat = SC.split_sampling_config(None, NATIVE, DECISION)
    assert dec == DECISION and nat == NATIVE
    dec["n"]["easy"] = 99
    nat["parameters"]["a"] = 99
    assert DECISION["n"]["easy"] == 1 and NATIVE["parameters"]["a"] == 1


def test_split_new_and_legacy_formats():
    override = {"decision": {"x": 1}, "native": copy.deepcopy(NATIVE)}
    dec, nat = SC.split_sampling_config(override, NATIVE, DECISION)
    assert dec == {"x": 1} and nat == NATIVE
    override["native"]["parameters"]["a"] = 5
    assert nat["parameters"]["a"] == 1, "return value must not alias the input"
    dec, nat = SC.split_sampling_config({"native": NATIVE}, NATIVE, DECISION)
    assert dec == DECISION
    dec, nat = SC.split_sampling_config(copy.deepcopy(NATIVE), NATIVE, DECISION)  # old format = only native given
    assert dec == DECISION and nat == NATIVE


@pytest.mark.parametrize("bad", [
    [],  # not a dict
    {"decision": {}},  # missing native
    {"native": NATIVE, "extra": 1},  # extra key
    {"native": {"parameters": {}}},  # native missing positions
    {"native": NATIVE, "decision": []},  # decision is not a dict
])
def test_split_rejects_malformed(bad):
    with pytest.raises(SC.SamplingConfigError):
        SC.split_sampling_config(bad, NATIVE, DECISION)


# ── fill_missing_newvalue ─────────────────────────────────────────────────────


def test_fill_adds_only_v6_tiers_next_to_existing_xhard4():
    default = {"t": {"xhard4": {"k": 4}, "xhard1": {"k": 1}, "xhard2": {"k": 2}, "xhard5": {"k": 5}}}
    got = SC.fill_missing_newvalue({"t": {"xhard4": {"k": 40}}}, default)
    assert got == {"t": {"xhard4": {"k": 40}, "xhard1": {"k": 1}, "xhard2": {"k": 2}}}, "xhard5 is outside the fill range"
    # no xhard4 on the same level (or only the historical key xhard): never fill
    assert SC.fill_missing_newvalue({"t": {"xhard": {"k": 0}}}, default) == {"t": {"xhard": {"k": 0}}}
    assert SC.fill_missing_newvalue({"t": {}}, default) == {"t": {}}


def test_fill_does_not_alias_default():
    default = {"xhard4": {"k": 4}, "xhard1": {"lst": [1]}}
    got = SC.fill_missing_newvalue({"xhard4": {"k": 4}}, default)
    got["xhard1"]["lst"].append(2)
    assert default["xhard1"]["lst"] == [1]


# ── assert_native_decision ────────────────────────────────────────────────────


def test_guard_accepts_value_changes_inside_declared_newvalue_tiers():
    dec = copy.deepcopy(DECISION)
    dec["n"]["xhard4"] = 12
    dec["xhard1"]["k"] = 7
    SC.assert_native_decision(dec, DECISION, "T")


def test_guard_rejects_native_deviation():
    dec = copy.deepcopy(DECISION)
    dec["n"]["hard"] = 4
    with pytest.raises(SC.SamplingConfigError, match="original value"):
        SC.assert_native_decision(dec, DECISION, "T")


def test_guard_rejects_undeclared_newvalue_key():
    dec = copy.deepcopy(DECISION)
    dec["xhard1"]["undeclared"] = 1
    with pytest.raises(SC.SamplingConfigError, match="xhard"):
        SC.assert_native_decision(dec, DECISION, "T")


def test_guard_ignores_tiers_absent_from_snapshot():
    dec = copy.deepcopy(DECISION)
    del dec["xhard1"]
    del dec["n"]["xhard1"]
    SC.assert_native_decision(dec, DECISION, "T")


# ── difficulty ────────────────────────────────────────────────────────────────


def test_difficulty_family_tier_numbers_and_normalisation():
    assert [D.newvalue_tier(t) for t in D.ALL_NEWVALUE_TIERS] == list(range(1, len(D.ALL_NEWVALUE_TIERS) + 1))
    assert D.newvalue_tier("hard") == 0 and D.newvalue_tier(None) == 0
    assert D.is_newvalue_difficulty(" XHARD2 ") and not D.is_newvalue_difficulty("hard")
    assert not D.is_newvalue_difficulty(None) and not D.is_newvalue_difficulty("xhard")
    assert D.normalize_robomme_difficulty(" Medium ") == "medium"
    assert D.normalize_robomme_difficulty(None) is None
    with pytest.raises(ValueError):
        D.normalize_robomme_difficulty("xhard9")
    with pytest.raises(TypeError):
        D.normalize_robomme_difficulty(3)


def test_require_xhard4_only():
    D.require_xhard4_only("xhard4", "MoveCube")
    D.require_xhard4_only("hard", "MoveCube")
    D.require_xhard4_only(None, "MoveCube")
    for tier in ("xhard1", "xhard2", "xhard3", "xhard5"):
        with pytest.raises(ValueError):
            D.require_xhard4_only(tier, "MoveCube")


# -- real _resolve_sampling_config of all 16 tasks ---------------------------------------


@pytest.mark.parametrize("task,tier", O.delivered_cells())
def test_task_accepts_packaged_header_config(task, tier):
    header, _ = O.delivered_rows(task, tier, 0)
    mod = O.task_module(task)
    resolved = mod._resolve_sampling_config(O.task_class(task), header["sampling_config"][task])
    # resolved result = the two native blocks + the attached decision (the return shape of production _resolve_sampling_config);
    # every native key in the header is kept as-is (BinFill and RouteStick additionally add configs under parameters without changing existing keys)
    assert set(resolved) == {"parameters", "positions", "decision"}
    assert resolved["decision"] == header["sampling_config"][task]["decision"]
    native = header["sampling_config"][task]["native"]
    for sec in ("parameters", "positions"):
        assert {k: resolved[sec][k] for k in native[sec]} == native[sec]


@pytest.mark.parametrize("task", sorted({t for t, _ in O.delivered_cells()}))
def test_task_rejects_tampered_native_decision(task):
    tier = O.tiers_of(task)[0]
    header, _ = O.delivered_rows(task, tier, 0)
    cfg = copy.deepcopy(header["sampling_config"][task])
    mod, cls = O.task_module(task), O.task_class(task)
    # add an undeclared key visible to the original three tiers in decision: the original-value guard must reject it
    cfg["decision"]["__undeclared__"] = 1
    with pytest.raises(SC.SamplingConfigError):
        mod._resolve_sampling_config(cls, cfg)
