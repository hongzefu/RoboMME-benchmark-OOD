"""Offline export digests of the original three tiers (easy/medium/hard) equal the golden file (replaces the old tests' "original tiers verbatim unchanged" AST lock).

How the golden file ``native_golden.json`` is generated: see ``native_golden.py`` (generated on maintenance plan BASE).
"""
from __future__ import annotations

import json

import pytest

from . import native_golden as NG
from . import offline_scene as O

GOLD = json.loads(NG.GOLDEN.read_text(encoding="utf-8"))


def test_golden_covers_all_tasks_tiers_and_seeds():
    assert GOLD["tiers"] == list(NG.NATIVE_TIERS) and GOLD["seeds"] == list(NG.SEEDS)
    assert set(GOLD["digests"]) == {NG.key(t, tier, s) for t in O.ALL_TASKS for tier in NG.NATIVE_TIERS
                                     for s in NG.SEEDS}


def test_golden_errors_raised_in_production_code():
    """Exception entries in the golden file must name the raise-site module and it must be inside ``robomme_ood``; errors of the stand-ins themselves (``tests.*``) must not be pinned as a contract."""
    errors = {k: v for k, v in GOLD["digests"].items() if v.startswith("error:")}
    for k, v in errors.items():
        name, sep, module = v[len("error:"):].partition("@")
        assert sep and name, (k, v)
        assert module.startswith("robomme_ood."), (k, v)
    # currently only two entries, VPO medium/hard seed 101 (V4 H2 / K2: original-tier layout failure shows up as TypeError)
    assert set(errors) == {"VideoPlaceOrder/medium/101", "VideoPlaceOrder/hard/101"}


def test_raise_site_module_distinguishes_test_double():
    """Negative case: for an exception raised in a test module, the raise site is recorded as the test module name and never misrecorded as ``robomme_ood``."""
    try:
        raise TypeError("stand-in error")
    except TypeError as exc:
        assert NG.raise_site_module(exc) == __name__
        assert not NG.raise_site_module(exc).startswith("robomme_ood.")


def test_digest_depends_on_seed():
    """Negative case: digests are seed-sensitive; two seeds of the same task and tier give different digests (except exception episodes), otherwise the golden file cannot catch layout changes."""
    d = GOLD["digests"]
    for t in O.ALL_TASKS:
        for tier in NG.NATIVE_TIERS:
            a, b = (d[NG.key(t, tier, s)] for s in NG.SEEDS)
            if not (a.startswith("error:") or b.startswith("error:")):
                assert a != b, (t, tier)


@pytest.mark.parametrize("task", O.ALL_TASKS)
@pytest.mark.parametrize("tier", NG.NATIVE_TIERS)
def test_native_export_matches_golden(task, tier):
    for seed in NG.SEEDS:
        assert NG.summarize(task, tier, seed) == GOLD["digests"][NG.key(task, tier, seed)], (task, tier, seed)
