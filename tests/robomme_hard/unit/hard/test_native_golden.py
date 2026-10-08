"""原三档（easy／medium／hard）离线导出摘要等于金标准（取代旧测试里的「原三档逐字不变」AST 锁）。

金标准 ``native_golden.json`` 的生成方式见 ``native_golden.py``（在维护计划 BASE 上生成）。
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
    """金标准里的异常条目必须注明抛出点模块且在 ``robomme_hard`` 内——替身（``tests.*``）自身的错误不能被钉成契约。"""
    errors = {k: v for k, v in GOLD["digests"].items() if v.startswith("error:")}
    for k, v in errors.items():
        name, sep, module = v[len("error:"):].partition("@")
        assert sep and name, (k, v)
        assert module.startswith("robomme_hard."), (k, v)
    # 现状只有 VPO medium／hard 种子 101 两条（V4 H2 / K2：原三档布局失败表现为 TypeError）
    assert set(errors) == {"VideoPlaceOrder/medium/101", "VideoPlaceOrder/hard/101"}


def test_raise_site_module_distinguishes_test_double():
    """负例：测试模块里抛出的异常，抛出点记为测试模块名，不会被误记成 ``robomme_hard``。"""
    try:
        raise TypeError("替身错误")
    except TypeError as exc:
        assert NG.raise_site_module(exc) == __name__
        assert not NG.raise_site_module(exc).startswith("robomme_hard.")


def test_digest_depends_on_seed():
    """负例：摘要对种子敏感——同任务同档两个种子的摘要不相同（异常局除外），否则金标准抓不住布局变化。"""
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
