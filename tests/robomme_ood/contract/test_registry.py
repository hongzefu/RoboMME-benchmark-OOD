"""L1 契约（慢）：三种导入顺序下，16 个环境 id 都归 ``robomme_ood``。

每种顺序起一个独立子进程（同一进程里导入顺序只能发生一次）：只导入 ``robomme_ood``；先 ``robomme`` 后
``robomme_ood``；先 ``robomme_ood`` 后 ``robomme``。子进程只读注册表、不建环境；资源守卫经 sitecustomize 继承。
另有一个对照：只导入官方 ``robomme`` 的进程里，16 个 id 都归官方包（证明子进程的判定能区分两种归属）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.robomme_ood._support.loaders import REPO
from tests.robomme_ood.contract.test_constants import TASKS

pytestmark = pytest.mark.slow

ORDERS = {
    "hard_only": ["robomme_ood"],
    "official_then_hard": ["robomme", "robomme_ood"],
    "hard_then_official": ["robomme_ood", "robomme"],
    "official_only": ["robomme"],
}


def owners(order: list[str]) -> dict[str, str]:
    code = (
        "import importlib, json, sys, warnings\n"
        "warnings.simplefilter('ignore')\n"
        f"for name in {order!r}:\n"
        "    importlib.import_module(name)\n"
        "    if name == 'robomme':\n"
        "        importlib.import_module('robomme.robomme_env')\n"
        "from mani_skill.utils.registration import REGISTERED_ENVS\n"
        f"ids = {list(TASKS)!r}\n"
        "print('OWNERS=' + json.dumps({u: REGISTERED_ENVS[u].cls.__module__ for u in ids if u in REGISTERED_ENVS}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=240)
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("OWNERS="))
    return json.loads(line[len("OWNERS="):])


@pytest.mark.parametrize("name", ["hard_only", "official_then_hard", "hard_then_official"])
def test_all_ids_owned_by_hard(name):
    got = owners(ORDERS[name])
    assert set(got) == set(TASKS)
    stray = {uid: mod for uid, mod in got.items() if not mod.startswith("robomme_ood.")}
    assert stray == {}


def test_official_only_control():
    got = owners(ORDERS["official_only"])
    assert set(got) == set(TASKS)
    assert all(mod.startswith("robomme.") for mod in got.values())
