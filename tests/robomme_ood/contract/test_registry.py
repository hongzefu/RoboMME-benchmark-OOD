"""L1 contract (slow): under three import orders, all 16 env ids belong to ``robomme_ood``.

Each order runs in its own subprocess (import order can only happen once per process): import only ``robomme_ood``; ``robomme`` then
``robomme_ood``; ``robomme_ood`` then ``robomme``. Subprocesses only read the registry and build no env; the resource guard is inherited via sitecustomize.
There is also a control: in a process that imports only the official ``robomme``, all 16 ids belong to the official package (proving the subprocess check can tell the two apart).
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
