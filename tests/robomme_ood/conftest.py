"""根 conftest：只放全套件共享的路径夹具。资源守卫与 --allow-sim-reset 在 tests/robomme_ood/_support/resource_policy.py。"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
