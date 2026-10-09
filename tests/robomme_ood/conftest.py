"""Root conftest: holds only path fixtures shared by the whole suite. Resource guard and --allow-sim-reset live in tests/robomme_ood/_support/resource_policy.py."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
