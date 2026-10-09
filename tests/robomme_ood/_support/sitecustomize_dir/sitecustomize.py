"""Test subprocesses inherit the resource guard: active only when the test session has set the guard environment variables."""
import os

if os.environ.get("ROBOMME_TEST_RESOURCE_POLICY") == "cpu":
    import importlib.util
    import sys
    from pathlib import Path

    _p = Path(__file__).resolve().parents[1] / "resource_policy.py"
    _spec = importlib.util.spec_from_file_location("_robomme_test_resource_policy", _p)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["_robomme_test_resource_policy"] = _mod
    _spec.loader.exec_module(_mod)
    _mod.install()
