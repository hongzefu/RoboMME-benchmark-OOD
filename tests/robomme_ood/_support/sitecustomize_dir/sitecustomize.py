"""测试子进程继承资源守卫：只在测试会话设置了守卫环境变量时生效。"""
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
