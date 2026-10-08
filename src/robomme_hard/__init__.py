"""robomme_hard：RoboMME 新值四档（xhard1～xhard4）环境包，与官方 ``robomme`` 并列、分层继承。

``src/robomme/`` 逐字节等于官方 ``1fadc0ec``（清单见 ``UPSTREAM.json``）；本包只放差异：
16 个环境类与改过／新增／传递依赖改过模块的 utils、wrapper 一律复制，依赖闭包干净的官方模块用 shim 借用，
``BenchmarkEnvBuilder`` 子类化并新增 ``dataset="ood"``。

导入本包即以 ``override=True`` 接管 16 个环境 id（用户 U-3 批准的 P2 覆盖项）：同一进程里之后 ``gym.make``
这 16 个 id 一律得到本包的类；要官方行为须另开只导入 ``robomme`` 的进程。导入末尾断言：

* 注册表归属：``REGISTERED_ENVS[uid].cls.__module__`` 以 ``robomme_hard.`` 开头，否则 ``ImportError``；
* 命名空间归属：16 个环境模块与 ``utils`` 包里，凡名字在本包复制模块里有定义的可调用对象都必须属于本包
  （专挡官方 ``subgoal_evaluate_func`` 的 ``from robomme.robomme_env.utils import *`` 把官方函数灌回来的问题）；
* 借用目标 cheap 校验（字节数 + 首尾 1 MiB blake2b），不符只警告；full 档由 ``scripts/parity/upstream_guard.py`` 负责。
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
UPSTREAM = json.loads((_ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))

if "robomme.robomme_env" in sys.modules and "robomme_hard.robomme_env" not in sys.modules:
    warnings.warn("官方 robomme.robomme_env 已先导入；robomme_hard 将以 override=True 接管 16 个环境 id")

# 注册期间压低 ManiSkill 的「Override registered env」提示：必须拿 logger 对象本身
# （它的名字带尾随空格 "mani_skill "，按名字 getLogger("mani_skill") 拿到的是另一个 logger）
from mani_skill import logger as _maniskill_logger  # noqa: E402

_saved_level = _maniskill_logger.level
_maniskill_logger.setLevel(logging.ERROR)
try:
    from . import robomme_env  # noqa: E402
finally:
    _maniskill_logger.setLevel(_saved_level)


def _check_registry_owner() -> None:
    from mani_skill.utils.registration import REGISTERED_ENVS

    stray = {uid: REGISTERED_ENVS[uid].cls.__module__ for uid in robomme_env.ENV_IDS
             if not REGISTERED_ENVS[uid].cls.__module__.startswith("robomme_hard.")}
    if stray:
        raise ImportError(f"robomme_hard 注册表归属断言失败（这些 id 不归本包）：{stray}")


def own_callable_names() -> dict[str, str]:
    """本包复制模块里定义的可调用对象名 → 定义模块（shim 借用的模块不在内）。"""
    names: dict[str, str] = {}
    for modname, module in list(sys.modules.items()):
        if not modname.startswith("robomme_hard.") or getattr(module, "__name__", "") != modname:
            continue
        for name, obj in vars(module).items():
            if callable(obj) and getattr(obj, "__module__", None) == modname:
                names.setdefault(name, modname)
    return names


def namespace_strays() -> list[str]:
    """16 个环境模块与 utils 包命名空间里，名字属于本包复制模块、对象却来自别处的条目。"""
    own = own_callable_names()
    modules = [f"robomme_hard.robomme_env.{uid}" for uid in robomme_env.ENV_IDS]
    modules.append("robomme_hard.robomme_env.utils")
    strays = []
    for modname in modules:
        for name, obj in vars(sys.modules[modname]).items():
            if name in own and callable(obj) and not str(getattr(obj, "__module__", "")).startswith("robomme_hard."):
                strays.append(f"{modname}.{name}<-{obj.__module__}")
    return strays


def _check_namespace_owner() -> None:
    strays = namespace_strays()
    if strays:
        raise ImportError(f"robomme_hard 命名空间归属断言失败（官方同名对象灌回了本包）：{strays[:10]}")


def _cheap_check_shims() -> None:
    spec = importlib.util.find_spec("robomme")
    root = Path(list(spec.submodule_search_locations)[0])
    for entry in UPSTREAM["shims"]:
        path = root / entry["target_file"][len("src/robomme/"):]
        data = path.read_bytes() if path.is_file() else b""
        cheap = hashlib.blake2b(data[: 1 << 20] + data[-(1 << 20):]).hexdigest()
        if len(data) != entry["target_bytes"] or cheap != entry["target_cheap"]:
            warnings.warn(f"借用目标 {entry['target_module']} 与 UPSTREAM.json 不符（cheap 档；只警告）")


_check_registry_owner()
_check_namespace_owner()
_cheap_check_shims()
