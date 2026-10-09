"""robomme_ood: RoboMME new-value tiers (xhard1-xhard4) environment package, alongside upstream ``robomme`` with layered inheritance.

``src/robomme/`` is byte-identical to upstream ``1fadc0ec`` (manifest: ``UPSTREAM.json``); this package holds only the differences:
the 16 environment classes, plus utils/wrappers that were changed, added, or whose transitive dependencies changed, are copied;
upstream modules with a clean dependency closure are borrowed via shims; ``BenchmarkEnvBuilder`` is subclassed to add ``dataset="ood"``.

Importing this package takes over the 16 environment ids with ``override=True`` (P2 override approved by the user, U-3): afterwards,
``gym.make`` on these 16 ids in the same process always returns this package's classes; for upstream behavior use a separate
process that imports only ``robomme``. Assertions at the end of import:

* Registry ownership: ``REGISTERED_ENVS[uid].cls.__module__`` must start with ``robomme_ood.``, otherwise ``ImportError``;
* Namespace ownership: in the 16 environment modules and the ``utils`` package, every callable whose name is defined in one of
  this package's copied modules must belong to this package (guards against upstream ``subgoal_evaluate_func``'s
  ``from robomme.robomme_env.utils import *`` pulling upstream functions back in);
* Cheap check of borrow targets (byte size + blake2b of the first/last 1 MiB), warning only on mismatch; the full level is handled
  by the benchmark repo's BENCH_UPSTREAM gate (``git diff --quiet 016ac1c4 HEAD -- src/robomme ...``: zero changes to upstream files
  relative to the upstream commit).
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

if "robomme.robomme_env" in sys.modules and "robomme_ood.robomme_env" not in sys.modules:
    warnings.warn("upstream robomme.robomme_env was imported first; robomme_ood will take over the 16 environment ids with override=True")

# Silence ManiSkill's "Override registered env" message during registration: must use the logger object itself
# (its name has a trailing space "mani_skill "; getLogger("mani_skill") by name returns a different logger)
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
             if not REGISTERED_ENVS[uid].cls.__module__.startswith("robomme_ood.")}
    if stray:
        raise ImportError(f"robomme_ood registry ownership assertion failed (these ids do not belong to this package): {stray}")


def own_callable_names() -> dict[str, str]:
    """Callable names defined in this package's copied modules -> defining module (shim-borrowed modules excluded)."""
    names: dict[str, str] = {}
    for modname, module in list(sys.modules.items()):
        if not modname.startswith("robomme_ood.") or getattr(module, "__name__", "") != modname:
            continue
        for name, obj in vars(module).items():
            if callable(obj) and getattr(obj, "__module__", None) == modname:
                names.setdefault(name, modname)
    return names


def namespace_strays() -> list[str]:
    """Entries in the namespaces of the 16 environment modules and the utils package whose name belongs to this package's copied modules but whose object comes from elsewhere."""
    own = own_callable_names()
    modules = [f"robomme_ood.robomme_env.{uid}" for uid in robomme_env.ENV_IDS]
    modules.append("robomme_ood.robomme_env.utils")
    strays = []
    for modname in modules:
        for name, obj in vars(sys.modules[modname]).items():
            if name in own and callable(obj) and not str(getattr(obj, "__module__", "")).startswith("robomme_ood."):
                strays.append(f"{modname}.{name}<-{obj.__module__}")
    return strays


def _check_namespace_owner() -> None:
    strays = namespace_strays()
    if strays:
        raise ImportError(f"robomme_ood namespace ownership assertion failed (same-named upstream objects leaked back into this package): {strays[:10]}")


def _cheap_check_shims() -> None:
    spec = importlib.util.find_spec("robomme")
    root = Path(list(spec.submodule_search_locations)[0])
    for entry in UPSTREAM["shims"]:
        path = root / entry["target_file"][len("src/robomme/"):]
        data = path.read_bytes() if path.is_file() else b""
        cheap = hashlib.blake2b(data[: 1 << 20] + data[-(1 << 20):]).hexdigest()
        if len(data) != entry["target_bytes"] or cheap != entry["target_cheap"]:
            warnings.warn(f"borrow target {entry['target_module']} does not match UPSTREAM.json (cheap level; warning only)")


_check_registry_owner()
_check_namespace_owner()
_cheap_check_shims()
