"""Computation of offline export digests for the original three tiers (easy/medium/hard) and golden file generation.

Digest = for 16 tasks × 3 tiers × seeds ``SEEDS``, with the evaluation chain's parameters (four runtime items + seed + difficulty, without
sampling_config or spec, i.e. the native branch) run the real ``_load_scene`` + two ``_initialize_episode`` offline and take the exported spec document's
``hard_specs.digest`` (sha256 of canonical JSON, removing key-order effects).

Scene-build exceptions are recorded as ``error:<exception class name>@<raise-site module>``, where the raise-site module is the ``__name__`` of the module of the innermost traceback frame,
to distinguish exceptions from production code (``robomme_ood.*``) from those of test stand-ins (``tests.*``, e.g. ``offline_scene``);
``test_native_golden.py`` asserts that the raise site of every exception entry in the golden file is inside ``robomme_ood``.

Origin of the two existing exception entries ``VideoPlaceOrder/medium/101`` and ``VideoPlaceOrder/hard/101`` (measured in T12):
the real ``spawn_random_target`` layout fails and raises RuntimeError → inside ``VideoPlaceOrder._load_scene``
``raise _SceneGenError(...)`` raises TypeError (module not callable) because the name ``SceneGenerationError`` is shadowed by ``from .utils import *`` into a same-named submodule
→ the outer ``except _SceneGenError:`` raises TypeError again (the except clause is not an exception class),
and the innermost frame is in ``robomme_ood.robomme_env.VideoPlaceOrder._load_scene``, not an error of the stand-in itself. This is deliberately kept official behavior:
see the "V4 H2" comment in ``src/robomme_ood/robomme_env/VideoPlaceOrder.py`` (the xhard branch uses a real exception class, original-tier
behavior unchanged verbatim), and decision K2 in ``docs/plans/0922-newtask-release-v4-plan.md`` ("fix only in xhard, original tiers still
TypeError, H2").

The golden file ``native_golden.json`` was first generated on maintenance plan BASE (``93014f27``); T12 added raise-site modules to the exception entries
and regenerated it on ``6db45911`` (``git diff 93014f27 6db45911 -- src/robomme_ood`` is empty, production code byte-identical to the original BASE);
94 of the 96 non-exception digests are identical entry by entry to the original file::

    UV_PROJECT_ENVIRONMENT=<main checkout .venv> PYTHONPATH=<worktree>/src:<worktree> \\
        uv run --no-sync python -m tests.robomme_ood.unit.hard.native_golden --write

Afterwards, if a production code change alters any value site, random stream consumption, exception type or exception raise site of the original three tiers, ``test_native_golden.py`` fails;
intentional changes must regenerate the file and state the reason in the commit message.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from robomme_ood.env_record_wrapper.hard_specs import digest

from . import offline_scene as O
from .world import World, cpu_world

NATIVE_TIERS = ("easy", "medium", "hard")
SEEDS = (101, 202)
GOLDEN = Path(__file__).with_name("native_golden.json")


def key(task: str, tier: str, seed: int) -> str:
    return f"{task}/{tier}/{seed}"


def raise_site_module(exc: BaseException) -> str:
    """``__name__`` of the module of the innermost traceback frame (distinguishes production code from test stand-ins)."""
    tb = exc.__traceback__
    if tb is None:
        return "?"
    while tb.tb_next is not None:
        tb = tb.tb_next
    return str(tb.tb_frame.f_globals.get("__name__", "?"))


def summarize(task: str, tier: str, seed: int) -> str:
    try:
        with cpu_world():
            env = O.make_offline(task, seed=seed, difficulty=tier)
            World.from_env(env)
    except Exception as exc:  # noqa: BLE001 -- the exception type and raise site are themselves the pinned behavior
        return f"error:{type(exc).__name__}@{raise_site_module(exc)}"
    doc = env._spec.to_dict()
    if doc["spec_kind"] != "native-parity/1":
        return f"kind:{doc['spec_kind']}"
    return digest(doc)


def compute() -> dict[str, str]:
    return {key(t, tier, s): summarize(t, tier, s) for t in O.ALL_TASKS for tier in NATIVE_TIERS for s in SEEDS}


def main(argv: list[str]) -> int:
    table = compute()
    if "--write" in argv:
        GOLDEN.write_text(json.dumps({
            "base_commit": "6db45911",
            "first_base_commit": "93014f27",
            "seeds": list(SEEDS),
            "tiers": list(NATIVE_TIERS),
            "digests": table,
        }, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"NATIVE_GOLDEN=WRITTEN entries={len(table)} path={GOLDEN}")
    else:
        print(json.dumps(table, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
