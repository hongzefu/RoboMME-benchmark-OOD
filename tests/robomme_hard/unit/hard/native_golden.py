"""原三档（easy／medium／hard）离线导出摘要的计算与金标准生成。

摘要 = 对 16 任务 × 3 档 × 种子 ``SEEDS``，按评估链参数（runtime 四项 + seed + difficulty，不传
sampling_config 与规格，即原生分支）离线跑真实 ``_load_scene`` + 两次 ``_initialize_episode``，取导出规格文档的
``hard_specs.digest``（剔除键序影响的 canonical JSON 的 sha256）。

建场抛异常的记 ``error:<异常类名>@<抛出点模块>``，抛出点模块取 traceback 最内层帧所在模块的 ``__name__``，
用来区分异常来自生产代码（``robomme_hard.*``）还是来自测试替身（``tests.*``，如 ``offline_scene``）；
``test_native_golden.py`` 断言金标准里所有异常条目的抛出点都在 ``robomme_hard`` 内。

现有两条异常条目 ``VideoPlaceOrder/medium/101``、``VideoPlaceOrder/hard/101`` 的出处（T12 实测）：
真实 ``spawn_random_target`` 布局失败抛 RuntimeError → ``VideoPlaceOrder._load_scene`` 内
``raise _SceneGenError(...)`` 因名字 ``SceneGenerationError`` 被 ``from .utils import *`` 遮蔽成同名子模块而抛
TypeError（模块不可调用）→ 外层 ``except _SceneGenError:`` 再抛 TypeError（except 子句不是异常类），
最内层帧在 ``robomme_hard.robomme_env.VideoPlaceOrder._load_scene``，不是替身自身的错误。这是有意保留的官方现状：
见 ``src/robomme_hard/robomme_env/VideoPlaceOrder.py`` 的注释「V4 H2」（xhard 分支改用真正的异常类，原三档
行为逐字不变），以及 ``docs/plans/0922-newtask-release-v4-plan.md`` 决策表 K2（「只在 xhard 修，原三档仍
TypeError，H2」）。

金标准文件 ``native_golden.json`` 最初在维护计划 BASE（``93014f27``）上生成；T12 为异常条目补抛出点模块，
在 ``6db45911`` 上重新生成（``git diff 93014f27 6db45911 -- src/robomme_hard`` 为空，生产代码与原 BASE
逐字节相同），96 条中 94 条非异常摘要与原文件逐条相同::

    UV_PROJECT_ENVIRONMENT=<主检出 .venv> PYTHONPATH=<worktree>/src:<worktree> \\
        uv run --no-sync python -m tests.robomme_hard.unit.hard.native_golden --write

之后生产代码改动若改变了原三档的任何取值点、随机流消费、异常类型或异常抛出点，``test_native_golden.py`` 即失败；
确属有意的改动须重新生成并在提交说明里写明原因。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from robomme_hard.env_record_wrapper.hard_specs import digest

from . import offline_scene as O
from .world import World, cpu_world

NATIVE_TIERS = ("easy", "medium", "hard")
SEEDS = (101, 202)
GOLDEN = Path(__file__).with_name("native_golden.json")


def key(task: str, tier: str, seed: int) -> str:
    return f"{task}/{tier}/{seed}"


def raise_site_module(exc: BaseException) -> str:
    """异常 traceback 最内层帧所在模块的 ``__name__``（区分生产代码与测试替身）。"""
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
    except Exception as exc:  # noqa: BLE001 — 异常类型与抛出点本身就是被钉住的行为
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
