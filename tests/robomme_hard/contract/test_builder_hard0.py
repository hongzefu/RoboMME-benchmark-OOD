"""L1 契约：``BenchmarkEnvBuilder(dataset="hard-verify")`` 只含 xhard0，且不读规格根；按档步数查表已删除。

``hard-verify``（1003 评估计划 1.1）＝官方 test 元数据里每任务 ``difficulty=="hard"`` 的 12 局（原 episode
3, 7, …, 47），按原 episode 升序编为 episode 0..11；与开关 ``XHARD0_IN_TEST_HARD`` 无关。

手段同 ``test_builder_800.py``：把 ``gym.make`` 换成「记录参数后抛哨兵异常」的替身，逐局调真实的
``make_env_for_episode``，不起任何仿真。期望值一律来自手写钉值（``test_constants``）与标准库 json 直接读官方
test 元数据，不读被测代码。规格读取函数全部打桩为「调用即失败并计数」，证明 hard-verify 不读规格。

- 16 任务 × 12 局 = 192，逐局 kwargs 恰为 runtime 四项 + 官方 seed + ``difficulty="hard"``；
- ``resolve_identity`` 字段集合与取值（无 candidate／规格摘要、不带 ``specs_root``）；
- 官方 hard 子集被破坏（少一局、多一局、seed 重复）时构造即报错；
- ``specs_root`` 被拒；
- 全仓 ``src/``、``scripts/``、``tests/`` 的 .py 里不再引用 ``TIER_MAX_STEPS``（S2 待清理名单除外，见下）。
"""
from __future__ import annotations

import io
import json
import tokenize
from pathlib import Path

import pytest

from tests.robomme_hard._support.loaders import REPO
from tests.robomme_hard.contract.test_constants import (
    RUNTIME,
    TASKS,
    XHARD0,
    XHARD0_EPISODES,
    XHARD0_PER_TASK,
)

OFFICIAL_TEST = REPO / "src" / "robomme" / "env_metadata" / "test"
#: hard-verify 全量：16 任务 × 12 局（手算）
TOTAL_HARD0 = 192
N_TASKS_HARD0 = 16
#: xhard0 局 resolve_identity 的字段集合（手写）
IDENTITY_KEYS = {"episode", "tier", "candidate", "seed", "source_dataset", "source_episode", "spec_sha256", "source_run"}
#: 被打桩为「调用即失败」的规格读取入口：(模块名, 属性名)
SPEC_READERS = (
    ("hard_specs", "load_specs_root"),
    ("hard_specs", "specs_root"),
    ("hard_specs", "packaged_specs_path"),
    ("hard_builder", "_root_specs"),
    ("hard_builder", "_override_cells"),
    ("hard_builder", "_ood_entries"),
)


class Sentinel(Exception):
    """替身 gym.make 抛出的哨兵：证明构建在 gym.make 处被拦下，没有起仿真。"""


@pytest.fixture
def recorder(monkeypatch):
    from robomme_hard.env_record_wrapper import hard_builder

    calls: list[tuple[tuple, dict]] = []

    def fake_make(*args, **kwargs):
        calls.append((args, kwargs))
        raise Sentinel

    monkeypatch.setattr(hard_builder.gym, "make", fake_make)
    return calls


@pytest.fixture
def spec_reads(monkeypatch, tmp_path):
    """把规格读取入口全部换成「计数后抛错」的桩，并把规格根环境变量指向不存在的目录；返回调用记录。"""
    from robomme_hard.env_record_wrapper import hard_builder, hard_specs

    modules = {"hard_specs": hard_specs, "hard_builder": hard_builder}
    reads: list[str] = []
    for module_name, attr in SPEC_READERS:
        def boom(*_args, _name=f"{module_name}.{attr}", **_kwargs):
            reads.append(_name)
            raise AssertionError(f"hard-verify 不应读规格：调用了 {_name}")

        monkeypatch.setattr(modules[module_name], attr, boom)
    monkeypatch.setenv(hard_specs.SPECS_ROOT_ENV, str(tmp_path / "no-such-specs-root"))
    return reads


def builder_cls():
    from robomme_hard.env_record_wrapper.hard_builder import BenchmarkEnvBuilder

    return BenchmarkEnvBuilder


def official_hard(task: str) -> list[dict]:
    """标准库 json 直接读官方 test 元数据，取 hard 子集按原 episode 升序。"""
    payload = json.loads((OFFICIAL_TEST / f"record_dataset_{task}_metadata.json").read_text(encoding="utf-8"))
    return sorted((r for r in payload["records"] if r["difficulty"] == "hard"), key=lambda r: int(r["episode"]))


def capture(builder, episode: int, calls: list) -> tuple[tuple, dict]:
    before = len(calls)
    with pytest.raises(Sentinel):
        builder.make_env_for_episode(episode, max_steps=1300)
    assert len(calls) == before + 1
    return calls[-1]


@pytest.mark.parametrize("switch", [False, True], ids=["switch-off", "switch-on"])
def test_hard_verify_every_episode(recorder, spec_reads, monkeypatch, switch):
    """16 任务逐局 gym.make 参数与身份；开关 XHARD0_IN_TEST_HARD 开或关结果相同；全程不读规格。"""
    from robomme_hard.env_record_wrapper import hard_specs

    monkeypatch.setattr(hard_specs, "XHARD0_IN_TEST_HARD", switch)
    total = 0
    tasks = 0
    for task in TASKS:
        builder = builder_cls()(env_id=task, dataset="hard-verify", action_space="joint_angle", max_steps=1300)
        assert builder.dataset == "hard-verify"
        assert builder.metadata_index == {}
        assert builder.get_episode_num() == XHARD0_PER_TASK
        hard = official_hard(task)
        assert tuple(int(r["episode"]) for r in hard) == XHARD0_EPISODES
        for episode, record in enumerate(hard):
            args, kwargs = capture(builder, episode, recorder)
            assert args == (task,)
            assert kwargs == {**RUNTIME, "seed": int(record["seed"]), "difficulty": "hard"}, (task, episode)
            assert builder.resolve_episode(episode) == (int(record["seed"]), XHARD0)
            ident = builder.resolve_identity(episode)
            assert set(ident) == IDENTITY_KEYS, ident
            assert ident == {"episode": episode, "tier": XHARD0, "candidate": None, "seed": int(record["seed"]),
                             "source_dataset": "test", "source_episode": int(record["episode"]),
                             "spec_sha256": None, "source_run": None}
            total += 1
        with pytest.raises(KeyError):
            builder.resolve_episode(XHARD0_PER_TASK)
        tasks += 1
    assert spec_reads == []
    assert (tasks, total) == (N_TASKS_HARD0, TOTAL_HARD0)
    if switch:  # 两种开关各跑一遍；只在最后一遍打印判定行
        print(f"HARD0_INTERFACE=PASS tasks={tasks} per_task={XHARD0_PER_TASK} total={total} specs_reads={len(spec_reads)}")


def test_hard_verify_rejects_specs_root(spec_reads, tmp_path):
    cls = builder_cls()
    for root in (tmp_path, str(tmp_path), REPO / "src" / "robomme_hard" / "env_metadata" / "ood"):
        with pytest.raises(ValueError, match="specs_root"):
            cls(env_id="StopCube", dataset="hard-verify", specs_root=root)
    with pytest.raises(ValueError):
        cls(env_id="StopCube", dataset="hard-verify", override_metadata_path=tmp_path)
    with pytest.raises(ValueError):
        cls(env_id="NotATask", dataset="hard-verify")
    assert spec_reads == []


def _broken_loader(monkeypatch, how: str):
    """把父类读官方 test 元数据的函数包一层，在 BinFill 的 hard 子集上做 ``how`` 指定的破坏。"""
    from robomme.env_record_wrapper import episode_config_resolver as official

    original = official.load_episode_metadata

    def broken(path):
        index = dict(original(path))
        hard = sorted((key for key, rec in index.items() if key[0] == "BinFill" and rec.get("difficulty") == "hard"),
                      key=lambda key: key[1])
        if not hard:  # 别的任务的元数据文件：原样返回
            return index
        if how == "drop":
            del index[hard[-1]]
        elif how == "extra":
            easy = next(key for key, rec in index.items() if key[0] == "BinFill" and rec.get("difficulty") != "hard")
            index[easy] = dict(index[easy], difficulty="hard")
        elif how == "dup_seed":
            index[hard[1]] = dict(index[hard[1]], seed=index[hard[0]]["seed"])
        return index

    monkeypatch.setattr(official, "load_episode_metadata", broken)


@pytest.mark.parametrize("how", ["drop", "extra", "dup_seed"])
def test_hard_verify_rejects_broken_official_subset(monkeypatch, spec_reads, how):
    """官方 hard 子集必须恰为原 episode 3,7,…,47 且 seed 唯一：少一局、多一局、seed 重复都在构造时报错。"""
    _broken_loader(monkeypatch, how)
    with pytest.raises(ValueError, match="xhard0"):
        builder_cls()(env_id="BinFill", dataset="hard-verify")
    # 其他任务不受影响
    assert builder_cls()(env_id="StopCube", dataset="hard-verify").get_episode_num() == XHARD0_PER_TASK
    assert spec_reads == []


# ── 按档步数查表已删除：全仓不再引用 ─────────────────────────────────────────

#: 过渡期豁免名单。S2（评估客户端、清单、报告）已于 12.437 合入并清掉全部查表引用，名单随之清空；
#: 扫描范围即为 src/、scripts/、tests/ 下全部 .py。保留空集合只为让历史提交可读，不得再往里加文件。
S2_PENDING: frozenset[str] = frozenset()
_LOOKUP_NAME = "TIER_MAX_STEPS"


def _name_refs(path: Path) -> int:
    """文件里名为查表的标识符（NAME 记号）个数：只数代码引用，字符串与注释里的文字不算。"""
    source = path.read_text(encoding="utf-8")
    return sum(1 for tok in tokenize.generate_tokens(io.StringIO(source).readline)
               if tok.type == tokenize.NAME and tok.string == _LOOKUP_NAME)


def test_no_tier_max_steps_refs():
    refs: dict[str, int] = {}
    scanned = 0
    for top in ("src", "scripts", "tests"):
        for path in sorted((REPO / top).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if rel in S2_PENDING:
                continue
            scanned += 1
            count = _name_refs(path)
            if count:
                refs[rel] = count
    assert scanned > 0
    assert refs == {}, refs
    print(f"STEP_LOOKUP=PASS tier_max_steps_refs={sum(refs.values())}")
