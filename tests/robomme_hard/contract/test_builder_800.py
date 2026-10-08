"""L1 契约：``robomme_hard`` 的 ``BenchmarkEnvBuilder(dataset="ood")`` 对 800 局逐局交给 ``gym.make`` 的参数。

手段：把 ``gym.make`` 换成「记录位置参数与 kwargs 后抛哨兵异常」的替身，逐局调真实的 ``make_env_for_episode``，
不建任何仿真场景。期望由标准库 json 直接读包内规格得出：档序 xhard1→xhard5 主序、档内交付行（selected 且
rollout ok）按 candidate 升序拼接成 episode 0..49（hard_builder 模块文档写明的契约）。

- 默认开关关：16 任务 × 50 局 = 800，逐局 kwargs = runtime 四项 + seed + difficulty（档名）+ sampling_config
  （header 该任务）+ native_episode_spec（该行 spec），恰好这些键；
- 开关开：每任务前置 12 局 xhard0，seed 逐条等于官方 test 元数据 hard 子集（按原 episode 升序）、difficulty
  传 ``"hard"``、无回注参数，合计 16 × 62 = 992；
- 规格根覆盖（参数与环境变量）与拒绝路径（含 ``dataset="hard-verify"`` 的接受与拒绝；其逐局参数见 ``test_builder_hard0.py``）。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.robomme_hard._support.loaders import REPO, load_script
from tests.robomme_hard.contract.test_constants import (
    NEW_TIERS,
    PER_TASK,
    PER_TASK_WITH_XHARD0,
    RUNTIME,
    TASKS,
    TOTAL,
    TOTAL_WITH_XHARD0,
    XHARD0,
    XHARD0_EPISODES,
    XHARD0_PER_TASK,
)

ROOT = REPO / "src" / "robomme_hard" / "env_metadata" / "ood"
OFFICIAL_TEST = REPO / "src" / "robomme" / "env_metadata" / "test"
EXPECTED_KEYS = {*RUNTIME, "seed", "difficulty", "sampling_config", "native_episode_spec"}


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


def builder_cls():
    from robomme_hard.env_record_wrapper.hard_builder import BenchmarkEnvBuilder

    return BenchmarkEnvBuilder


def expected_rows(root: Path = ROOT) -> dict[str, list[tuple[str, dict, dict]]]:
    """{task: [(tier, header, row), ...]}：独立读规格文件，按档序与 candidate 升序排好。"""
    out: dict[str, list] = {task: [] for task in TASKS}
    for tier in NEW_TIERS:
        path = root / tier / "specs.jsonl"
        if not path.is_file():
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        header, rows = records[0], records[1:]
        chosen = [r for r in rows if r["selected"] and (r["rollout"] or {}).get("status") == "ok"]
        for row in sorted(chosen, key=lambda r: r["candidate"]):
            out[row["task"]].append((tier, header, row))
    return out


def official_hard(task: str) -> list[dict]:
    payload = json.loads((OFFICIAL_TEST / f"record_dataset_{task}_metadata.json").read_text(encoding="utf-8"))
    return sorted((r for r in payload["records"] if r["difficulty"] == "hard"), key=lambda r: int(r["episode"]))


def capture(builder, episode: int, calls: list) -> tuple[tuple, dict]:
    before = len(calls)
    with pytest.raises(Sentinel):
        builder.make_env_for_episode(episode)
    assert len(calls) == before + 1
    return calls[-1]


def newvalue_kwargs(tier: str, header: dict, row: dict, task: str) -> dict:
    return {**RUNTIME, "seed": row["seed"], "difficulty": tier,
            "sampling_config": header["sampling_config"][task], "native_episode_spec": row["spec"]}


def test_builder_800_every_episode(recorder):
    expected = expected_rows()
    total = 0
    for task in TASKS:
        builder = builder_cls()(env_id=task, dataset="ood")
        assert builder.get_episode_num() == PER_TASK == len(expected[task])
        for episode, (tier, header, row) in enumerate(expected[task]):
            args, kwargs = capture(builder, episode, recorder)
            assert args == (task,)
            assert set(kwargs) == EXPECTED_KEYS
            assert kwargs == newvalue_kwargs(tier, header, row, task), (task, episode)
            assert builder.resolve_episode(episode) == (row["seed"], tier)
            ident = builder.resolve_identity(episode)
            assert (ident["tier"], ident["candidate"], ident["seed"], ident["spec_sha256"]) == \
                (tier, row["candidate"], row["seed"], row["spec_sha256"])
            total += 1
        with pytest.raises(KeyError):
            builder.resolve_episode(PER_TASK)
    assert total == TOTAL


def _poison(obj) -> int:
    """原地改坏嵌套结构：每个 dict 加一个标记键、每个数值叶子改成 -999；返回改动处数（证明真改到了东西）。"""
    changed = 0
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                obj[key] = -999
                changed += 1
            else:
                changed += _poison(value)
        obj["__poisoned__"] = True
        changed += 1
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                obj[i] = -999
                changed += 1
            else:
                changed += _poison(value)
    return changed


def test_known_defect_builder_kwargs_aliased_to_builder_state(recorder):
    """锁定现状（已知缺陷，登记 docs/1002-pending-decisions.md D5，2026-10-04 主会话裁决本轮不修）：
    make_env_for_episode 把 builder 内部 lru_cache 里的 sampling_config 与 row["spec"] 原对象直接交给 gym.make，
    调用方原地改动第一次拿到的 kwargs 会污染第二次构建（规格侧环境内 SpecRecorder 会 deepcopy，sampling_config 无此保护）。
    正确行为应当是：每次交出副本，第二次拿到的仍等于独立读出的规格值。修复时把本用例的断言反转为 `again == expected`。"""
    builder = builder_cls()(env_id="StopCube", dataset="ood")
    tier, header, row = expected_rows()["StopCube"][0]
    expected = newvalue_kwargs(tier, header, row, "StopCube")
    _, first = capture(builder, 0, recorder)
    assert first == expected
    keys = ("sampling_config", "native_episode_spec")
    try:
        for key in keys:
            assert _poison(first[key]) > 1
        assert first["sampling_config"] != expected["sampling_config"]
        _, again = capture(builder, 0, recorder)
        # 必须在复原之前判断：共用时 again 与 first 是同一对象，finally 的原地复原会把它一起洗掉
        aliased = {key: "__poisoned__" in again[key] for key in keys}
    finally:
        # 规格读取有进程级缓存（hard_builder 的 lru_cache）：共用对象时改坏的是缓存本身，必须原地复原，
        # 否则同进程后面的用例会读到 -999（顶层对象身份不变，内容换回独立读出的副本）
        for key in keys:
            first[key].clear()
            first[key].update(json.loads(json.dumps(expected[key])))
    # 现状：两项都共用（改坏的标记出现在第二次构建的 kwargs 里）；任何一项被修成交出副本，本断言即失败，提醒同步反转。
    assert aliased == {"sampling_config": True, "native_episode_spec": True}


def test_builder_992_with_xhard0_switch(recorder, monkeypatch):
    from robomme_hard.env_record_wrapper import hard_specs

    monkeypatch.setattr(hard_specs, "XHARD0_IN_TEST_HARD", True)
    expected = expected_rows()
    total = 0
    for task in TASKS:
        builder = builder_cls()(env_id=task, dataset="ood")
        assert builder.get_episode_num() == PER_TASK_WITH_XHARD0
        hard = official_hard(task)
        assert tuple(int(r["episode"]) for r in hard) == XHARD0_EPISODES
        for episode, record in enumerate(hard):
            args, kwargs = capture(builder, episode, recorder)
            assert args == (task,)
            assert kwargs == {**RUNTIME, "seed": int(record["seed"]), "difficulty": "hard"}, (task, episode)
            ident = builder.resolve_identity(episode)
            assert (ident["tier"], ident["seed"], ident["source_episode"]) == \
                (XHARD0, int(record["seed"]), int(record["episode"]))
        for offset, (tier, header, row) in enumerate(expected[task]):
            _, kwargs = capture(builder, XHARD0_PER_TASK + offset, recorder)
            assert kwargs == newvalue_kwargs(tier, header, row, task)
        total += builder.get_episode_num()
    assert total == TOTAL_WITH_XHARD0


def test_xhard0_switch_rejects_broken_official_subset(monkeypatch):
    """开关开时 xhard0 子集必须恰为原 episode 3,7,…,47：父类读到的元数据被改坏即拒绝。"""
    from robomme_hard.env_record_wrapper import hard_builder, hard_specs

    monkeypatch.setattr(hard_specs, "XHARD0_IN_TEST_HARD", True)
    index = {}
    for r in official_hard("BinFill")[:-1]:  # 少一局
        index[("BinFill", int(r["episode"]))] = r
    with pytest.raises(ValueError):
        hard_builder._xhard0_entries("BinFill", index)


# ── 规格根覆盖与拒绝 ─────────────────────────────────────────────────────


def partial_root(tmp_path: Path, tiers=("xhard5",)) -> Path:
    root = tmp_path / "root"
    for tier in tiers:
        (root / tier).mkdir(parents=True)
        shutil.copyfile(ROOT / tier / "specs.jsonl", root / tier / "specs.jsonl")
    return root


def test_specs_root_override_param_and_env(recorder, tmp_path, monkeypatch):
    from robomme_hard.env_record_wrapper import hard_specs

    root = partial_root(tmp_path)
    expected = expected_rows(root)
    for how in ("param", "env"):
        if how == "env":
            monkeypatch.setenv(hard_specs.SPECS_ROOT_ENV, str(root))
            builder = builder_cls()(env_id="StopCube", dataset="ood")
        else:
            monkeypatch.delenv(hard_specs.SPECS_ROOT_ENV, raising=False)
            builder = builder_cls()(env_id="StopCube", dataset="ood", specs_root=root)
        assert builder.get_episode_num() == len(expected["StopCube"]) > 0
        for episode, (tier, header, row) in enumerate(expected["StopCube"]):
            _, kwargs = capture(builder, episode, recorder)
            assert kwargs == newvalue_kwargs(tier, header, row, "StopCube")
            assert builder.resolve_identity(episode)["specs_root"] == str(root.resolve())
        # 局部根里没有的任务：0 局
        assert builder_cls()(env_id="PickXtimes", dataset="ood", specs_root=root).get_episode_num() == 0


def test_rejections(tmp_path):
    from robomme_hard.env_record_wrapper import hard_specs

    cls = builder_cls()
    with pytest.raises(ValueError):
        cls(env_id="StopCube", dataset="xhard1")
    with pytest.raises(ValueError):
        cls(env_id="StopCube", dataset="ood", override_metadata_path=tmp_path)
    with pytest.raises(ValueError):
        cls(env_id="StopCube", dataset="ood", action_space="torque")
    with pytest.raises(ValueError):
        cls(env_id="NotATask", dataset="ood")
    # hard-verify：接受（每任务恰 12 局 xhard0）；拼写变体、改名前的旧名、规格根、元数据覆盖、未知任务一律拒绝
    assert cls(env_id="StopCube", dataset="hard-verify").get_episode_num() == XHARD0_PER_TASK
    legacy = tuple(load_script("eval-official/official_defs.py").LEGACY_DATASET_ALIASES)  # 旧名只在别名表里
    assert len(legacy) == 2
    for wrong in ("hard_verify", "hard-verify0", "xhard0", "Hard-Verify", "OOD", *legacy):
        with pytest.raises(ValueError):
            cls(env_id="StopCube", dataset=wrong)
    with pytest.raises(ValueError):
        cls(env_id="StopCube", dataset="hard-verify", specs_root=ROOT)
    with pytest.raises(ValueError):
        cls(env_id="StopCube", dataset="hard-verify", override_metadata_path=tmp_path)
    with pytest.raises(ValueError):
        cls(env_id="NotATask", dataset="hard-verify")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(hard_specs.SpecsError):
        cls(env_id="StopCube", dataset="ood", specs_root=empty)
    # 档文件不是 /4
    old = tmp_path / "old"
    (old / "xhard5").mkdir(parents=True)
    lines = (ROOT / "xhard5" / "specs.jsonl").read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["schema"] = "hard-specs/3"
    (old / "xhard5" / "specs.jsonl").write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(hard_specs.SpecsError):
        cls(env_id="StopCube", dataset="ood", specs_root=old)
    # 改了一行不重签：整根校验拒绝
    bad = partial_root(tmp_path / "bad")
    lines = (bad / "xhard5" / "specs.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["seed"] += 1
    lines[1] = json.dumps(row)
    (bad / "xhard5" / "specs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(hard_specs.SpecsError):
        cls(env_id="StopCube", dataset="ood", specs_root=bad)


def test_official_datasets_unchanged_except_hard_train(recorder):
    """train／test／val 仍走官方元数据（四个 Unmask 任务的 train 改读 hard 包）；gym.make 不带回注参数。"""
    cls = builder_cls()
    for dataset in ("test", "val", "train"):
        builder = cls(env_id="BinFill", dataset=dataset)
        _, kwargs = capture(builder, 0, recorder)
        assert set(kwargs) == {*RUNTIME, "seed", "difficulty"}
    hard_train = cls(env_id="VideoUnmask", dataset="train")
    assert "robomme_hard" in hard_train._resolve_metadata_path()
    assert "robomme_hard" not in cls(env_id="BinFill", dataset="train")._resolve_metadata_path()
