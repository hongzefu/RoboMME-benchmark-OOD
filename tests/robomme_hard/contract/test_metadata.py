"""L1 契约：随包元数据。

- 官方 ``src/robomme/env_metadata/{train,val,test}``：每 split 恰 16 个文件（16 任务各一）、每文件局数为钉值、
  episode 恰为 0..N−1 连续、每条记录 task 与文件名一致、seed 为整数；
- test 的 hard 子集恰为原 episode 3,7,…,47（xhard0 的来源）；
- hard 包 ``src/robomme_hard/env_metadata/train`` 只有四个 Unmask 任务、各 400 条、episode 连续，builder 的 train
  元数据路径对这四个任务指向 hard 包、其余指向官方，且经真实 builder 读到的局数与文件一致。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.robomme_hard._support.loaders import REPO
from tests.robomme_hard.contract.test_constants import (
    HARD_TRAIN_EPISODES,
    HARD_TRAIN_TASKS,
    OFFICIAL_SPLIT_EPISODES,
    OFFICIAL_SPLIT_FILES,
    TASKS,
    XHARD0_EPISODES,
)

OFFICIAL = REPO / "src" / "robomme" / "env_metadata"
HARD = REPO / "src" / "robomme_hard" / "env_metadata" / "train"


def metadata_problems(path: Path, task: str, n: int) -> list[str]:
    """一个元数据文件的问题清单（空 = 合规）。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    out = []
    if not isinstance(records, list) or len(records) != n:
        return [f"局数 {len(records) if isinstance(records, list) else None} ≠ {n}"]
    episodes = [r.get("episode") for r in records]
    if sorted(episodes) != list(range(n)):
        out.append("episode 不连续")
    if any(r.get("task", payload.get("env_id")) != task for r in records):
        out.append("task 不符")
    if any(not isinstance(r.get("seed"), int) or isinstance(r.get("seed"), bool) for r in records):
        out.append("seed 不是整数")
    return out


@pytest.mark.parametrize("split", sorted(OFFICIAL_SPLIT_EPISODES))
def test_official_split(split):
    files = sorted((OFFICIAL / split).glob("*.json"))
    assert len(files) == OFFICIAL_SPLIT_FILES
    assert {p.name for p in files} == {f"record_dataset_{t}_metadata.json" for t in TASKS}
    bad = {t: p for t in TASKS
           if (p := metadata_problems(OFFICIAL / split / f"record_dataset_{t}_metadata.json", t,
                                      OFFICIAL_SPLIT_EPISODES[split]))}
    assert bad == {}


@pytest.mark.parametrize("task", TASKS)
def test_official_test_hard_subset_is_xhard0_source(task):
    records = json.loads((OFFICIAL / "test" / f"record_dataset_{task}_metadata.json").read_text(encoding="utf-8"))
    hard = sorted(int(r["episode"]) for r in records["records"] if r["difficulty"] == "hard")
    assert tuple(hard) == XHARD0_EPISODES


def test_hard_train_metadata():
    files = sorted(HARD.glob("*.json"))
    assert {p.name for p in files} == {f"record_dataset_{t}_metadata.json" for t in HARD_TRAIN_TASKS}
    bad = {t: p for t in HARD_TRAIN_TASKS
           if (p := metadata_problems(HARD / f"record_dataset_{t}_metadata.json", t, HARD_TRAIN_EPISODES))}
    assert bad == {}


def test_builder_reads_hard_train_for_unmask_tasks_only():
    from robomme_hard.env_record_wrapper import hard_builder

    assert hard_builder.HARD_TRAIN_TASKS == frozenset(HARD_TRAIN_TASKS)
    for task in TASKS:
        builder = hard_builder.BenchmarkEnvBuilder(env_id=task, dataset="train")
        expected = HARD_TRAIN_EPISODES if task in HARD_TRAIN_TASKS else OFFICIAL_SPLIT_EPISODES["train"]
        assert builder.get_episode_num() == expected, task
        root = HARD if task in HARD_TRAIN_TASKS else OFFICIAL / "train"
        assert Path(builder._resolve_metadata_path()) == root / f"record_dataset_{task}_metadata.json"


def test_metadata_problems_negative(tmp_path):
    """判定器负例：局数不符、episode 断号、task 不符、seed 为浮点都被点名。"""
    good = {"env_id": "BinFill", "records": [{"task": "BinFill", "episode": i, "seed": i} for i in range(3)]}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert metadata_problems(path, "BinFill", 3) == []
    assert metadata_problems(path, "BinFill", 4) != []
    for mutate, name in ((lambda r: r[1].__setitem__("episode", 5), "episode 不连续"),
                         (lambda r: r[2].__setitem__("task", "StopCube"), "task 不符"),
                         (lambda r: r[0].__setitem__("seed", 1.0), "seed 不是整数")):
        bad = json.loads(json.dumps(good))
        mutate(bad["records"])
        path.write_text(json.dumps(bad), encoding="utf-8")
        assert metadata_problems(path, "BinFill", 3) == [name]
